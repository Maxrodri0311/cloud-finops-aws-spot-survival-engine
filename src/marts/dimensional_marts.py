"""
src/marts/dimensional_marts.py - Kimball Star-Schema Semantic Layer & FinOps Data Marts.
Orchestrates dimensional modeling (Star Schema) and executive analytical aggregation marts
over Hive-partitioned Spot Event Data Lakes using AnalyticalStorageProtocol.
"""

import os
from typing import Optional, Dict, Any, List
import pandas as pd
from src.domain.contracts import AnalyticalStorageProtocol


class KimballLakehouseMarts:
    """
    Semantic Layer implementing Kimball Star-Schema Data Marts:
    - Dimensions: dim_instance_pool, dim_pipeline
    - Facts: fact_spot_exposure, fact_interruption_event, fact_checkpoint_event
    - Executive Marts: mart_spot_finops_summary, mart_hazard_by_pool, mart_signal_efficiency
    """

    def __init__(
        self,
        storage: AnalyticalStorageProtocol,
        lake_path: str = "data/spot_events",
    ):
        self.storage = storage
        self.lake_path = lake_path
        self.is_built: bool = False

    def build_dimensions_and_facts(self) -> None:
        """
        Builds the Kimball Star-Schema views over partitioned Parquet files.
        Executes sub-second virtual DDL via AnalyticalStorageProtocol.
        """
        parquet_glob = os.path.join(self.lake_path, "**", "*.parquet").replace("\\", "/")

        # 1. Dimension: dim_instance_pool
        q_dim_pool = f"""
            CREATE OR REPLACE VIEW dim_instance_pool AS
            SELECT DISTINCT
                md5(concat(region, '_', availability_zone, '_', instance_family, '_', instance_type)) as pool_id,
                region,
                availability_zone as az,
                instance_family,
                instance_type,
                vcpu as vcpus,
                memory_gib,
                CASE WHEN instance_family = 'g5' THEN true ELSE false END as has_gpu,
                ondemand_price_usd_per_hour
            FROM read_parquet('{parquet_glob}', hive_partitioning = true);
        """
        self.storage.execute_query(q_dim_pool)

        # 2. Dimension: dim_pipeline
        q_dim_pipeline = f"""
            CREATE OR REPLACE VIEW dim_pipeline AS
            SELECT DISTINCT
                md5(pipeline_type) as pipeline_id,
                pipeline_type
            FROM read_parquet('{parquet_glob}', hive_partitioning = true);
        """
        self.storage.execute_query(q_dim_pipeline)

        # 3. Fact: fact_spot_exposure
        q_fact_exposure = f"""
            CREATE OR REPLACE VIEW fact_spot_exposure AS
            SELECT
                event_id as exposure_id,
                job_id,
                instance_id,
                md5(concat(region, '_', availability_zone, '_', instance_family, '_', instance_type)) as pool_id,
                md5(pipeline_type) as pipeline_id,
                launch_ts as launch_time,
                terminal_ts as terminal_time,
                duration_seconds,
                event_observed,
                event_type,
                ondemand_price_usd_per_hour,
                spot_price_usd_per_hour,
                price_spread_ratio,
                price_drift_15m,
                cpu_utilization,
                gpu_utilization,
                cluster_size,
                rebalance_recommended,
                interruption_notice,
                checkpoint_age_seconds,
                checkpoint_size_bytes,
                replay_cost_usd,
                ROUND((duration_seconds / 3600.0) * ondemand_price_usd_per_hour, 4) as ondemand_gross_cost_usd,
                ROUND((duration_seconds / 3600.0) * spot_price_usd_per_hour, 4) as spot_compute_cost_usd,
                ROUND((duration_seconds / 3600.0) * (ondemand_price_usd_per_hour - spot_price_usd_per_hour), 4) as gross_savings_usd,
                year,
                month
            FROM read_parquet('{parquet_glob}', hive_partitioning = true);
        """
        self.storage.execute_query(q_fact_exposure)

        # 4. Fact: fact_interruption_event
        q_fact_interruption = """
            CREATE OR REPLACE VIEW fact_interruption_event AS
            SELECT
                concat('int-', exposure_id) as interruption_id,
                exposure_id,
                pool_id,
                terminal_time as eviction_timestamp,
                rebalance_recommended as had_rebalance_notice,
                interruption_notice as had_120s_notice,
                replay_cost_usd as sunk_replay_loss_usd,
                spot_compute_cost_usd as compute_burned_usd
            FROM fact_spot_exposure
            WHERE event_observed = true;
        """
        self.storage.execute_query(q_fact_interruption)

        # 5. Fact: fact_checkpoint_event
        q_fact_checkpoint = """
            CREATE OR REPLACE VIEW fact_checkpoint_event AS
            SELECT
                concat('chk-', exposure_id) as checkpoint_id,
                exposure_id,
                pool_id,
                terminal_time as checkpoint_timestamp,
                checkpoint_size_bytes,
                ROUND(checkpoint_size_bytes / (1024.0 * 1024.0 * 1024.0), 3) as checkpoint_size_gib,
                checkpoint_age_seconds,
                0.02 as estimated_io_cost_usd
            FROM fact_spot_exposure
            WHERE checkpoint_age_seconds > 0;
        """
        self.storage.execute_query(q_fact_checkpoint)

        self.is_built = True

    def query_finops_executive_mart(self) -> pd.DataFrame:
        """
        Queries the C-Level FinOps Executive Mart:
        Gross Savings, Sunk Replay Losses, Realized Net Savings, and Margins.
        """
        if not self.is_built:
            self.build_dimensions_and_facts()

        query = """
            SELECT
                COUNT(*) as total_tasks_executed,
                ROUND(SUM(ondemand_gross_cost_usd), 2) as total_ondemand_benchmark_usd,
                ROUND(SUM(spot_compute_cost_usd), 2) as total_spot_compute_usd,
                ROUND(SUM(gross_savings_usd), 2) as total_gross_savings_usd,
                ROUND(AVG(gross_savings_usd / NULLIF(ondemand_gross_cost_usd, 0)) * 100.0, 2) as avg_spot_discount_pct,
                COUNT(CASE WHEN event_observed THEN 1 END) as total_evictions,
                ROUND(SUM(replay_cost_usd), 2) as total_replay_loss_usd,
                ROUND(SUM(gross_savings_usd) - SUM(replay_cost_usd), 2) as net_realized_savings_usd,
                ROUND(((SUM(gross_savings_usd) - SUM(replay_cost_usd)) / NULLIF(SUM(ondemand_gross_cost_usd), 0)) * 100.0, 2) as net_savings_pct
            FROM fact_spot_exposure;
        """
        return self.storage.execute_query(query)

    def query_pool_hazard_mart(self) -> pd.DataFrame:
        """
        Queries dimensional hazard metrics broken down by instance family and region.
        """
        if not self.is_built:
            self.build_dimensions_and_facts()

        query = """
            SELECT
                p.instance_family,
                p.region,
                COUNT(f.exposure_id) as total_tasks,
                ROUND(AVG(f.duration_seconds / 3600.0), 2) as mean_exposure_hours,
                ROUND(AVG(CASE WHEN f.event_observed THEN 1.0 ELSE 0.0 END) * 100.0, 2) as eviction_rate_pct,
                ROUND(AVG(f.price_spread_ratio) * 100.0, 2) as avg_price_spread_pct,
                ROUND(AVG(f.spot_price_usd_per_hour), 4) as avg_spot_rate_usd,
                ROUND(SUM(f.gross_savings_usd), 2) as cumulative_gross_savings_usd
            FROM fact_spot_exposure f
            JOIN dim_instance_pool p ON f.pool_id = p.pool_id
            GROUP BY p.instance_family, p.region
            ORDER BY p.instance_family, cumulative_gross_savings_usd DESC;
        """
        return self.storage.execute_query(query)

    def query_signal_efficiency_mart(self) -> pd.DataFrame:
        """
        Queries early warning AWS telemetry:
        Coverage of 2-minute interruption notice and Rebalance recommendations.
        """
        if not self.is_built:
            self.build_dimensions_and_facts()

        query = """
            SELECT
                p.instance_family,
                COUNT(f.exposure_id) as total_evictions,
                COUNT(CASE WHEN f.rebalance_recommended THEN 1 END) as evictions_with_rebalance_notice,
                ROUND(AVG(CASE WHEN f.rebalance_recommended THEN 1.0 ELSE 0.0 END) * 100.0, 2) as rebalance_coverage_pct,
                COUNT(CASE WHEN f.interruption_notice THEN 1 END) as evictions_with_120s_notice,
                ROUND(SUM(f.replay_cost_usd), 2) as total_replay_sunk_loss_usd
            FROM fact_spot_exposure f
            JOIN dim_instance_pool p ON f.pool_id = p.pool_id
            WHERE f.event_observed = true
            GROUP BY p.instance_family
            ORDER BY total_evictions DESC;
        """
        return self.storage.execute_query(query)

    def export_marts_to_parquet(self, output_dir: str = "data/marts") -> Dict[str, str]:
        """
        Materializes Star-Schema dimensional tables into local Parquet files.
        """
        if not self.is_built:
            self.build_dimensions_and_facts()

        os.makedirs(output_dir, exist_ok=True)
        tables = [
            "dim_instance_pool",
            "dim_pipeline",
            "fact_spot_exposure",
            "fact_interruption_event",
            "fact_checkpoint_event",
        ]
        exported_paths = {}

        for tbl in tables:
            path = os.path.join(output_dir, f"{tbl}.parquet").replace("\\", "/")
            query = f"COPY (SELECT * FROM {tbl}) TO '{path}' (FORMAT PARQUET);"
            self.storage.execute_query(query)
            exported_paths[tbl] = path

        return exported_paths
