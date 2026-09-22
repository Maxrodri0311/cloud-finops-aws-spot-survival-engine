"""
src/core_engine.py - Core Analytical & Survival Engine for Coderio AWS Spot Lifecycle.
Implements dependency injection (DIP) over AnalyticalStorageProtocol,
orchestrating vectorized analytics on Hive-partitioned Parquet data lakes
and non-parametric survival analysis (Kaplan-Meier, Greenwood, Log-Rank).
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd

# Path resolution for standalone script invocation
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.domain.contracts import AnalyticalStorageProtocol
from src.domain.survival_models import (
    KaplanMeierEstimator,
    LogRankResult,
    log_rank_test,
)
from src.adapters.duckdb_adapter import DuckDBAnalyticalAdapter


class SpotAnalyticsEngine:
    """
    Decoupled Analytics Engine orchestrating queries over the Spot Event Lake.
    Depends strictly on the AnalyticalStorageProtocol abstraction (Anti-Buried Dependencies).
    """

    def __init__(
        self,
        storage: AnalyticalStorageProtocol,
        lake_path: str = "data/spot_events",
    ):
        self.storage = storage
        self.lake_path = lake_path

    def compute_fleet_summary(self) -> pd.DataFrame:
        """
        Executes a federated aggregate query across all regional partitions
        emulating AWS Athena partition projection with sub-50ms latency.
        """
        parquet_glob = os.path.join(self.lake_path, "**", "*.parquet")
        query = f"""
            SELECT 
                region,
                instance_family,
                COUNT(*) as total_tasks,
                ROUND(AVG(duration_seconds), 2) as mean_duration_sec,
                ROUND(AVG(CASE WHEN event_observed THEN 1.0 ELSE 0.0 END), 4) as eviction_rate,
                ROUND(AVG(ondemand_price_usd_per_hour), 4) as avg_ondemand_price,
                ROUND(AVG(spot_price_usd_per_hour), 4) as avg_spot_price,
                ROUND(AVG(ondemand_price_usd_per_hour - spot_price_usd_per_hour), 4) as avg_hourly_savings_usd,
                ROUND(SUM((duration_seconds / 3600.0) * (ondemand_price_usd_per_hour - spot_price_usd_per_hour)), 2) as cumulative_gross_savings_usd
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            GROUP BY region, instance_family
            ORDER BY cumulative_gross_savings_usd DESC;
        """
        return self.storage.execute_query(query)

    def fit_pool_survival(
        self,
        instance_family: str,
        region: Optional[str] = None,
    ) -> KaplanMeierEstimator:
        """
        Extracts duration and censoring observations for a specific instance pool
        and fits the non-parametric Kaplan-Meier survival estimator with Greenwood bands.
        """
        parquet_glob = os.path.join(self.lake_path, "**", "*.parquet")
        where_cond = f"WHERE instance_family = '{instance_family}'"
        if region:
            where_cond += f" AND region = '{region}'"

        query = f"""
            SELECT duration_seconds, event_observed
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            {where_cond};
        """
        df = self.storage.execute_query(query)
        if len(df) == 0:
            raise ValueError(f"No exposure data found for pool: family={instance_family}, region={region}")

        km = KaplanMeierEstimator()
        km.fit(df["duration_seconds"].values, df["event_observed"].values)
        return km

    def compare_instance_pools(
        self,
        family_1: str,
        family_2: str,
        region: Optional[str] = None,
    ) -> LogRankResult:
        """
        Executes a two-sample Log-Rank hypothesis test comparing two instance pools.
        H0: S_1(t) = S_2(t) (Equality of eviction risk).
        """
        parquet_glob = os.path.join(self.lake_path, "**", "*.parquet")
        region_filter = f"AND region = '{region}'" if region else ""

        q1 = f"""
            SELECT duration_seconds, event_observed
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            WHERE instance_family = '{family_1}' {region_filter};
        """
        q2 = f"""
            SELECT duration_seconds, event_observed
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            WHERE instance_family = '{family_2}' {region_filter};
        """
        df1 = self.storage.execute_query(q1)
        df2 = self.storage.execute_query(q2)

        return log_rank_test(
            durations_1=df1["duration_seconds"].values,
            events_1=df1["event_observed"].values,
            durations_2=df2["duration_seconds"].values,
            events_2=df2["event_observed"].values,
            name1=family_1,
            name2=family_2,
        )

    def compute_actuarial_mtbf_comparison(self) -> pd.DataFrame:
        """
        Computes the true actuarial MTBF integral vs naive sample mean
        demonstrating the censoring bias across instance families.
        """
        parquet_glob = os.path.join(self.lake_path, "**", "*.parquet")
        query = f"""
            SELECT DISTINCT instance_family
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            ORDER BY instance_family;
        """
        families = self.storage.execute_query(query)["instance_family"].tolist()

        rows = []
        for fam in families:
            km = self.fit_pool_survival(fam)
            mtbf_integral = km.calculate_mtbf()

            # Query naive mean
            q_naive = f"""
                SELECT 
                    COUNT(*) as total_tasks,
                    ROUND(AVG(duration_seconds), 2) as naive_mean_sec,
                    ROUND(AVG(CASE WHEN event_observed THEN 1.0 ELSE 0.0 END), 4) as eviction_rate
                FROM read_parquet('{parquet_glob}', hive_partitioning = true)
                WHERE instance_family = '{fam}';
            """
            naive_df = self.storage.execute_query(q_naive)
            naive_mean = float(naive_df["naive_mean_sec"].iloc[0])
            eviction_rate = float(naive_df["eviction_rate"].iloc[0])
            total_tasks = int(naive_df["total_tasks"].iloc[0])

            # Bias percentage
            bias_pct = ((naive_mean - mtbf_integral) / mtbf_integral) * 100.0

            rows.append({
                "instance_family": fam,
                "total_tasks": total_tasks,
                "eviction_rate": eviction_rate,
                "naive_mean_sec": round(naive_mean, 2),
                "actuarial_mtbf_sec": round(mtbf_integral, 2),
                "actuarial_mtbf_hours": round(mtbf_integral / 3600.0, 2),
                "censoring_bias_pct": round(bias_pct, 2),
            })

        return pd.DataFrame(rows)


def create_engine(lake_path: str = "data/spot_events") -> SpotAnalyticsEngine:
    """Composition Root: Injects DuckDB adapter into domain engine."""
    adapter = DuckDBAnalyticalAdapter()
    return SpotAnalyticsEngine(storage=adapter, lake_path=lake_path)


if __name__ == "__main__":
    from src.data_generator import generate_spot_lifecycle_dataset

    data_dir = "data/spot_events"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print("[Core Engine] Generating bootstrap dataset (10,000 records)...")
        generate_spot_lifecycle_dataset(num_records=10000, base_output_dir=data_dir)

    engine = create_engine(lake_path=data_dir)
    print("\n" + "="*85)
    print("  CODERIO AWS SPOT LIFECYCLE - FLEET SUMMARY & FINANCIALS")
    print("="*85)
    summary = engine.compute_fleet_summary()
    print(summary.to_string(index=False))

    print("\n" + "="*85)
    print("  ACTUARIAL MTBF vs NAIVE SAMPLE MEAN (Censoring Bias Audit)")
    print("="*85)
    mtbf_df = engine.compute_actuarial_mtbf_comparison()
    print(mtbf_df.to_string(index=False))

    print("\n" + "="*85)
    print("  LOG-RANK HYPOTHESIS TEST: Compute-Optimized (c6i) vs GPU-Accelerated (g5)")
    print("="*85)
    lr_res = engine.compare_instance_pools("c6i", "g5")
    print(f"  Group 1 ({lr_res.group1_name}): Observed={lr_res.observed_events_1}, Expected={lr_res.expected_events_1}")
    print(f"  Group 2 ({lr_res.group2_name}): Observed={lr_res.observed_events_2}, Expected={lr_res.expected_events_2}")
    print(f"  Chi-Square Statistic: {lr_res.chi2_statistic:.4f} | p-value: {lr_res.p_value:.6e}")
    print(f"  Statistically Significant Difference (p < 0.05): {lr_res.is_significant}")
    print("="*85 + "\n")
