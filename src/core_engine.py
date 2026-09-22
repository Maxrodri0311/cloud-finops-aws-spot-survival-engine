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
from src.domain.entities import RuntimeContext, CheckpointPolicyDecision
from src.domain.survival_models import (
    KaplanMeierEstimator,
    LogRankResult,
    log_rank_test,
)
from src.domain.cox_model import CoxProportionalHazardsModel
from src.domain.cost_model import AWSFinOpsCostModel
from src.domain.checkpoint_policy import (
    CheckpointPolicyProtocol,
    DynamicSurvivalCheckpointPolicy,
    FixedIntervalCheckpointPolicy,
    PolicySimulator,
)
from src.marts.dimensional_marts import KimballLakehouseMarts
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
        self.marts = KimballLakehouseMarts(storage=self.storage, lake_path=self.lake_path)

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

    def train_cox_model(
        self,
        feature_cols: Optional[List[str]] = None,
        penalizer: float = 0.01,
        sample_limit: int = 25000,
    ) -> CoxProportionalHazardsModel:
        """
        Loads training exposure data from the data lake and fits a regularized
        multivariate Cox Proportional Hazards model with Breslow baseline hazard.
        """
        if feature_cols is None:
            feature_cols = [
                "price_drift_15m",
                "price_spread_ratio",
                "cpu_utilization",
                "gpu_utilization",
                "cluster_size",
            ]

        cols_select = ", ".join(["duration_seconds", "event_observed"] + feature_cols)
        parquet_glob = os.path.join(self.lake_path, "**", "*.parquet")
        query = f"""
            SELECT {cols_select}
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            LIMIT {sample_limit};
        """
        df = self.storage.execute_query(query)
        if len(df) == 0:
            raise ValueError(f"No exposure data found in lake: {self.lake_path}")

        model = CoxProportionalHazardsModel(penalizer=penalizer)
        model.fit(
            df=df,
            duration_col="duration_seconds",
            event_col="event_observed",
            feature_cols=feature_cols,
        )
        return model

    def evaluate_live_decision(
        self,
        context: RuntimeContext,
        policy: Optional[CheckpointPolicyProtocol] = None,
        trained_model: Optional[CoxProportionalHazardsModel] = None,
    ) -> CheckpointPolicyDecision:
        """
        Evaluates a real-time checkpointing decision for an active Spot task.
        If no policy is passed, builds a DynamicSurvivalCheckpointPolicy
        using the provided trained_model (or fits a fresh Cox PH model)
        and an AWSFinOpsCostModel.
        """
        if policy is None:
            if trained_model is None:
                trained_model = self.train_cox_model(sample_limit=10000)
            cost_model = AWSFinOpsCostModel()
            policy = DynamicSurvivalCheckpointPolicy(
                hazard_estimator=trained_model,
                cost_model=cost_model,
            )
        return policy.evaluate(context)

    def simulate_policy_benchmark(
        self,
        sample_size: int = 1000,
        trained_model: Optional[CoxProportionalHazardsModel] = None,
    ) -> pd.DataFrame:
        """
        Runs Monte Carlo evaluation benchmark across actual spot workloads comparing:
        1. On-Demand Pure Baseline
        2. Fixed-Interval Policy (15m periodic)
        3. Dynamic Survival Causal Policy (Ours)
        """
        parquet_glob = os.path.join(self.lake_path, "**", "*.parquet")
        query = f"""
            SELECT 
                duration_seconds,
                event_observed,
                spot_price_usd_per_hour,
                ondemand_price_usd_per_hour,
                rebalance_recommended,
                instance_type,
                price_drift_15m,
                price_spread_ratio,
                cpu_utilization,
                gpu_utilization,
                cluster_size
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            LIMIT {sample_size};
        """
        df = self.storage.execute_query(query)
        if len(df) == 0:
            raise ValueError("No workloads available for policy benchmark simulation.")

        if trained_model is None:
            trained_model = CoxProportionalHazardsModel(penalizer=0.01)
            trained_model.fit(df)

        cost_model = AWSFinOpsCostModel()
        dynamic_policy = DynamicSurvivalCheckpointPolicy(
            hazard_estimator=trained_model,
            cost_model=cost_model,
        )
        fixed_policy = FixedIntervalCheckpointPolicy(interval_seconds=900.0)

        workloads = df.to_dict(orient="records")
        return PolicySimulator.simulate_fleet(
            workloads=workloads,
            dynamic_policy=dynamic_policy,
            fixed_policy=fixed_policy,
        )

    def build_marts(self) -> None:
        """Constructs Kimball Star-Schema dimensional views."""
        self.marts.build_dimensions_and_facts()

    def query_finops_executive_mart(self) -> pd.DataFrame:
        """Queries C-level executive FinOps mart."""
        return self.marts.query_finops_executive_mart()

    def query_pool_hazard_mart(self) -> pd.DataFrame:
        """Queries granular hazard breakdown by instance pool."""
        return self.marts.query_pool_hazard_mart()

    def query_signal_efficiency_mart(self) -> pd.DataFrame:
        """Queries AWS early warning signal efficiency mart."""
        return self.marts.query_signal_efficiency_mart()

    def export_dimensional_marts(self, output_dir: str = "data/marts") -> Dict[str, str]:
        """Materializes Star Schema dimensional models into Parquet files."""
        return self.marts.export_marts_to_parquet(output_dir=output_dir)


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

    print("\n" + "="*85)
    print("  MULTIVARIATE COX PROPORTIONAL HAZARDS MODEL (Regularized Partial Likelihood)")
    print("="*85)
    cox = engine.train_cox_model(sample_limit=10000)
    print(f"  Concordance Index (C-Index): {cox.concordance_index_:.4f}")
    print("\n  Covariate Hazard Ratios (HR = exp(beta)):")
    print(cox.summary_table().to_string())

    print("\n" + "="*85)
    print("  DYNAMIC CHECKPOINT POLICY BENCHMARK (Monte Carlo Fleet Simulation)")
    print("="*85)
    bench_df = engine.simulate_policy_benchmark(sample_size=1000, trained_model=cox)
    print(bench_df.to_string(index=False))

    print("\n" + "="*85)
    print("  KIMBALL STAR-SCHEMA DATA MARTS: FINOPS EXECUTIVE MART")
    print("="*85)
    engine.build_marts()
    finops_mart = engine.query_finops_executive_mart()
    print(finops_mart.to_string(index=False))

    print("\n" + "="*85)
    print("  KIMBALL STAR-SCHEMA DATA MARTS: AWS EARLY WARNING EFFICIENCY")
    print("="*85)
    signals_mart = engine.query_signal_efficiency_mart()
    print(signals_mart.to_string(index=False))
    print("="*85 + "\n")
