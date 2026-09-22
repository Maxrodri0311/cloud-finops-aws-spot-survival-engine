"""
tests/test_checkpoint_policy.py - Comprehensive Unit & Integration Tests for Fase 3.
Verifies Cox PH regularization and C-index, AWS FinOps Cost Model calculations,
Causal Dynamic Checkpoint Policy triggers, 120s emergency flush ceilings, and Monte Carlo simulator.
"""

import os
import pytest
import numpy as np
import pandas as pd
from src.domain.entities import RuntimeContext, CheckpointPolicyDecision
from src.domain.cox_model import CoxProportionalHazardsModel
from src.domain.cost_model import AWSFinOpsCostModel
from src.domain.checkpoint_policy import (
    DynamicSurvivalCheckpointPolicy,
    FixedIntervalCheckpointPolicy,
    PolicySimulator,
)
from src.core_engine import SpotAnalyticsEngine
from src.adapters.duckdb_adapter import DuckDBAnalyticalAdapter
from src.data_generator import generate_spot_lifecycle_dataset


@pytest.fixture(scope="module")
def sample_training_df():
    """Generates synthetic training data with calibrated hazard risk predictors."""
    np.random.seed(42)
    n = 1500
    price_drift = np.random.normal(0.01, 0.05, n)
    spread_ratio = np.random.uniform(0.2, 0.6, n)
    cpu_util = np.random.uniform(0.3, 0.95, n)
    gpu_util = np.random.uniform(0.0, 1.0, n)
    cluster_size = np.random.poisson(12, n) + 2

    # Linear predictor eta with clear risk separation
    eta = 3.0 * price_drift + 1.5 * spread_ratio + 0.4 * cpu_util + 0.02 * (cluster_size - 10)
    # Calibrated AWS Spot scale: 50,000s (~14h baseline characteristic lifespan)
    t_eviction = 50000.0 * ((-np.log(np.random.uniform(1e-4, 0.999, n))) / np.exp(eta)) ** (1.0 / 1.35)
    t_planned = np.random.uniform(3600.0, 36000.0, n)

    observed_durations = np.minimum(t_eviction, t_planned)
    event_observed = (t_eviction <= t_planned).astype(int)

    return pd.DataFrame({
        "duration_seconds": observed_durations,
        "event_observed": event_observed,
        "price_drift_15m": price_drift,
        "price_spread_ratio": spread_ratio,
        "cpu_utilization": cpu_util,
        "gpu_utilization": gpu_util,
        "cluster_size": cluster_size,
    })


def test_cox_ph_model_fitting_and_concordance(sample_training_df):
    model = CoxProportionalHazardsModel(penalizer=0.01)
    model.fit(sample_training_df)

    assert model.is_fitted
    assert model.concordance_index_ > 0.55, f"C-index too low: {model.concordance_index_}"
    assert model.baseline_hazard_ is not None
    assert model.baseline_cumulative_hazard_ is not None

    summary = model.summary_table()
    assert "hazard_ratio" in summary.columns
    assert "hr_lower_95" in summary.columns
    assert "hr_upper_95" in summary.columns
    assert len(summary) == 5


def test_cox_ph_instantaneous_hazard_and_survival_prediction(sample_training_df):
    model = CoxProportionalHazardsModel(penalizer=0.01)
    model.fit(sample_training_df)

    ctx = RuntimeContext(
        instance_id="i-test-01",
        job_id="job-test-01",
        pipeline_type="SageMaker_Distributed_Train",
        instance_type="g5.2xlarge",
        region="us-east-1",
        duration_seconds=3600.0,
        current_hazard=0.0001,
        seconds_since_last_checkpoint=300.0,
        compute_cost_rate_per_sec=0.0002,
    )

    hazard = model.predict_hazard(ctx)
    assert hazard > 0.0
    assert hazard <= 0.01

    surv_prob_short = model.predict_survival_probability(ctx, horizon_seconds=60.0)
    surv_prob_long = model.predict_survival_probability(ctx, horizon_seconds=3600.0)

    # Monotonicity with horizon
    assert 0.0 <= surv_prob_short <= 1.0
    assert 0.0 <= surv_prob_long <= 1.0
    assert surv_prob_short >= surv_prob_long, "Survival probability must decrease with longer horizon"


def test_aws_finops_cost_model_replay_and_io():
    cost_model = AWSFinOpsCostModel()

    ctx = RuntimeContext(
        instance_id="i-test-cost",
        job_id="job-test-cost",
        pipeline_type="EMR_Spark_ETL",
        instance_type="c6i.4xlarge",
        region="us-east-1",
        duration_seconds=7200.0,
        current_hazard=0.0002,
        seconds_since_last_checkpoint=600.0,  # 10 minutes uncommitted
        compute_cost_rate_per_sec=0.68 / 3600.0,  # ~$0.68/hr
        restart_cost_usd=0.15,
        reload_cost_usd=0.10,
    )

    marginal_replay_cost = cost_model.expected_replay_cost(ctx, marginal_only=True)
    assert 0.10 < marginal_replay_cost < 0.15

    total_replay_cost = cost_model.expected_replay_cost(ctx, marginal_only=False)
    assert 0.30 < total_replay_cost < 0.45

    io_cost = cost_model.io_checkpoint_cost(ctx)
    assert io_cost > 0.0
    assert io_cost < 0.05  # Calibrated S3 PUT + hold cost


def test_causal_dynamic_checkpoint_economic_trigger(sample_training_df):
    model = CoxProportionalHazardsModel(penalizer=0.01).fit(sample_training_df)
    cost_model = AWSFinOpsCostModel()
    policy = DynamicSurvivalCheckpointPolicy(hazard_estimator=model, cost_model=cost_model)

    # Scenario A: High uncommitted work (2400s) on high-throughput node -> save_cost < expected_loss
    ctx_high_risk = RuntimeContext(
        instance_id="i-risk-high",
        job_id="job-01",
        pipeline_type="Train",
        instance_type="g5.4xlarge",
        region="us-east-1",
        duration_seconds=3600.0,
        current_hazard=0.001,
        seconds_since_last_checkpoint=2400.0,
        compute_cost_rate_per_sec=1.624 / 3600.0,
        decision_window_seconds=300.0,
    )
    decision_high = policy.evaluate(ctx_high_risk)
    assert decision_high.action == "DYNAMIC_CHECKPOINT"
    assert decision_high.net_benefit_usd > 0.0
    assert decision_high.can_complete_before_eviction

    # Scenario B: Just saved state (5s ago) -> save_cost > expected_loss -> NO_OP
    ctx_low_risk = RuntimeContext(
        instance_id="i-risk-low",
        job_id="job-02",
        pipeline_type="Train",
        instance_type="c6i.4xlarge",
        region="us-east-1",
        duration_seconds=7200.0,
        current_hazard=0.00001,
        seconds_since_last_checkpoint=5.0,
        compute_cost_rate_per_sec=0.00005,
        decision_window_seconds=60.0,
    )
    decision_low = policy.evaluate(ctx_low_risk)
    assert decision_low.action == "NO_OP"


def test_emergency_interruption_notice_120s_ceiling(sample_training_df):
    model = CoxProportionalHazardsModel(penalizer=0.01).fit(sample_training_df)
    cost_model = AWSFinOpsCostModel()
    policy = DynamicSurvivalCheckpointPolicy(
        hazard_estimator=model,
        cost_model=cost_model,
        emergency_margin_seconds=20.0,
    )

    # Case 1: Fast checkpoint (15s) with 120s notice -> JIT_EMERGENCY_FLUSH succeeds
    ctx_emergency_ok = RuntimeContext(
        instance_id="i-notice-ok",
        job_id="job-notice-1",
        pipeline_type="Train",
        instance_type="c6i.4xlarge",
        region="us-east-1",
        duration_seconds=5000.0,
        current_hazard=0.001,
        seconds_since_last_checkpoint=400.0,
        compute_cost_rate_per_sec=0.0002,
        interruption_notice=True,
        remaining_seconds_notice=120.0,
        estimated_checkpoint_duration_sec=15.0,  # 15s < (120 - 20)
    )
    dec_ok = policy.evaluate(ctx_emergency_ok)
    assert dec_ok.action == "JIT_EMERGENCY_FLUSH"
    assert dec_ok.can_complete_before_eviction

    # Case 2: Slow checkpoint (110s) with 120s notice -> 110s >= (120 - 20) -> MIGRATE_ON_DEMAND
    ctx_emergency_fail = RuntimeContext(
        instance_id="i-notice-fail",
        job_id="job-notice-2",
        pipeline_type="Train",
        instance_type="g5.4xlarge",
        region="us-east-1",
        duration_seconds=5000.0,
        current_hazard=0.001,
        seconds_since_last_checkpoint=400.0,
        compute_cost_rate_per_sec=0.0005,
        interruption_notice=True,
        remaining_seconds_notice=120.0,
        estimated_checkpoint_duration_sec=110.0,  # exceeds safety ceiling!
    )
    dec_fail = policy.evaluate(ctx_emergency_fail)
    assert dec_fail.action == "MIGRATE_ON_DEMAND"
    assert not dec_fail.can_complete_before_eviction


def test_rebalance_recommendation_trigger(sample_training_df):
    model = CoxProportionalHazardsModel(penalizer=0.01).fit(sample_training_df)
    cost_model = AWSFinOpsCostModel()
    policy = DynamicSurvivalCheckpointPolicy(hazard_estimator=model, cost_model=cost_model)

    ctx_rebalance = RuntimeContext(
        instance_id="i-reb",
        job_id="job-reb",
        pipeline_type="Train",
        instance_type="c6i.4xlarge",
        region="us-east-1",
        duration_seconds=4000.0,
        current_hazard=0.0002,
        seconds_since_last_checkpoint=400.0,  # >= 300s
        compute_cost_rate_per_sec=0.0002,
        rebalance_recommended=True,
    )
    decision = policy.evaluate(ctx_rebalance)
    assert decision.action == "DYNAMIC_CHECKPOINT"
    assert "Rebalance Recommendation" in decision.reason


def test_policy_simulator_monte_carlo_superiority(sample_training_df):
    model = CoxProportionalHazardsModel(penalizer=0.01).fit(sample_training_df)
    cost_model = AWSFinOpsCostModel()
    dyn_policy = DynamicSurvivalCheckpointPolicy(hazard_estimator=model, cost_model=cost_model)
    fixed_policy = FixedIntervalCheckpointPolicy(interval_seconds=900.0)

    np.random.seed(42)
    n_w = 10
    durs = np.random.uniform(3600.0, 14400.0, n_w)
    evs = np.random.binomial(1, 0.4, n_w).astype(bool)
    spots = np.random.uniform(0.50, 1.20, n_w)
    ondemands = spots * np.random.uniform(2.5, 3.5, n_w)
    workloads = [
        {
            "duration_seconds": d,
            "event_observed": ev,
            "spot_price_usd_per_hour": sp,
            "ondemand_price_usd_per_hour": od,
            "hazard": 0.00006,
            "rebalance_recommended": ev,
        }
        for d, ev, sp, od in zip(durs, evs, spots, ondemands)
    ]

    bench_df = PolicySimulator.simulate_fleet(workloads, dyn_policy, fixed_policy)
    assert len(bench_df) == 3
    assert "Policy" in bench_df.columns
    assert "Total Financial Cost ($)" in bench_df.columns
    assert "Net Savings ($)" in bench_df.columns
    assert "Checkpoint ROI (%)" in bench_df.columns

    # Dynamic Survival Policy must have lower or equal total cost compared to Fixed Interval
    dyn_row = bench_df[bench_df["Policy"] == "Dynamic Survival (Ours)"].iloc[0]
    fixed_row = bench_df[bench_df["Policy"] == "Fixed-Interval (15m)"].iloc[0]
    assert dyn_row["Total Financial Cost ($)"] <= fixed_row["Total Financial Cost ($)"]


def test_core_engine_fase3_end_to_end_integration(tmp_path):
    lake_dir = str(tmp_path / "lake_test_fase3")
    generate_spot_lifecycle_dataset(num_records=1200, base_output_dir=lake_dir, random_seed=42)

    adapter = DuckDBAnalyticalAdapter()
    engine = SpotAnalyticsEngine(storage=adapter, lake_path=lake_dir)

    # 1. Train Cox model through engine
    cox = engine.train_cox_model(sample_limit=1000)
    assert cox.is_fitted
    assert cox.concordance_index_ > 0.55

    # 2. Live decision evaluation
    ctx = RuntimeContext(
        instance_id="i-live-01",
        job_id="job-live-01",
        pipeline_type="SageMaker_Distributed_Train",
        instance_type="c6i.4xlarge",
        region="us-east-1",
        duration_seconds=3600.0,
        current_hazard=0.0005,
        seconds_since_last_checkpoint=1200.0,
        compute_cost_rate_per_sec=0.0003,
    )
    decision = engine.evaluate_live_decision(context=ctx, trained_model=cox)
    assert isinstance(decision, CheckpointPolicyDecision)
    assert decision.action in ["DYNAMIC_CHECKPOINT", "NO_OP"]

    # 3. Simulate policy benchmark
    bench = engine.simulate_policy_benchmark(sample_size=300, trained_model=cox)
    assert len(bench) == 3
    assert "Net Savings (%)" in bench.columns
