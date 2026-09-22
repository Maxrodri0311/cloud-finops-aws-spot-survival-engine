"""
tests/test_survival_models.py - Comprehensive Unit Tests for Actuarial & Survival Models.
Verifies Kaplan-Meier product-limit properties, Greenwood log-log confidence bounds,
right-censoring behavior, MTBF numerical integration, and Log-Rank hypothesis testing.
"""

import pytest
import numpy as np
import pandas as pd
from src.domain.survival_models import (
    KaplanMeierEstimator,
    LogRankResult,
    log_rank_test,
)
from src.core_engine import SpotAnalyticsEngine
from src.adapters.duckdb_adapter import DuckDBAnalyticalAdapter
from src.data_generator import generate_spot_lifecycle_dataset


@pytest.fixture(scope="module")
def synthetic_survival_data():
    np.random.seed(42)
    n = 1000
    # True event times ~ Weibull(shape=1.5, scale=5000)
    t_event = 5000.0 * np.random.weibull(1.5, n)
    # Censoring times ~ Uniform(1000, 8000)
    t_censor = np.random.uniform(1000.0, 8000.0, n)
    
    observed_durations = np.minimum(t_event, t_censor)
    event_observed = (t_event <= t_censor)
    return observed_durations, event_observed


def test_kaplan_meier_monotonicity_and_bounds(synthetic_survival_data):
    durations, events = synthetic_survival_data
    km = KaplanMeierEstimator()
    km.fit(durations, events)

    assert km.is_fitted_
    assert km.survival_table_ is not None

    table = km.survival_table_
    surv_probs = table["survival_prob"].values

    # Bounds check [0, 1]
    assert surv_probs[0] == 1.0
    assert (surv_probs >= 0.0).all()
    assert (surv_probs <= 1.0).all()

    # Monotonicity check: S(t) must be non-increasing
    diffs = np.diff(surv_probs)
    assert (diffs <= 1e-9).all(), "Survival probability S(t) must be monotonically non-increasing"


def test_greenwood_log_log_confidence_intervals(synthetic_survival_data):
    durations, events = synthetic_survival_data
    km = KaplanMeierEstimator()
    km.fit(durations, events)

    table = km.survival_table_
    ci_lower = table["ci_lower_95"].values
    ci_upper = table["ci_upper_95"].values
    surv_probs = table["survival_prob"].values

    # Invariant: 0 <= CI_lower <= S(t) <= CI_upper <= 1
    assert (ci_lower >= 0.0).all()
    assert (ci_upper <= 1.0).all()
    assert (ci_lower <= surv_probs + 1e-9).all()
    assert (surv_probs <= ci_upper + 1e-9).all()


def test_right_censoring_impact_on_survival_curve():
    """
    Validates that right-censoring preserves higher survival probability
    compared to treating censored units as failures.
    """
    durations = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    
    # Scenario A: 100% evictions
    events_all = np.array([True, True, True, True, True])
    km_all = KaplanMeierEstimator().fit(durations, events_all)
    
    # Scenario B: 60% evictions, 40% censored
    events_censored = np.array([True, False, True, False, True])
    km_cens = KaplanMeierEstimator().fit(durations, events_censored)

    # At t = 300, censored scenario must have higher survival
    surv_a = km_all.predict_survival(300.0)
    surv_b = km_cens.predict_survival(300.0)

    assert surv_b > surv_a, "Censored dataset must have strictly higher survival probability"


def test_actuarial_life_table_generation(synthetic_survival_data):
    durations, events = synthetic_survival_data
    km = KaplanMeierEstimator().fit(durations, events)
    life_table = km.actuarial_life_table(bin_width_seconds=1800.0)

    assert len(life_table) > 0
    assert "interval_hours" in life_table.columns
    assert "effective_exposure" in life_table.columns
    assert "conditional_eviction_prob" in life_table.columns
    assert "cumulative_survival" in life_table.columns

    # Probabilities in [0, 1]
    probs = life_table["conditional_eviction_prob"].values
    assert (probs >= 0.0).all()
    assert (probs <= 1.0).all()


def test_mtbf_integral_computation(synthetic_survival_data):
    durations, events = synthetic_survival_data
    km = KaplanMeierEstimator().fit(durations, events)
    mtbf = km.calculate_mtbf()

    assert mtbf > 0.0
    # True MTBF under right-censoring should be strictly positive and finite
    assert np.isfinite(mtbf)


def test_log_rank_test_identical_distributions():
    np.random.seed(99)
    d1 = np.random.exponential(scale=3000.0, size=500)
    e1 = np.random.binomial(n=1, p=0.5, size=500).astype(bool)
    d2 = np.random.exponential(scale=3000.0, size=500)
    e2 = np.random.binomial(n=1, p=0.5, size=500).astype(bool)

    res = log_rank_test(d1, e1, d2, e2, name1="A", name2="B")
    assert res.p_value > 0.05
    assert not res.is_significant


def test_log_rank_test_significantly_different_pools():
    np.random.seed(99)
    # Pool 1 (high reliability): scale=10,000s
    d1 = np.random.exponential(scale=10000.0, size=500)
    e1 = np.random.binomial(n=1, p=0.3, size=500).astype(bool)

    # Pool 2 (low reliability / high eviction): scale=2,000s
    d2 = np.random.exponential(scale=2000.0, size=500)
    e2 = np.random.binomial(n=1, p=0.8, size=500).astype(bool)

    res = log_rank_test(d1, e1, d2, e2, name1="Reliable", name2="Fragile")
    assert res.is_significant
    assert res.p_value < 0.001
    assert res.chi2_statistic > 20.0


def test_core_engine_survival_fleet_integration(tmp_path):
    lake_dir = str(tmp_path / "lake_test_km")
    generate_spot_lifecycle_dataset(num_records=1500, base_output_dir=lake_dir, random_seed=42)

    adapter = DuckDBAnalyticalAdapter()
    engine = SpotAnalyticsEngine(storage=adapter, lake_path=lake_dir)

    # 1. Fit pool survival for c6i
    km_c6i = engine.fit_pool_survival(instance_family="c6i")
    assert km_c6i.is_fitted_
    mtbf_c6i = km_c6i.calculate_mtbf()
    assert mtbf_c6i > 1000.0

    # 2. Compare pools c6i vs g5
    lr_res = engine.compare_instance_pools("c6i", "g5")
    assert isinstance(lr_res, LogRankResult)
    assert lr_res.group1_name == "c6i"
    assert lr_res.group2_name == "g5"

    # 3. Compute actuarial MTBF comparison
    mtbf_summary = engine.compute_actuarial_mtbf_comparison()
    assert len(mtbf_summary) >= 3
    assert "actuarial_mtbf_hours" in mtbf_summary.columns
    assert "censoring_bias_pct" in mtbf_summary.columns
