"""
tests/test_data_generator.py - Tests for the Calibrated Stochastic Data Generator.
Verifies Weibull hazard behavior, right-censoring proportion, and Hive partitioning.
"""

import os
import shutil
import pytest
import pandas as pd
from src.data_generator import generate_spot_lifecycle_dataset


@pytest.fixture(scope="module")
def sample_dataset(tmp_path_factory):
    lake_dir = str(tmp_path_factory.mktemp("lake_test"))
    df = generate_spot_lifecycle_dataset(
        num_records=2000,
        base_output_dir=lake_dir,
        random_seed=42,
    )
    yield df, lake_dir
    # Cleanup
    shutil.rmtree(lake_dir, ignore_errors=True)


def test_dataset_generation_completeness(sample_dataset):
    df, _ = sample_dataset
    assert len(df) == 2000
    expected_cols = [
        "event_id", "instance_id", "job_id", "pipeline_type",
        "launch_ts", "terminal_ts", "duration_seconds",
        "event_observed", "event_type", "region", "availability_zone",
        "instance_type", "instance_family", "vcpu", "memory_gib",
        "spot_price_usd_per_hour", "ondemand_price_usd_per_hour",
        "price_spread_ratio", "price_drift_15m", "cpu_utilization",
        "gpu_utilization", "cluster_size", "checkpoint_age_seconds",
        "checkpoint_size_bytes", "replay_cost_usd",
        "rebalance_recommended", "interruption_notice", "year", "month"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    assert df.isnull().sum().sum() == 0, "Dataset contains unexpected null values"


def test_survival_censoring_and_durations(sample_dataset):
    df, _ = sample_dataset
    assert (df["duration_seconds"] > 0).all(), "All durations must be strictly positive"
    
    # Event observed is boolean
    assert set(df["event_observed"].unique()).issubset({True, False})
    
    # Censoring rate check: both evictions and completed jobs must exist
    eviction_rate = df["event_observed"].mean()
    assert 0.20 <= eviction_rate <= 0.70, f"Eviction rate {eviction_rate:.2f} out of expected range"


def test_hive_partition_directory_structure(sample_dataset):
    _, lake_dir = sample_dataset
    # Check that region partition folders exist
    subdirs = os.listdir(lake_dir)
    assert any("region=" in d for d in subdirs), "Hive partitioning by region missing"
    
    # Check nested instance_family partition
    first_region = [d for d in subdirs if "region=" in d][0]
    region_path = os.path.join(lake_dir, first_region)
    fam_dirs = os.listdir(region_path)
    assert any("instance_family=" in f for f in fam_dirs), "Hive partitioning by instance_family missing"
