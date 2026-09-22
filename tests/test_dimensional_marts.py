"""
tests/test_dimensional_marts.py - Unit & Integration Tests for Fase 4 Kimball Star-Schema Marts.
Verifies dimensional view creation, referential integrity between dimensions and facts,
C-level FinOps executive aggregates, signal efficiency metrics, and Parquet export.
"""

import os
import pytest
import pandas as pd
from src.data_generator import generate_spot_lifecycle_dataset
from src.adapters.duckdb_adapter import DuckDBAnalyticalAdapter
from src.marts.dimensional_marts import KimballLakehouseMarts
from src.core_engine import SpotAnalyticsEngine


@pytest.fixture(scope="module")
def lakehouse_fixture(tmp_path_factory):
    lake_dir = str(tmp_path_factory.mktemp("marts_lake_test"))
    df = generate_spot_lifecycle_dataset(
        num_records=1500,
        base_output_dir=lake_dir,
        random_seed=123,
    )
    adapter = DuckDBAnalyticalAdapter()
    marts = KimballLakehouseMarts(storage=adapter, lake_path=lake_dir)
    marts.build_dimensions_and_facts()
    return marts, adapter, lake_dir


def test_kimball_star_schema_views_creation(lakehouse_fixture):
    marts, adapter, _ = lakehouse_fixture
    assert marts.is_built

    # Verify existence of star schema tables
    assert adapter.table_exists("dim_instance_pool")
    assert adapter.table_exists("dim_pipeline")
    assert adapter.table_exists("fact_spot_exposure")
    assert adapter.table_exists("fact_interruption_event")
    assert adapter.table_exists("fact_checkpoint_event")


def test_star_schema_referential_integrity(lakehouse_fixture):
    _, adapter, _ = lakehouse_fixture

    # Test FK integrity between fact_spot_exposure and dim_instance_pool
    orphan_pools_query = """
        SELECT COUNT(*) as orphans
        FROM fact_spot_exposure f
        LEFT JOIN dim_instance_pool p ON f.pool_id = p.pool_id
        WHERE p.pool_id IS NULL;
    """
    orphans = adapter.execute_query(orphan_pools_query)["orphans"].iloc[0]
    assert orphans == 0, "No orphaned fact records allowed in Star Schema."

    # Test FK integrity between fact_spot_exposure and dim_pipeline
    orphan_pipelines_query = """
        SELECT COUNT(*) as orphans
        FROM fact_spot_exposure f
        LEFT JOIN dim_pipeline p ON f.pipeline_id = p.pipeline_id
        WHERE p.pipeline_id IS NULL;
    """
    orphans_pipe = adapter.execute_query(orphan_pipelines_query)["orphans"].iloc[0]
    assert orphans_pipe == 0, "No orphaned pipeline records allowed."


def test_finops_executive_mart(lakehouse_fixture):
    marts, _, _ = lakehouse_fixture
    df = marts.query_finops_executive_mart()

    assert len(df) == 1
    row = df.iloc[0]
    assert row["total_tasks_executed"] == 1500
    assert row["total_ondemand_benchmark_usd"] > row["total_spot_compute_usd"]
    assert row["total_gross_savings_usd"] > 0.0
    assert row["net_realized_savings_usd"] > 0.0
    assert 50.0 < row["avg_spot_discount_pct"] < 80.0
    assert row["net_savings_pct"] > 40.0


def test_pool_hazard_mart(lakehouse_fixture):
    marts, _, _ = lakehouse_fixture
    df = marts.query_pool_hazard_mart()

    assert len(df) >= 4  # c6i, m6i, r6i, g5
    assert "instance_family" in df.columns
    assert "region" in df.columns
    assert "eviction_rate_pct" in df.columns
    assert "cumulative_gross_savings_usd" in df.columns

    # GPU instances (g5) should exhibit higher eviction rates than compute (c6i)
    g5_eviction = df[df["instance_family"] == "g5"]["eviction_rate_pct"].mean()
    c6i_eviction = df[df["instance_family"] == "c6i"]["eviction_rate_pct"].mean()
    assert g5_eviction > c6i_eviction, "GPU pools must reflect higher empirical eviction risk"


def test_signal_efficiency_mart(lakehouse_fixture):
    marts, _, _ = lakehouse_fixture
    df = marts.query_signal_efficiency_mart()

    assert len(df) >= 4
    assert "total_evictions" in df.columns
    assert "rebalance_coverage_pct" in df.columns
    assert "evictions_with_120s_notice" in df.columns

    # Rebalance recommendation coverage should be around 70-85% for evictions
    avg_rebalance_cov = df["rebalance_coverage_pct"].mean()
    assert 60.0 <= avg_rebalance_cov <= 90.0


def test_export_marts_to_parquet(lakehouse_fixture, tmp_path):
    marts, _, _ = lakehouse_fixture
    export_dir = str(tmp_path / "exported_marts")

    exported = marts.export_marts_to_parquet(output_dir=export_dir)
    assert len(exported) == 5

    for tbl_name, file_path in exported.items():
        assert os.path.exists(file_path)
        assert os.path.getsize(file_path) > 0


def test_core_engine_marts_integration(tmp_path):
    lake_dir = str(tmp_path / "engine_marts_lake")
    generate_spot_lifecycle_dataset(num_records=1000, base_output_dir=lake_dir, random_seed=42)

    adapter = DuckDBAnalyticalAdapter()
    engine = SpotAnalyticsEngine(storage=adapter, lake_path=lake_dir)

    engine.build_marts()
    finops = engine.query_finops_executive_mart()
    assert len(finops) == 1

    hazard = engine.query_pool_hazard_mart()
    assert len(hazard) > 0

    signals = engine.query_signal_efficiency_mart()
    assert len(signals) > 0

    export_dir = str(tmp_path / "engine_exported_marts")
    exported = engine.export_dimensional_marts(output_dir=export_dir)
    assert len(exported) == 5
