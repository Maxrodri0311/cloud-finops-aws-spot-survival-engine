"""
tests/test_duckdb_adapter.py - Tests for the DuckDB Analytical Adapter.
Verifies Athena/Glue emulation, Hive partition projection, and predicate pushdown.
"""

import os
import shutil
import pytest
import pandas as pd
from src.data_generator import generate_spot_lifecycle_dataset
from src.adapters.duckdb_adapter import DuckDBAnalyticalAdapter


@pytest.fixture(scope="module")
def partitioned_lake(tmp_path_factory):
    lake_dir = str(tmp_path_factory.mktemp("lake_duckdb"))
    generate_spot_lifecycle_dataset(
        num_records=1500,
        base_output_dir=lake_dir,
        random_seed=123,
    )
    yield lake_dir
    shutil.rmtree(lake_dir, ignore_errors=True)


def test_duckdb_basic_query():
    adapter = DuckDBAnalyticalAdapter()
    df = adapter.execute_query("SELECT 42 as answer, 'Coderio' as tenant;")
    assert len(df) == 1
    assert df.iloc[0]["answer"] == 42
    assert df.iloc[0]["tenant"] == "Coderio"


def test_duckdb_scan_parquet_with_predicate_pushdown(partitioned_lake):
    adapter = DuckDBAnalyticalAdapter()
    parquet_glob = os.path.join(partitioned_lake, "**", "*.parquet")
    
    # Filter by region = us-east-1
    res = adapter.scan_parquet(
        base_path=parquet_glob,
        filters={"region": "us-east-1"},
        columns=["instance_id", "region", "duration_seconds", "event_observed"]
    )
    assert len(res) > 0
    assert (res["region"] == "us-east-1").all()
    assert set(res.columns) == {"instance_id", "region", "duration_seconds", "event_observed"}


def test_duckdb_registered_view_aggregation(partitioned_lake):
    adapter = DuckDBAnalyticalAdapter()
    parquet_glob = os.path.join(partitioned_lake, "**", "*.parquet")
    
    adapter.register_view("spot_lifecycle_view", parquet_glob)
    assert adapter.table_exists("spot_lifecycle_view")
    
    agg = adapter.execute_query("""
        SELECT 
            region,
            instance_family,
            COUNT(*) as total_nodes,
            ROUND(AVG(duration_seconds), 2) as mean_lifetime_sec,
            ROUND(AVG(CASE WHEN event_observed THEN 1.0 ELSE 0.0 END), 4) as interruption_rate
        FROM spot_lifecycle_view
        GROUP BY region, instance_family
        ORDER BY total_nodes DESC;
    """)
    assert len(agg) > 0
    assert "mean_lifetime_sec" in agg.columns
    assert "interruption_rate" in agg.columns
    assert (agg["total_nodes"] > 0).all()
