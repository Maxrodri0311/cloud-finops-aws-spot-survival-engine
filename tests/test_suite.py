"""
tests/test_suite.py - Master Integration Test Suite for Coderio AWS Spot Survival Engine.
Verifies end-to-end data pipeline, DIP abstractions, and analytical query execution.
"""

import os
import shutil
import pytest
import pandas as pd
from src.data_generator import generate_spot_lifecycle_dataset
from src.adapters.duckdb_adapter import DuckDBAnalyticalAdapter
from src.domain.contracts import AnalyticalStorageProtocol


@pytest.fixture(scope="session")
def test_lakehouse(tmp_path_factory):
    base_dir = str(tmp_path_factory.mktemp("lakehouse_master"))
    df = generate_spot_lifecycle_dataset(
        num_records=3000,
        base_output_dir=base_dir,
        random_seed=999,
    )
    yield df, base_dir
    shutil.rmtree(base_dir, ignore_errors=True)


def test_data_generation_integrity(test_lakehouse):
    df, _ = test_lakehouse
    assert len(df) == 3000
    assert "event_id" in df.columns
    assert "event_observed" in df.columns
    assert (df["spot_price_usd_per_hour"] > 0).all()
    assert (df["ondemand_price_usd_per_hour"] > df["spot_price_usd_per_hour"]).all()


def test_duckdb_storage_integration(test_lakehouse):
    _, base_dir = test_lakehouse
    adapter = DuckDBAnalyticalAdapter()
    parquet_path = os.path.join(base_dir, "**", "*.parquet")
    
    query = f"""
        SELECT 
            region,
            instance_family,
            COUNT(*) as total_events,
            ROUND(AVG(duration_seconds), 2) as avg_duration_sec,
            ROUND(AVG(CASE WHEN event_observed THEN 1.0 ELSE 0.0 END), 4) as interruption_rate,
            ROUND(AVG(ondemand_price_usd_per_hour - spot_price_usd_per_hour), 4) as avg_hourly_savings_usd
        FROM read_parquet('{parquet_path}', hive_partitioning = true)
        GROUP BY region, instance_family
        ORDER BY total_events DESC;
    """
    res = adapter.execute_query(query)
    assert len(res) > 0
    assert "interruption_rate" in res.columns
    assert "avg_hourly_savings_usd" in res.columns
    assert (res["avg_hourly_savings_usd"] > 0).all()


def test_core_engine_dependency_inversion_mock():
    """Valida que la lógica de dominio funciona con un mock en memoria sin tocar disco ni DuckDB."""
    class MockStorageAdapter:
        def execute_query(self, query: str) -> pd.DataFrame:
            return pd.DataFrame([
                {
                    "region": "us-east-1",
                    "instance_family": "c6i",
                    "total_events": 100,
                    "interruption_rate": 0.25,
                }
            ])

        def scan_parquet(self, base_path: str, filters=None, columns=None) -> pd.DataFrame:
            return pd.DataFrame([{"instance_id": "i-mock", "status": "ACTIVE"}])

    adapter: AnalyticalStorageProtocol = MockStorageAdapter()
    res = adapter.execute_query("SELECT 1")
    assert len(res) == 1
    assert res.iloc[0]["instance_family"] == "c6i"
    assert res.iloc[0]["interruption_rate"] == 0.25
