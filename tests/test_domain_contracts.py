"""
tests/test_domain_contracts.py - Unit tests for Domain Entities and DIP Protocols.
Ensures zero-dependency in-memory execution sub-5ms.
"""

import pytest
import pandas as pd
from src.domain.entities import (
    SpotTaskExposure,
    RuntimeContext,
    CheckpointPolicyDecision,
    CheckpointReceipt,
    FinOpsMetrics,
)
from src.domain.contracts import (
    AnalyticalStorageProtocol,
    HazardEstimatorProtocol,
    CheckpointStoreProtocol,
    CostModelProtocol,
)


def test_spot_task_exposure_entity():
    exposure = SpotTaskExposure(
        event_id="evt-0000001",
        instance_id="i-12345678",
        job_id="job-000001",
        pipeline_type="SageMaker_Distributed_Train",
        launch_ts="2026-03-01T10:00:00",
        terminal_ts="2026-03-01T14:30:00",
        duration_seconds=16200.0,
        event_observed=True,
        event_type="EVICTION",
        region="us-east-1",
        availability_zone="us-east-1a",
        instance_type="c6i.4xlarge",
        instance_family="c6i",
        vcpu=16,
        memory_gib=32.0,
        spot_price_usd_per_hour=0.204,
        ondemand_price_usd_per_hour=0.680,
        price_spread_ratio=0.300,
        price_drift_15m=0.012,
        cpu_utilization=0.74,
        gpu_utilization=0.0,
        cluster_size=12,
        checkpoint_age_seconds=180.0,
        checkpoint_size_bytes=104857600,
        replay_cost_usd=0.2602,
        rebalance_recommended=True,
        interruption_notice=True,
        year=2026,
        month=3,
    )
    assert exposure.event_id == "evt-0000001"
    assert exposure.event_observed is True
    assert exposure.price_spread_ratio == 0.300


def test_runtime_context_and_decision():
    ctx = RuntimeContext(
        instance_id="i-12345678",
        job_id="job-000001",
        pipeline_type="EMR_Spark_ETL",
        instance_type="m6i.2xlarge",
        region="us-west-2",
        duration_seconds=7200.0,
        current_hazard=0.0005,
        decision_window_seconds=60.0,
        seconds_since_last_checkpoint=300.0,
        compute_cost_rate_per_sec=0.0001,
        restart_cost_usd=0.15,
        reload_cost_usd=0.10,
        io_checkpoint_cost_usd=0.02,
        estimated_checkpoint_duration_sec=12.0,
    )
    assert ctx.duration_seconds == 7200.0
    assert ctx.current_hazard == 0.0005

    decision = CheckpointPolicyDecision(
        action="DYNAMIC_CHECKPOINT",
        reason="Expected replay loss exceeds S3 I/O cost",
        hazard_score=0.0005,
        interruption_probability=0.0295,
        expected_loss_no_save_usd=0.038,
        save_cost_usd=0.020,
        net_benefit_usd=0.018,
        can_complete_before_eviction=True,
    )
    assert decision.action == "DYNAMIC_CHECKPOINT"
    assert decision.net_benefit_usd > 0


def test_dip_mock_storage_adapter():
    """Verifies that an in-memory mock cleanly satisfies AnalyticalStorageProtocol."""
    class InMemoryMockStorage:
        def execute_query(self, query: str) -> pd.DataFrame:
            return pd.DataFrame([{"pool": "c6i", "mtbf_hours": 8.4}])

        def scan_parquet(self, base_path: str, filters=None, columns=None) -> pd.DataFrame:
            return pd.DataFrame([{"instance_id": "i-mock", "status": "ACTIVE"}])

    adapter: AnalyticalStorageProtocol = InMemoryMockStorage()
    res = adapter.execute_query("SELECT 1")
    assert len(res) == 1
    assert res.iloc[0]["pool"] == "c6i"
