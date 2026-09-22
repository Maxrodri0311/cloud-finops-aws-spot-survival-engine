"""
src/domain/entities.py - Domain Models and Value Objects.
Pure Python / Pydantic entities representing the AWS Spot Lifecycle domain.
No external I/O or vendor SDKs imported here.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SpotTaskExposure(BaseModel):
    """
    Statistical Unit: A single Spot instance task execution.
    Handles right-censored observations (event_observed=False when job completes before eviction).
    """
    event_id: str
    instance_id: str
    job_id: str
    pipeline_type: str  # SageMaker_Distributed_Train, EMR_Spark_ETL, Glue_Streaming_Job, StepFunctions_Batch
    launch_ts: str
    terminal_ts: str
    duration_seconds: float
    event_observed: bool  # True: Spot eviction / interruption; False: Right-censored completion
    event_type: str       # EVICTION or COMPLETED
    region: str
    availability_zone: str
    instance_type: str
    instance_family: str
    vcpu: int
    memory_gib: float
    spot_price_usd_per_hour: float
    ondemand_price_usd_per_hour: float
    price_spread_ratio: float
    price_drift_15m: float
    cpu_utilization: float
    gpu_utilization: float
    cluster_size: int
    checkpoint_age_seconds: float
    checkpoint_size_bytes: int
    replay_cost_usd: float
    rebalance_recommended: bool
    interruption_notice: bool
    year: int
    month: int


class RuntimeContext(BaseModel):
    """
    Real-time operational context of an active Spot node evaluated by the decision engine.
    """
    instance_id: str
    job_id: str
    pipeline_type: str
    instance_type: str
    region: str
    duration_seconds: float
    current_hazard: float = Field(default=0.0001, description="Instantaneous hazard h(t)")
    decision_window_seconds: float = Field(default=60.0, description="Delta evaluation window in seconds")
    seconds_since_last_checkpoint: float = Field(default=0.0, description="L: age of last committed state")
    compute_cost_rate_per_sec: float = Field(default=0.0005, description="Hourly spot cost / 3600")
    restart_cost_usd: float = Field(default=0.15, description="Fixed cold restart overhead")
    reload_cost_usd: float = Field(default=0.10, description="State and shuffle reload overhead")
    io_checkpoint_cost_usd: float = Field(default=0.02, description="C_io: cost to flush snapshot to S3")
    estimated_checkpoint_duration_sec: float = Field(default=15.0, description="T_checkpoint: duration to serialize and upload")
    rebalance_recommended: bool = Field(default=False, description="EC2 Rebalance Recommendation signal")
    interruption_notice: bool = Field(default=False, description="2-minute EC2 Spot Interruption notice")
    remaining_seconds_notice: float = Field(default=120.0, description="Time left before hard termination")


class CheckpointPolicyDecision(BaseModel):
    """
    Output of the dynamic causal decision policy.
    Answers: Should we checkpoint now, flush immediately, or migrate?
    """
    action: str  # NO_OP, DYNAMIC_CHECKPOINT, JIT_EMERGENCY_FLUSH, MIGRATE_ON_DEMAND
    reason: str
    hazard_score: float
    interruption_probability: float
    expected_loss_no_save_usd: float
    save_cost_usd: float
    net_benefit_usd: float
    can_complete_before_eviction: bool


class CheckpointReceipt(BaseModel):
    """
    Immutable receipt of a committed checkpoint snapshot.
    """
    checkpoint_id: str
    job_id: str
    instance_id: str
    storage_uri: str
    size_bytes: int
    write_duration_sec: float
    checksum_sha256: str
    timestamp: str
    success: bool = True


class FinOpsMetrics(BaseModel):
    """
    C-Level FinOps and reliability aggregated metrics.
    """
    total_exposures: int
    total_exposure_hours: float
    observed_evictions: int
    censored_completions: int
    interruption_rate: float
    gross_savings_usd: float
    total_checkpoint_cost_usd: float
    total_replay_loss_usd: float
    total_downtime_loss_usd: float
    net_savings_usd: float
    checkpoint_roi: float
    mtbf_hours: float
