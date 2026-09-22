"""
src/domain/contracts.py - Inversion of Dependencies (DIP) Protocols.
Defines abstract contracts for analytical storage, hazard inference,
checkpoint persistence, and FinOps cost models.
All domain services depend exclusively on these interfaces.
"""

from typing import Protocol, Optional, List, Dict, Any
import pandas as pd
from src.domain.entities import (
    RuntimeContext,
    CheckpointReceipt,
    CheckpointPolicyDecision,
    FinOpsMetrics,
)


class AnalyticalStorageProtocol(Protocol):
    """
    Abstract analytical storage interface (DuckDB, AWS Athena, or In-Memory Mock).
    Enables zero-touch switching between local Hive Parquet files and cloud catalogs.
    """
    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes a vectorized SQL analytical query and returns a Pandas DataFrame."""
        ...

    def scan_parquet(
        self,
        base_path: str,
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Reads Parquet data using partition pruning and predicate pushdown."""
        ...


class HazardEstimatorProtocol(Protocol):
    """
    Abstract hazard and survival estimator interface (Kaplan-Meier, Cox PH, or Constant).
    """
    def predict_hazard(self, context: RuntimeContext) -> float:
        """Returns the instantaneous hazard rate h(t) for the given execution context."""
        ...

    def predict_survival_probability(self, context: RuntimeContext, horizon_seconds: float) -> float:
        """Returns S(t + horizon | T >= t), probability of surviving the next horizon_seconds."""
        ...


class CheckpointStoreProtocol(Protocol):
    """
    Abstract state persistence interface (Local S3 emulator, S3 Multipart, DynamoDB manifest).
    """
    def commit_checkpoint(self, receipt: CheckpointReceipt) -> bool:
        """Persists the checkpoint metadata atomically."""
        ...

    def get_latest_checkpoint(self, job_id: str) -> Optional[CheckpointReceipt]:
        """Retrieves the latest verified checkpoint for a job."""
        ...


class CostModelProtocol(Protocol):
    """
    Abstract economic model evaluating the replay cost and I/O overhead.
    """
    def expected_replay_cost(self, context: RuntimeContext) -> float:
        """Calculates C_r(L) = L * c_compute + C_restart + C_reload."""
        ...

    def io_checkpoint_cost(self, context: RuntimeContext) -> float:
        """Calculates C_io (network transfer + S3 PUT + serialization overhead)."""
        ...
