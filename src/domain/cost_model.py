"""
src/domain/cost_model.py - AWS FinOps Cost and Replay Loss Model.
Implements CostModelProtocol.
Calculates expected recomputation cost Cr(L) and S3 snapshot I/O overhead Cio.
"""

from src.domain.contracts import CostModelProtocol
from src.domain.entities import RuntimeContext


class AWSFinOpsCostModel(CostModelProtocol):
    """
    Evaluates the economic trade-off between S3 storage I/O and sunk compute loss.
    """

    def __init__(
        self,
        s3_put_cost_usd: float = 0.005,        # Cost per 1,000 PUT requests ($0.005/1k)
        s3_storage_per_gib_month: float = 0.023, # S3 Standard $0.023 / GiB-month
        s3_data_transfer_per_gib: float = 0.000, # Same-region transfer is free
    ):
        self.s3_put_cost_usd = s3_put_cost_usd
        self.s3_storage_per_gib_month = s3_storage_per_gib_month
        self.s3_data_transfer_per_gib = s3_data_transfer_per_gib

    def expected_replay_cost(self, context: RuntimeContext, marginal_only: bool = True) -> float:
        """
        Calculates the compute progress loss avoided by checkpointing:
        C_r(L) = L * c_compute (marginal uncommitted progress).
        If marginal_only is False, adds fixed restart + reload overhead.
        """
        compute_burned = context.seconds_since_last_checkpoint * context.compute_cost_rate_per_sec
        if marginal_only:
            return float(round(compute_burned, 6))
        total_replay = compute_burned + context.restart_cost_usd + context.reload_cost_usd
        return float(round(total_replay, 6))

    def io_checkpoint_cost(self, context: RuntimeContext) -> float:
        """
        Calculates C_io (I/O, network and temporary S3 holding cost).
        Defaults to context.io_checkpoint_cost_usd or calibrated model.
        """
        if context.io_checkpoint_cost_usd > 0:
            return context.io_checkpoint_cost_usd

        # Calibrated default for ~2 GiB checkpoint snapshot
        estimated_size_gib = 2.0
        holding_hours = 24.0  # Checkpoint retention window
        holding_cost = (estimated_size_gib * self.s3_storage_per_gib_month / 720.0) * holding_hours
        total_io = self.s3_put_cost_usd + holding_cost
        return float(round(total_io, 6))
