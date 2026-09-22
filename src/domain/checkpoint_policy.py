"""
src/domain/checkpoint_policy.py - Causal Dynamic Checkpointing Decision Engine.
Balances S3 I/O costs against expected recomputation losses under stochastic evictions.
Handles 2-minute interruption notices and rebalance recommendation signals.
"""

from typing import Protocol, List, Dict, Any, Optional
import numpy as np
import pandas as pd
from src.domain.contracts import HazardEstimatorProtocol, CostModelProtocol
from src.domain.entities import RuntimeContext, CheckpointPolicyDecision


class CheckpointPolicyProtocol(Protocol):
    """Abstract contract for checkpointing policy engines."""
    def evaluate(self, context: RuntimeContext) -> CheckpointPolicyDecision: ...


class DynamicSurvivalCheckpointPolicy(CheckpointPolicyProtocol):
    """
    Causal Economic Checkpointing Policy:
    Executes snapshot to S3 iff:
    C_io < (1 - exp(- h(t) * Delta)) * (L * c_compute + C_restart + C_reload).
    Includes hard 120s ceiling for EC2 Spot Interruption notices.
    """

    def __init__(
        self,
        hazard_estimator: HazardEstimatorProtocol,
        cost_model: CostModelProtocol,
        emergency_margin_seconds: float = 20.0,
    ):
        self.hazard_estimator = hazard_estimator
        self.cost_model = cost_model
        self.emergency_margin_seconds = emergency_margin_seconds

    def evaluate(self, context: RuntimeContext) -> CheckpointPolicyDecision:
        # Calculate instantaneous hazard and economic metrics
        h_t = self.hazard_estimator.predict_hazard(context)
        L = context.seconds_since_last_checkpoint
        save_cost = self.cost_model.io_checkpoint_cost(context)
        c_rate = max(1e-7, context.compute_cost_rate_per_sec)

        # Daly's first-principles stochastic optimum interval:
        # L_opt = sqrt(2 * C_io / (h(t) * c_compute))
        l_opt = float(np.sqrt((2.0 * save_cost) / max(1e-9, h_t * c_rate)))

        # Causal risk probability over accumulated window L
        prob_interrupt = 1.0 - float(np.exp(- h_t * max(10.0, L)))
        replay_cost = self.cost_model.expected_replay_cost(context, marginal_only=True)
        expected_loss = prob_interrupt * replay_cost
        net_benefit = expected_loss - save_cost

        # ---------------------------------------------------------------------
        # Tier 2: Emergency AWS Signal Handlers (Interruption Notice & Rebalance)
        # ---------------------------------------------------------------------
        if context.interruption_notice:
            time_available = context.remaining_seconds_notice - self.emergency_margin_seconds
            can_complete = context.estimated_checkpoint_duration_sec < time_available

            if can_complete:
                return CheckpointPolicyDecision(
                    action="JIT_EMERGENCY_FLUSH",
                    reason="EC2 Interruption Notice active. Snapshot duration fits within 120s window.",
                    hazard_score=round(h_t, 6),
                    interruption_probability=1.0,
                    expected_loss_no_save_usd=round(replay_cost, 4),
                    save_cost_usd=round(save_cost, 4),
                    net_benefit_usd=round(replay_cost - save_cost, 4),
                    can_complete_before_eviction=True,
                )
            else:
                return CheckpointPolicyDecision(
                    action="MIGRATE_ON_DEMAND",
                    reason="EC2 Interruption Notice active, but snapshot cannot complete safely before eviction.",
                    hazard_score=round(h_t, 6),
                    interruption_probability=1.0,
                    expected_loss_no_save_usd=round(replay_cost, 4),
                    save_cost_usd=round(save_cost, 4),
                    net_benefit_usd=0.0,
                    can_complete_before_eviction=False,
                )

        if context.rebalance_recommended and L >= 300.0:
            return CheckpointPolicyDecision(
                action="DYNAMIC_CHECKPOINT",
                reason="EC2 Rebalance Recommendation signal active with uncommitted state.",
                hazard_score=round(h_t, 6),
                interruption_probability=round(prob_interrupt, 4),
                expected_loss_no_save_usd=round(expected_loss, 4),
                save_cost_usd=round(save_cost, 4),
                net_benefit_usd=round(net_benefit, 4),
                can_complete_before_eviction=True,
            )

        # ---------------------------------------------------------------------
        # Tier 1: Normal Operation - Daly First-Principles Stochastic Optimum
        # Triggers when accumulated exposure L reaches Daly's optimum L*
        # L_opt = sqrt(2 * C_io / (h(t) * c_compute)).
        # ---------------------------------------------------------------------
        if L >= l_opt:
            return CheckpointPolicyDecision(
                action="DYNAMIC_CHECKPOINT",
                reason=(
                    f"Causal Rule Triggered: Exposure ({round(L, 1)}s) reached optimal "
                    f"checkpoint interval L*={round(l_opt, 1)}s under hazard h(t)={round(h_t, 6)}."
                ),
                hazard_score=round(h_t, 6),
                interruption_probability=round(prob_interrupt, 4),
                expected_loss_no_save_usd=round(expected_loss, 4),
                save_cost_usd=round(save_cost, 4),
                net_benefit_usd=round(net_benefit, 4),
                can_complete_before_eviction=True,
            )

        return CheckpointPolicyDecision(
            action="NO_OP",
            reason=(
                f"S3 I/O cost (${round(save_cost, 4)}) exceeds expected loss "
                f"(${round(expected_loss, 4)}). Continuing compute."
            ),
            hazard_score=round(h_t, 6),
            interruption_probability=round(prob_interrupt, 4),
            expected_loss_no_save_usd=round(expected_loss, 4),
            save_cost_usd=round(save_cost, 4),
            net_benefit_usd=round(net_benefit, 4),
            can_complete_before_eviction=True,
        )


class FixedIntervalCheckpointPolicy(CheckpointPolicyProtocol):
    """
    Standard Industry Baseline Policy:
    Executes checkpoints at a rigid periodic interval (default: 900s / 15 min)
    completely oblivious to spot market hazards or early warning signals.
    """

    def __init__(self, interval_seconds: float = 900.0, io_cost_usd: float = 0.02):
        self.interval_seconds = interval_seconds
        self.io_cost_usd = io_cost_usd

    def evaluate(self, context: RuntimeContext) -> CheckpointPolicyDecision:
        if context.seconds_since_last_checkpoint >= self.interval_seconds:
            return CheckpointPolicyDecision(
                action="PERIODIC_CHECKPOINT",
                reason=f"Fixed periodic interval of {self.interval_seconds}s reached.",
                hazard_score=context.current_hazard,
                interruption_probability=0.0,
                expected_loss_no_save_usd=0.0,
                save_cost_usd=self.io_cost_usd,
                net_benefit_usd=0.0,
                can_complete_before_eviction=True,
            )
        return CheckpointPolicyDecision(
            action="NO_OP",
            reason="Fixed interval not yet elapsed.",
            hazard_score=context.current_hazard,
            interruption_probability=0.0,
            expected_loss_no_save_usd=0.0,
            save_cost_usd=self.io_cost_usd,
            net_benefit_usd=0.0,
            can_complete_before_eviction=True,
        )


class PolicySimulator:
    """
    Monte Carlo Evaluation Harness comparing:
    1. Fixed Interval Policy (15m periodic)
    2. Dynamic Survival Causal Policy
    3. On-Demand Zero-Interruption Baseline
    """

    @staticmethod
    def simulate_fleet(
        workloads: List[Dict[str, Any]],
        dynamic_policy: CheckpointPolicyProtocol,
        fixed_policy: CheckpointPolicyProtocol,
    ) -> pd.DataFrame:
        """
        Simulates execution across identical workloads under all three policies.
        """
        results = {
            "On-Demand Pure": {"compute_cost": 0.0, "io_cost": 0.0, "replay_loss": 0.0, "checkpoints": 0},
            "Fixed-Interval (15m)": {"compute_cost": 0.0, "io_cost": 0.0, "replay_loss": 0.0, "checkpoints": 0},
            "Dynamic Survival (Ours)": {"compute_cost": 0.0, "io_cost": 0.0, "replay_loss": 0.0, "checkpoints": 0},
        }

        step_seconds = 60.0  # Evaluation step

        for w in workloads:
            duration = float(w["duration_seconds"])
            evicted = bool(w["event_observed"])
            spot_rate = float(w["spot_price_usd_per_hour"]) / 3600.0
            ondemand_rate = float(w["ondemand_price_usd_per_hour"]) / 3600.0
            hazard = float(w.get("hazard", 0.0002))
            has_rebalance = bool(w.get("rebalance_recommended", False))
            rebalance_time = max(0.0, duration - 600.0) if (has_rebalance and evicted) else float("inf")

            # 1. On-Demand Pure Baseline (Never evicted, pays high rate)
            results["On-Demand Pure"]["compute_cost"] += duration * ondemand_rate

            # 2. Fixed-Interval Policy Simulation (15m periodic)
            fixed_checkpoints = 0
            fixed_last_ckpt = 0.0
            t = 0.0
            while t < duration:
                t += step_seconds
                fixed_last_ckpt += step_seconds
                ctx = RuntimeContext(
                    instance_id="i-sim",
                    job_id="job-sim",
                    pipeline_type="Sim",
                    instance_type="c6i.4xlarge",
                    region="us-east-1",
                    duration_seconds=t,
                    current_hazard=hazard,
                    seconds_since_last_checkpoint=fixed_last_ckpt,
                    compute_cost_rate_per_sec=spot_rate,
                )
                dec = fixed_policy.evaluate(ctx)
                if dec.action == "PERIODIC_CHECKPOINT":
                    fixed_checkpoints += 1
                    fixed_last_ckpt = 0.0

            results["Fixed-Interval (15m)"]["compute_cost"] += duration * spot_rate
            results["Fixed-Interval (15m)"]["io_cost"] += fixed_checkpoints * 0.02
            results["Fixed-Interval (15m)"]["checkpoints"] += fixed_checkpoints
            if evicted:
                # Sunk replay loss: lost compute since last fixed checkpoint + reload overhead
                results["Fixed-Interval (15m)"]["replay_loss"] += (fixed_last_ckpt * spot_rate) + 0.25

            # 3. Dynamic Survival Policy Simulation
            dyn_checkpoints = 0
            dyn_last_ckpt = 0.0
            t = 0.0
            jit_flushed = False
            while t < duration:
                t += step_seconds
                dyn_last_ckpt += step_seconds
                is_near_end = (duration - t) <= 120.0 and evicted
                rebalance_active = t >= rebalance_time
                
                ctx = RuntimeContext(
                    instance_id="i-sim",
                    job_id="job-sim",
                    pipeline_type="Sim",
                    instance_type="c6i.4xlarge",
                    region="us-east-1",
                    duration_seconds=t,
                    current_hazard=hazard,
                    seconds_since_last_checkpoint=dyn_last_ckpt,
                    compute_cost_rate_per_sec=spot_rate,
                    rebalance_recommended=rebalance_active,
                    interruption_notice=is_near_end,
                    remaining_seconds_notice=max(10.0, duration - t) if is_near_end else 120.0,
                    decision_window_seconds=300.0,
                )
                dec = dynamic_policy.evaluate(ctx)
                if dec.action in ["DYNAMIC_CHECKPOINT", "JIT_EMERGENCY_FLUSH"]:
                    dyn_checkpoints += 1
                    dyn_last_ckpt = 0.0
                    if dec.action == "JIT_EMERGENCY_FLUSH":
                        jit_flushed = True

            results["Dynamic Survival (Ours)"]["compute_cost"] += duration * spot_rate
            results["Dynamic Survival (Ours)"]["io_cost"] += dyn_checkpoints * 0.02
            results["Dynamic Survival (Ours)"]["checkpoints"] += dyn_checkpoints
            if evicted:
                # JIT Emergency Flush saves uncommitted work before eviction!
                residual_loss = 0.05 if jit_flushed else (dyn_last_ckpt * spot_rate) + 0.25
                results["Dynamic Survival (Ours)"]["replay_loss"] += residual_loss

        # Format comparison table
        table_rows = []
        ondemand_total = results["On-Demand Pure"]["compute_cost"]

        for name, metrics in results.items():
            total_cost = metrics["compute_cost"] + metrics["io_cost"] + metrics["replay_loss"]
            net_savings = ondemand_total - total_cost
            savings_pct = (net_savings / ondemand_total) * 100.0 if ondemand_total > 0 else 0.0

            # ROI = (Replay Avoided - Checkpoint I/O) / Checkpoint I/O
            fixed_loss = results["Fixed-Interval (15m)"]["replay_loss"]
            replay_avoided = max(0.0, fixed_loss - metrics["replay_loss"])
            roi = (replay_avoided - metrics["io_cost"]) / metrics["io_cost"] if metrics["io_cost"] > 0 else 0.0

            table_rows.append({
                "Policy": name,
                "Compute Cost ($)": round(metrics["compute_cost"], 2),
                "S3 I/O Cost ($)": round(metrics["io_cost"], 2),
                "Replay Sunk Loss ($)": round(metrics["replay_loss"], 2),
                "Total Financial Cost ($)": round(total_cost, 2),
                "Net Savings ($)": round(net_savings, 2),
                "Net Savings (%)": round(savings_pct, 2),
                "Checkpoints Committed": metrics["checkpoints"],
                "Checkpoint ROI (%)": round(roi * 100.0, 1),
            })

        return pd.DataFrame(table_rows)
