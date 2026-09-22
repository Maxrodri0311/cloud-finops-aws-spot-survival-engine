"""
src/domain/cox_model.py - Multivariate Cox Proportional Hazards Model.
Implements HazardEstimatorProtocol using regularized partial likelihood estimation.
Computes Hazard Ratios (HR), baseline hazard h0(t), and real-time hazard scores h(t | X).
"""

from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from src.domain.contracts import HazardEstimatorProtocol
from src.domain.entities import RuntimeContext


class CoxProportionalHazardsModel(HazardEstimatorProtocol):
    """
    Decoupled Cox PH Model for Spot Eviction Hazards.
    h(t | X) = h_0(t) * exp(beta^T * X)
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.fitter = CoxPHFitter(penalizer=penalizer)
        self.is_fitted: bool = False
        self.feature_columns: List[str] = []
        self.baseline_hazard_: Optional[pd.DataFrame] = None
        self.baseline_cumulative_hazard_: Optional[pd.DataFrame] = None
        self.concordance_index_: float = 0.0

    def fit(
        self,
        df: pd.DataFrame,
        duration_col: str = "duration_seconds",
        event_col: str = "event_observed",
        feature_cols: Optional[List[str]] = None,
    ) -> "CoxProportionalHazardsModel":
        """
        Fits the regularized Cox PH model on Spot task exposures.
        """
        if feature_cols is None:
            # Default market and workload features
            feature_cols = [
                "price_drift_15m",
                "price_spread_ratio",
                "cpu_utilization",
                "gpu_utilization",
                "cluster_size",
            ]

        self.feature_columns = feature_cols
        train_df = df[[duration_col, event_col] + feature_cols].copy()
        
        # Ensure correct datatypes
        train_df[duration_col] = train_df[duration_col].astype(float)
        train_df[event_col] = train_df[event_col].astype(int)
        for col in feature_cols:
            train_df[col] = train_df[col].astype(float)

        self.fitter.fit(
            train_df,
            duration_col=duration_col,
            event_col=event_col,
            show_progress=False,
        )

        self.baseline_hazard_ = self.fitter.baseline_hazard_
        self.baseline_cumulative_hazard_ = self.fitter.baseline_cumulative_hazard_
        self.concordance_index_ = float(self.fitter.concordance_index_)
        self.is_fitted = True
        return self

    def summary_table(self) -> pd.DataFrame:
        """Returns coefficients, hazard ratios (exp(coef)), and 95% confidence intervals."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before retrieving summary table.")
        
        summary = self.fitter.summary.copy()
        summary["hazard_ratio"] = np.exp(summary["coef"])
        summary["hr_lower_95"] = np.exp(summary["coef lower 95%"])
        summary["hr_upper_95"] = np.exp(summary["coef upper 95%"])
        
        return summary[[
            "coef",
            "hazard_ratio",
            "se(coef)",
            "hr_lower_95",
            "hr_upper_95",
            "p",
        ]]

    def predict_partial_hazard(self, feature_vector: Dict[str, float]) -> float:
        """Computes exp(beta^T * X), the relative risk multiplier."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting hazard.")
            
        x_vals = np.array([feature_vector.get(col, 0.0) for col in self.feature_columns])
        betas = self.fitter.params_.values
        eta = float(np.dot(betas, x_vals))
        return float(np.exp(eta))

    def predict_hazard(self, context: RuntimeContext) -> float:
        """
        Implements HazardEstimatorProtocol.predict_hazard.
        Returns the instantaneous hazard rate h(t) = h_0(t) * exp(beta^T * X).
        """
        if not self.is_fitted or self.baseline_cumulative_hazard_ is None:
            # Fallback to context hazard if model is not yet trained
            return context.current_hazard

        # Estimate baseline hazard rate h_0(t) via numerical derivative of H_0(t)
        t = context.duration_seconds
        timeline = self.baseline_cumulative_hazard_.index.values
        cum_vals = self.baseline_cumulative_hazard_.iloc[:, 0].values

        delta_eval = 300.0  # 5-minute evaluation horizon
        c1 = float(np.interp(t + delta_eval, timeline, cum_vals, left=0.0, right=cum_vals[-1]))
        c0 = float(np.interp(t, timeline, cum_vals, left=0.0, right=cum_vals[-1]))
        h0 = (c1 - c0) / delta_eval

        # If flat or beyond timeline, use long-run average baseline hazard
        if h0 < 1e-7 and len(timeline) > 0 and timeline[-1] > 0:
            h0 = float(cum_vals[-1] / timeline[-1])

        # Feature mapping from runtime context
        features = {
            "price_drift_15m": getattr(context, "price_drift_15m", 0.01),
            "price_spread_ratio": getattr(context, "price_spread_ratio", 0.30),
            "cpu_utilization": getattr(context, "cpu_utilization", 0.70),
            "gpu_utilization": getattr(context, "gpu_utilization", 0.80 if "g5" in context.instance_type else 0.0),
            "cluster_size": getattr(context, "cluster_size", 12.0),
        }
        partial_hazard = self.predict_partial_hazard(features)
        instantaneous_hazard = h0 * partial_hazard
        
        # Guard against zero or extreme hazards
        return float(np.clip(instantaneous_hazard, 1e-6, 0.01))

    def predict_survival_probability(self, context: RuntimeContext, horizon_seconds: float) -> float:
        """
        Implements HazardEstimatorProtocol.predict_survival_probability.
        Returns P(T > t + horizon | T >= t) = exp(- h(t) * horizon).
        """
        h_t = self.predict_hazard(context)
        prob = float(np.exp(- h_t * horizon_seconds))
        return float(np.clip(prob, 0.0, 1.0))
