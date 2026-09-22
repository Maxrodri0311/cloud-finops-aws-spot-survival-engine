"""
src/domain/survival_models.py - Pure Actuarial & Survival Modeling Engine.
Implements Kaplan-Meier product-limit estimator, Greenwood log-log confidence bands,
actuarial life tables, MTBF integration, and Log-Rank cross-pool hypothesis testing.
Pure Python / NumPy / SciPy with zero vendor locking.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class LogRankResult:
    """Statistical summary of Log-Rank hypothesis test between two exposure pools."""
    group1_name: str
    group2_name: str
    observed_events_1: int
    expected_events_1: float
    observed_events_2: int
    expected_events_2: float
    chi2_statistic: float
    p_value: float
    is_significant: bool  # Reject H0 (equality of survival curves) at alpha = 0.05


class KaplanMeierEstimator:
    """
    Non-parametric Kaplan-Meier Product-Limit Survival Estimator.
    Handles right-censored Spot instance task durations with Greenwood log-log variance.
    """

    def __init__(self):
        self.timeline_: np.ndarray = np.array([])
        self.survival_table_: Optional[pd.DataFrame] = None
        self.is_fitted_: bool = False

    def fit(self, durations: np.ndarray, events: np.ndarray) -> "KaplanMeierEstimator":
        """
        Fits the survival curve S(t) = P(T > t) using observed durations and event indicators.
        durations: strictly positive observed exposure times (seconds).
        events: boolean indicator (1 = Eviction / Failure, 0 = Right-Censored Completion).
        """
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=bool)

        if len(durations) == 0:
            raise ValueError("Durations and events cannot be empty.")
        if len(durations) != len(events):
            raise ValueError("Durations and events arrays must have identical length.")
        if np.any(durations <= 0):
            raise ValueError("All durations must be strictly positive (>0).")

        n_total = len(durations)
        
        # Sort distinct times
        unique_times = np.unique(durations)
        unique_times = np.sort(unique_times)

        # Prepend t = 0 for full curve integration
        times_with_zero = np.insert(unique_times, 0, 0.0)

        # Count events and censorings at each distinct time
        # d_j: observed failures at t_j
        # c_j: right-censored observations at t_j
        event_counts = np.zeros_like(unique_times, dtype=int)
        censor_counts = np.zeros_like(unique_times, dtype=int)

        for i, t in enumerate(unique_times):
            mask = (durations == t)
            event_counts[i] = np.sum(events[mask])
            censor_counts[i] = np.sum(~events[mask])

        # Compute risk sets n_j iteratively
        # n_0 = n_total
        # n_{j+1} = n_j - d_j - c_j
        risk_sets = np.zeros_like(unique_times, dtype=int)
        current_n = n_total
        for i in range(len(unique_times)):
            risk_sets[i] = current_n
            current_n -= (event_counts[i] + censor_counts[i])

        # Kaplan-Meier product-limit formula:
        # S(t) = prod_{t_j <= t} (1 - d_j / n_j)
        prob_step = 1.0 - (event_counts / np.maximum(risk_sets, 1))
        survival_curve = np.cumprod(prob_step)

        # Greenwood variance summation: sum (d_j / (n_j * (n_j - d_j)))
        # Safe denominator avoiding divide by zero when n_j == d_j
        denom = risk_sets * (risk_sets - event_counts)
        safe_denom = np.where(denom <= 0, np.nan, denom)
        greenwood_terms = np.where(denom <= 0, 0.0, event_counts / safe_denom)
        greenwood_sum = np.cumsum(greenwood_terms)
        greenwood_variance = (survival_curve ** 2) * greenwood_sum

        # Greenwood 95% Log-Log Confidence Bands:
        # g(S) = log(-log(S))
        # theta = exp(z * sqrt(greenwood_sum) / |log(S)|)
        # CI_lower = S^theta, CI_upper = S^(1/theta)
        z_crit = 1.95996  # 95% two-tailed
        ci_lower = np.zeros_like(survival_curve)
        ci_upper = np.zeros_like(survival_curve)

        for i, s in enumerate(survival_curve):
            if s <= 0.0:
                ci_lower[i] = 0.0
                ci_upper[i] = 0.0
            elif s >= 1.0:
                ci_lower[i] = 1.0
                ci_upper[i] = 1.0
            else:
                log_s = np.log(s)
                se_log_log = np.sqrt(greenwood_sum[i]) / np.abs(log_s)
                theta = np.exp(z_crit * se_log_log)
                ci_lower[i] = float(np.clip(s ** theta, 0.0, 1.0))
                ci_upper[i] = float(np.clip(s ** (1.0 / theta), 0.0, 1.0))

        # Build full table including t = 0
        df_table = pd.DataFrame({
            "timeline": np.insert(unique_times, 0, 0.0),
            "n_at_risk": np.insert(risk_sets, 0, n_total),
            "n_events": np.insert(event_counts, 0, 0),
            "n_censored": np.insert(censor_counts, 0, 0),
            "survival_prob": np.insert(survival_curve, 0, 1.0),
            "greenwood_var": np.insert(greenwood_variance, 0, 0.0),
            "ci_lower_95": np.insert(ci_lower, 0, 1.0),
            "ci_upper_95": np.insert(ci_upper, 0, 1.0),
        })

        self.timeline_ = df_table["timeline"].values
        self.survival_table_ = df_table
        self.is_fitted_ = True
        return self

    def predict_survival(self, t: float) -> float:
        """Returns S(t) = P(T > t) using right-continuous step interpolation."""
        if not self.is_fitted_ or self.survival_table_ is None:
            raise RuntimeError("Model must be fitted before predicting survival.")
        if t <= 0:
            return 1.0

        idx = np.searchsorted(self.timeline_, t, side="right") - 1
        idx = max(0, min(idx, len(self.timeline_) - 1))
        return float(self.survival_table_["survival_prob"].iloc[idx])

    def predict_hazard(self, t: float, delta_t: float = 60.0) -> float:
        """
        Approximates instantaneous hazard rate h(t) = - (1/S(t)) * dS/dt.
        h(t) ≈ (S(t) - S(t + delta_t)) / (delta_t * S(t)).
        """
        s_t = self.predict_survival(t)
        if s_t <= 1e-7:
            return 0.05  # High risk ceiling if near certainty of failure
        s_future = self.predict_survival(t + delta_t)
        hazard = (s_t - s_future) / (delta_t * s_t)
        return float(np.clip(hazard, 1e-6, 0.1))

    def calculate_mtbf(self, horizon_seconds: Optional[float] = None) -> float:
        """
        Calculates the true Mean Time Between Failures (MTBF) via numerical integration
        over the survival curve: MTBF = integral_0^tau S(t) dt.
        Avoids naive sample mean bias on right-censored datasets.
        """
        if not self.is_fitted_ or self.survival_table_ is None:
            raise RuntimeError("Model must be fitted before calculating MTBF.")

        table = self.survival_table_
        times = table["timeline"].values
        surv = table["survival_prob"].values

        tau = horizon_seconds if horizon_seconds is not None else float(times[-1])
        
        # Trapezoidal / rectangular step integration
        area = 0.0
        for i in range(len(times) - 1):
            t_curr = times[i]
            t_next = times[i + 1]
            if t_curr >= tau:
                break
            dt = min(t_next, tau) - t_curr
            area += surv[i] * dt

        return float(area)

    def actuarial_life_table(self, bin_width_seconds: float = 3600.0) -> pd.DataFrame:
        """
        Constructs an aggregated Actuarial Life Table grouped into discrete time intervals
        (e.g., 1-hour = 3600s bins) for executive reporting.
        """
        if not self.is_fitted_ or self.survival_table_ is None:
            raise RuntimeError("Model must be fitted before generating actuarial table.")

        max_t = self.timeline_[-1]
        bins = np.arange(0, max_t + bin_width_seconds, bin_width_seconds)
        
        intervals = []
        n_exposed = []
        w_censored = []
        d_evictions = []
        effective_exposed = []
        cond_eviction_prob = []
        interval_survival = []

        curr_surv = 1.0

        for i in range(len(bins) - 1):
            t_start = bins[i]
            t_end = bins[i + 1]
            label = f"[{int(t_start/3600)}h - {int(t_end/3600)}h)"

            # Filter table in interval
            mask = (self.survival_table_["timeline"] >= t_start) & (self.survival_table_["timeline"] < t_end)
            sub = self.survival_table_[mask]

            events_in_bin = int(sub["n_events"].sum())
            censored_in_bin = int(sub["n_censored"].sum())
            at_start = int(sub["n_at_risk"].iloc[0]) if len(sub) > 0 else 0

            # Actuarial adjustment: N_j* = N_j - W_j / 2
            n_star = max(0.0, at_start - (censored_in_bin / 2.0))
            q_j = (events_in_bin / n_star) if n_star > 0 else 0.0
            q_j = min(1.0, q_j)
            curr_surv *= (1.0 - q_j)

            intervals.append(label)
            n_exposed.append(at_start)
            w_censored.append(censored_in_bin)
            d_evictions.append(events_in_bin)
            effective_exposed.append(round(n_star, 1))
            cond_eviction_prob.append(round(q_j, 4))
            interval_survival.append(round(curr_surv, 4))

        return pd.DataFrame({
            "interval_hours": intervals,
            "instances_at_start": n_exposed,
            "evictions_observed": d_evictions,
            "censored_completions": w_censored,
            "effective_exposure": effective_exposed,
            "conditional_eviction_prob": cond_eviction_prob,
            "cumulative_survival": interval_survival,
        })


def log_rank_test(
    durations_1: np.ndarray,
    events_1: np.ndarray,
    durations_2: np.ndarray,
    events_2: np.ndarray,
    name1: str = "Pool_1",
    name2: str = "Pool_2",
) -> LogRankResult:
    """
    Executes a two-sample Log-Rank hypothesis test comparing the survival distributions
    of two Spot instance pools (e.g., c6i vs m6i or us-east-1 vs us-east-2).
    H0: S_1(t) = S_2(t) (Equal eviction hazard profiles).
    """
    d1 = np.asarray(durations_1, dtype=float)
    e1 = np.asarray(events_1, dtype=bool)
    d2 = np.asarray(durations_2, dtype=float)
    e2 = np.asarray(events_2, dtype=bool)

    # Combined distinct event times
    all_durations = np.concatenate([d1, d2])
    all_events = np.concatenate([e1, e2])
    unique_event_times = np.unique(all_durations[all_events])
    unique_event_times = np.sort(unique_event_times)

    o1_total = 0
    e1_total = 0.0
    v1_total = 0.0

    o2_total = 0
    e2_total = 0.0

    for t in unique_event_times:
        # Group 1 at risk and events at time t
        n1 = np.sum(d1 >= t)
        d1_t = np.sum((d1 == t) & e1)

        # Group 2 at risk and events at time t
        n2 = np.sum(d2 >= t)
        d2_t = np.sum((d2 == t) & e2)

        n_t = n1 + n2
        d_t = d1_t + d2_t

        if n_t <= 1 or d_t == 0:
            continue

        # Expected events under H0
        e1_t = n1 * (d_t / n_t)
        e2_t = n2 * (d_t / n_t)

        # Hypergeometric variance under H0
        var_t = (n1 * n2 * d_t * (n_t - d_t)) / ((n_t ** 2) * (n_t - 1))

        o1_total += d1_t
        e1_total += e1_t
        v1_total += var_t

        o2_total += d2_t
        e2_total += e2_t

    if v1_total <= 1e-9:
        chi2 = 0.0
        p_val = 1.0
    else:
        # Z = (O_1 - E_1) / sqrt(V_1)
        z = (o1_total - e1_total) / np.sqrt(v1_total)
        chi2 = float(z ** 2)
        p_val = float(1.0 - stats.chi2.cdf(chi2, df=1))

    return LogRankResult(
        group1_name=name1,
        group2_name=name2,
        observed_events_1=int(o1_total),
        expected_events_1=round(float(e1_total), 2),
        observed_events_2=int(o2_total),
        expected_events_2=round(float(e2_total), 2),
        chi2_statistic=round(chi2, 4),
        p_value=round(p_val, 6),
        is_significant=bool(p_val < 0.05),
    )
