"""
src/data_generator.py - Calibrated Stochastic Data Generator for AWS Spot Lifecycle.
Generates realistic, physically sound Spot instance-task exposures (50,000+ records)
with right-censoring, Weibull base hazards, Ornstein-Uhlenbeck price drift,
and writes to a Hive-partitioned Apache Parquet data lake.
"""

import os
import time
import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds


def generate_spot_lifecycle_dataset(
    num_records: int = 50000,
    base_output_dir: str = "data/spot_events",
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic AWS Spot lifecycle records calibrated against AWS EC2 spot market behavior.
    """
    print(f"[Data Generator] Initiating stochastic generation of {num_records:,} AWS Spot exposures...")
    start_time = time.time()
    np.random.seed(random_seed)

    # 1. Regions and Availability Zones (Conditional)
    regions = ["us-east-1", "us-east-2", "us-west-2", "eu-west-1"]
    region_probs = [0.40, 0.25, 0.25, 0.10]
    assigned_regions = np.random.choice(regions, size=num_records, p=region_probs)

    az_suffixes = ["a", "b", "c"]
    assigned_azs = [
        f"{r}{np.random.choice(az_suffixes)}"
        for r in assigned_regions
    ]

    # 2. Instance Families and Types
    families = ["c6i", "m6i", "r6i", "g5"]
    family_probs = [0.35, 0.35, 0.15, 0.15]
    assigned_families = np.random.choice(families, size=num_records, p=family_probs)

    family_type_map = {
        "c6i": {
            "types": ["c6i.2xlarge", "c6i.4xlarge", "c6i.8xlarge"],
            "vcpu": [8, 16, 32],
            "mem": [16.0, 32.0, 64.0],
            "ondemand_rate": [0.34, 0.68, 1.36],
            "has_gpu": False,
        },
        "m6i": {
            "types": ["m6i.2xlarge", "m6i.4xlarge", "m6i.8xlarge"],
            "vcpu": [8, 16, 32],
            "mem": [32.0, 64.0, 128.0],
            "ondemand_rate": [0.384, 0.768, 1.536],
            "has_gpu": False,
        },
        "r6i": {
            "types": ["r6i.2xlarge", "r6i.4xlarge", "r6i.8xlarge"],
            "vcpu": [8, 16, 32],
            "mem": [64.0, 128.0, 256.0],
            "ondemand_rate": [0.504, 1.008, 2.016],
            "has_gpu": False,
        },
        "g5": {
            "types": ["g5.xlarge", "g5.2xlarge", "g5.4xlarge"],
            "vcpu": [4, 8, 16],
            "mem": [16.0, 32.0, 64.0],
            "ondemand_rate": [1.006, 1.212, 1.624],
            "has_gpu": True,
        },
    }

    assigned_types = []
    assigned_vcpus = []
    assigned_mems = []
    assigned_ondemand_prices = []
    has_gpu_list = []

    for fam in assigned_families:
        spec = family_type_map[fam]
        idx = np.random.choice(len(spec["types"]))
        assigned_types.append(spec["types"][idx])
        assigned_vcpus.append(spec["vcpu"][idx])
        assigned_mems.append(spec["mem"][idx])
        assigned_ondemand_prices.append(spec["ondemand_rate"][idx])
        has_gpu_list.append(spec["has_gpu"])

    assigned_ondemand_prices = np.array(assigned_ondemand_prices)

    # 3. Workload and Pipeline Types
    pipelines = [
        "SageMaker_Distributed_Train",
        "EMR_Spark_ETL",
        "Glue_Streaming_Job",
        "StepFunctions_Batch",
    ]
    pipeline_probs = [0.35, 0.35, 0.20, 0.10]
    assigned_pipelines = np.random.choice(pipelines, size=num_records, p=pipeline_probs)

    # 4. Spot Market Economics (Spread, Drift)
    # Spot price is typically 65-75% discount vs on-demand
    discount_ratio = np.random.beta(a=14, b=6, size=num_records)  # mean ~0.70 discount
    assigned_spot_prices = np.round(assigned_ondemand_prices * (1.0 - discount_ratio), 4)
    price_spread_ratios = np.round(assigned_spot_prices / assigned_ondemand_prices, 4)

    # Price drift across 15m window (AR(1) shock)
    price_drift_15m = np.round(np.random.normal(loc=0.01, scale=0.04, size=num_records), 4)

    # 5. Node Resource Telemetry
    cpu_utils = np.round(np.random.beta(a=5, b=2, size=num_records), 4)
    gpu_utils = np.where(
        has_gpu_list,
        np.round(np.random.beta(a=6, b=2, size=num_records), 4),
        0.0,
    )
    cluster_sizes = np.random.poisson(lam=12, size=num_records) + 2  # 2 to 40 nodes

    # 6. Survival Physics: Weibull Hazard & Right-Censoring
    # h0(t) = (k/lambda) * (t/lambda)^(k-1)
    # k = 1.35 (positive aging hazard: risk accumulates with market shifts)
    # lambda = 21600 seconds (6 hours characteristic lifespan under pressure)
    k_shape = 1.35
    lambda_scale = 21600.0

    # Covariate risk coefficients beta
    # Higher price drift, higher cluster size, higher GPU usage increase eviction risk
    beta_drift = 2.5
    beta_spread = 1.8
    beta_cluster = 0.02
    beta_gpu = 0.4

    # Family relative hazard adjustment
    family_beta = {"c6i": 0.0, "m6i": -0.15, "r6i": 0.10, "g5": 0.50}
    fam_risk = np.array([family_beta[f] for f in assigned_families])

    # Region hazard adjustment
    region_beta = {"us-east-1": 0.20, "us-east-2": -0.05, "us-west-2": 0.0, "eu-west-1": -0.15}
    reg_risk = np.array([region_beta[r] for r in assigned_regions])

    eta = (
        beta_drift * price_drift_15m
        + beta_spread * price_spread_ratios
        + beta_cluster * (cluster_sizes - 10)
        + beta_gpu * gpu_utils
        + fam_risk
        + reg_risk
    )
    # Inversion method for Weibull: T = lambda * (-log(U) / exp(eta))^(1/k)
    uniform_rv = np.random.uniform(low=1e-5, high=0.99999, size=num_records)
    t_eviction = lambda_scale * ((-np.log(uniform_rv)) / np.exp(eta)) ** (1.0 / k_shape)

    # Job target duration (Planned job completion time)
    # Distributed trains: 3h-12h; ETL: 1h-5h; Glue: 30m-3h; StepFunc: 15m-1h
    pipeline_duration_params = {
        "SageMaker_Distributed_Train": (14400.0, 7200.0),
        "EMR_Spark_ETL": (9000.0, 3600.0),
        "Glue_Streaming_Job": (5400.0, 1800.0),
        "StepFunctions_Batch": (2400.0, 900.0),
    }

    t_planned = np.zeros(num_records)
    for p_name, (mean_d, std_d) in pipeline_duration_params.items():
        mask = assigned_pipelines == p_name
        count = int(np.sum(mask))
        if count > 0:
            durations = np.random.normal(loc=mean_d, scale=std_d, size=count)
            t_planned[mask] = np.clip(durations, a_min=600.0, a_max=43200.0)

    # Observed exposure duration T_i and indicator delta_i
    # delta_i = 1 if evicted, 0 if job finished before eviction (Right-Censored)
    is_evicted = t_eviction <= t_planned
    observed_durations = np.where(is_evicted, t_eviction, t_planned)
    observed_durations = np.round(np.clip(observed_durations, a_min=60.0, a_max=86400.0), 2)
    event_observed = is_evicted.astype(bool)
    event_type = np.where(event_observed, "EVICTION", "COMPLETED")

    # 7. AWS Early Warning Signals
    # Rebalance recommendation: arrives with 75% prob for evictions ~15-30m prior, 5% false alarms
    rebalance_recommended = np.where(
        event_observed,
        np.random.binomial(n=1, p=0.75, size=num_records) == 1,
        np.random.binomial(n=1, p=0.05, size=num_records) == 1,
    )
    # Interruption notice: guaranteed 120s notice before eviction
    interruption_notice = event_observed.copy()

    # 8. Checkpoint Telemetry & Replay Sunk Costs
    # Checkpoints performed periodically (60s to 600s interval)
    checkpoint_ages = np.round(np.random.uniform(low=60.0, high=600.0, size=num_records), 2)
    checkpoint_sizes = np.random.lognormal(mean=20.5, sigma=0.8, size=num_records).astype(np.int64)  # ~500MB to 5GB

    # Replay cost: compute burned since last checkpoint + fixed reload
    compute_rate_sec = assigned_spot_prices / 3600.0
    restart_reload_overhead = 0.25  # USD fixed reload penalty
    replay_costs = np.where(
        event_observed,
        np.round((checkpoint_ages * compute_rate_sec) + restart_reload_overhead, 4),
        0.0,
    )

    # 9. Temporal Metadata (2026 Partitioning)
    base_date = datetime(2026, 1, 1, 0, 0, 0)
    time_offsets_sec = np.random.uniform(0, 180 * 86400, size=num_records)  # 6 months span
    launch_datetimes = [base_date + timedelta(seconds=float(s)) for s in time_offsets_sec]
    terminal_datetimes = [
        ld + timedelta(seconds=float(d))
        for ld, d in zip(launch_datetimes, observed_durations)
    ]

    event_ids = [f"evt-{i:07d}" for i in range(1, num_records + 1)]
    instance_ids = [f"i-{hash(f'{i}_{r}') & 0xFFFFFFFF:08x}" for i, r in enumerate(assigned_regions)]
    job_ids = [f"job-{i:06d}" for i in range(1, num_records + 1)]

    years = [dt.year for dt in launch_datetimes]
    months = [dt.month for dt in launch_datetimes]

    df = pd.DataFrame({
        "event_id": event_ids,
        "instance_id": instance_ids,
        "job_id": job_ids,
        "pipeline_type": assigned_pipelines,
        "launch_ts": [dt.isoformat() for dt in launch_datetimes],
        "terminal_ts": [dt.isoformat() for dt in terminal_datetimes],
        "duration_seconds": observed_durations,
        "event_observed": event_observed,
        "event_type": event_type,
        "region": assigned_regions,
        "availability_zone": assigned_azs,
        "instance_type": assigned_types,
        "instance_family": assigned_families,
        "vcpu": assigned_vcpus,
        "memory_gib": assigned_mems,
        "has_gpu": has_gpu_list,
        "spot_price_usd_per_hour": assigned_spot_prices,
        "ondemand_price_usd_per_hour": np.round(assigned_ondemand_prices, 4),
        "price_spread_ratio": price_spread_ratios,
        "price_drift_15m": price_drift_15m,
        "cpu_utilization": cpu_utils,
        "gpu_utilization": gpu_utils,
        "cluster_size": cluster_sizes,
        "checkpoint_age_seconds": checkpoint_ages,
        "checkpoint_size_bytes": checkpoint_sizes,
        "replay_cost_usd": replay_costs,
        "rebalance_recommended": rebalance_recommended,
        "interruption_notice": interruption_notice,
        "year": years,
        "month": months,
    })

    # 10. Physical Hive-Partitioned Parquet Lake Persistence
    os.makedirs(base_output_dir, exist_ok=True)
    table = pa.Table.from_pandas(df)
    
    # Write dataset partitioned by region / instance_family / year / month
    partition_cols = ["region", "instance_family", "year", "month"]
    ds.write_dataset(
        data=table,
        base_dir=base_output_dir,
        format="parquet",
        partitioning=ds.partitioning(
            schema=pa.schema([
                ("region", pa.string()),
                ("instance_family", pa.string()),
                ("year", pa.int64()),
                ("month", pa.int64()),
            ]),
            flavor="hive",
        ),
        existing_data_behavior="overwrite_or_ignore",
    )

    elapsed = time.time() - start_time
    eviction_rate = float(np.mean(event_observed)) * 100.0
    print(f"[Data Generator] Successfully generated {len(df):,} records in {elapsed:.2f}s.")
    print(f"                 - Interruption Rate (Evictions): {eviction_rate:.2f}%")
    print(f"                 - Right-Censored Completions:   {100.0 - eviction_rate:.2f}%")
    print(f"                 - Hive Partition Lake Root:     {base_output_dir}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic AWS Spot lifecycle dataset.")
    parser.add_argument("--records", type=int, default=50000, help="Number of records to generate.")
    parser.add_argument("--output", type=str, default="data/spot_events", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    generate_spot_lifecycle_dataset(
        num_records=args.records,
        base_output_dir=args.output,
        random_seed=args.seed,
    )
