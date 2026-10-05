"""
tests/benchmark.py - Quantitative Latency & Memory Benchmark.
Measures p50, p95, and p99 query latency on Hive-partitioned Parquet datasets.
Executes 30 iterations using time.perf_counter().
"""

import os
import sys
import time
import tempfile
from pathlib import Path
import numpy as np

# Robust path resolution for direct CLI execution and CI environments
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_generator import generate_spot_lifecycle_dataset
from src.adapters.duckdb_adapter import DuckDBAnalyticalAdapter


def run_benchmarks(iterations=30, num_records=10000):
    print(f"[Benchmark] Preparing dataset with {num_records:,} records...")
    with tempfile.TemporaryDirectory() as temp_dir:
        lake_path = os.path.join(temp_dir, "spot_events")
        generate_spot_lifecycle_dataset(
            num_records=num_records,
            base_output_dir=lake_path,
            random_seed=42,
        )
        
        adapter = DuckDBAnalyticalAdapter()
        parquet_glob = os.path.join(lake_path, "**", "*.parquet")
        
        query = f"""
            SELECT 
                region,
                instance_family,
                COUNT(*) as total_nodes,
                AVG(duration_seconds) as mean_lifetime_sec,
                AVG(CASE WHEN event_observed THEN 1.0 ELSE 0.0 END) as interruption_rate,
                AVG(ondemand_price_usd_per_hour - spot_price_usd_per_hour) as hourly_savings
            FROM read_parquet('{parquet_glob}', hive_partitioning = true)
            WHERE region = 'us-east-1'
            GROUP BY region, instance_family
            ORDER BY total_nodes DESC;
        """
        
        # Warmup
        adapter.execute_query(query)
        
        print(f"[Benchmark] Executing {iterations} iterations of Vectorized Analytical Query...")
        latencies = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            adapter.execute_query(query)
            latencies.append((time.perf_counter() - t0) * 1000)
            
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        
        # 2. Decision Engine Real-Time Inference Latency
        from src.core_engine import SpotAnalyticsEngine
        from src.domain.entities import RuntimeContext
        
        engine = SpotAnalyticsEngine(storage=adapter, lake_path=lake_path)
        cox_model = engine.train_cox_model(sample_limit=5000)
        
        ctx = RuntimeContext(
            instance_id="i-bench-01",
            job_id="job-bench-01",
            pipeline_type="SageMaker_Distributed_Train",
            instance_type="g5.4xlarge",
            region="us-east-1",
            duration_seconds=3600.0,
            current_hazard=0.0002,
            seconds_since_last_checkpoint=600.0,
            compute_cost_rate_per_sec=0.00045,
            decision_window_seconds=300.0,
        )
        
        decision_latencies = []
        for _ in range(iterations * 5):
            t0 = time.perf_counter()
            engine.evaluate_live_decision(context=ctx, trained_model=cox_model)
            decision_latencies.append((time.perf_counter() - t0) * 1000)
            
        dec_p50 = float(np.percentile(decision_latencies, 50))
        dec_p95 = float(np.percentile(decision_latencies, 95))
        dec_p99 = float(np.percentile(decision_latencies, 99))

        print("\n" + "="*65)
        print("  Cloud Architecture & FinOps Practice AWS SPOT SURVIVAL ENGINE - QUANTITATIVE BENCHMARKS")
        print("="*65)
        print(f"  Dataset Size: {num_records:,} spot task exposures")
        print(f"  Iterations:   {iterations}")
        print("\n  [1] Vectorized Lakehouse Aggregation Latency:")
        print(f"      - p50 Latency:  {p50:.2f} ms")
        print(f"      - p95 Latency:  {p95:.2f} ms")
        print(f"      - p99 Latency:  {p99:.2f} ms")
        print("\n  [2] Real-Time Causal Checkpoint Decision Inference:")
        print(f"      - p50 Latency:  {dec_p50:.3f} ms")
        print(f"      - p95 Latency:  {dec_p95:.3f} ms")
        print(f"      - p99 Latency:  {dec_p99:.3f} ms")
        print("="*65 + "\n")


if __name__ == "__main__":
    run_benchmarks()
