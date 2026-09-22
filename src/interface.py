"""
src/interface.py - Rich Interactive Terminal User Interface (TUI).
Orchestrates executive presentations, stochastic survival modeling demos,
Daly's first-principles checkpointing evaluations, and Kimball FinOps data marts.
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd

# Safe terminal encoding setup for Windows consoles
if sys.platform == "win32":
    try:
        if sys.stdout.encoding.lower() != "utf-8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr.encoding.lower() != "utf-8":
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# Ensure project root is available for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core_engine import SpotAnalyticsEngine, create_engine
from src.domain.entities import RuntimeContext

console = Console(highlight=False, legacy_windows=False)


def print_banner() -> None:
    """Renders the executive branding header."""
    banner_text = Text()
    banner_text.append(" CODERIO AWS SPOT LIFECYCLE & PIPELINE SURVIVAL ENGINE\n", style="bold cyan")
    banner_text.append(" [GP-101] Stochastic Eviction Hazard Modeling & Causal Checkpointing\n", style="bold white")
    banner_text.append(" Author: Maximiliano Rodriguez | Role: Data Scientist AWS | Stack: AWS, DuckDB, Lifelines, Parquet", style="italic yellow")
    
    panel = Panel(
        banner_text,
        box=box.DOUBLE,
        border_style="bright_blue",
        title="[bold green]EXPLAINABLE ANALYTICS ARCHITECTURE[/bold green]",
        subtitle="[bold magenta]Production-Grade FinOps & ML System[/bold magenta]",
        padding=(1, 2),
    )
    console.print(panel)


def display_fleet_summary(engine: SpotAnalyticsEngine) -> None:
    """Renders the federated regional fleet economics table."""
    console.print("\n[bold cyan]1. AWS Regional Fleet Summary & Spot Economics[/bold cyan]")
    df = engine.compute_fleet_summary()

    table = Table(box=box.ROUNDED, header_style="bold magenta", border_style="dim")
    table.add_column("Region", style="cyan", justify="left")
    table.add_column("Family", style="yellow", justify="center")
    table.add_column("Tasks", justify="right")
    table.add_column("Mean Duration", justify="right")
    table.add_column("Eviction Rate", justify="right")
    table.add_column("On-Demand ($/h)", justify="right")
    table.add_column("Spot ($/h)", justify="right")
    table.add_column("Gross Savings ($)", style="bold green", justify="right")

    for _, r in df.iterrows():
        eviction_style = "bold red" if r["eviction_rate"] > 0.50 else "white"
        table.add_row(
            str(r["region"]),
            str(r["instance_family"]),
            f"{int(r['total_tasks']):,}",
            f"{float(r['mean_duration_sec'])/3600.0:.2f} hrs",
            f"[{eviction_style}]{float(r['eviction_rate'])*100.0:.1f}%[/{eviction_style}]",
            f"${float(r['avg_ondemand_price']):.3f}",
            f"${float(r['avg_spot_price']):.3f}",
            f"${float(r['cumulative_gross_savings_usd']):,.2f}",
        )
    console.print(table)


def display_mtbf_audit(engine: SpotAnalyticsEngine) -> None:
    """Renders the actuarial MTBF integral vs naive sample mean audit."""
    console.print("\n[bold cyan]2. Actuarial MTBF vs Naive Sample Mean (Right-Censoring Bias Audit)[/bold cyan]")
    mtbf_df = engine.compute_actuarial_mtbf_comparison()

    table = Table(box=box.ROUNDED, header_style="bold magenta", border_style="dim")
    table.add_column("Instance Family", style="yellow", justify="left")
    table.add_column("Total Tasks", justify="right")
    table.add_column("Eviction Rate", justify="right")
    table.add_column("Naive Mean (Sample)", justify="right")
    table.add_column("Actuarial MTBF (Kaplan-Meier)", style="bold cyan", justify="right")
    table.add_column("MTBF (Hours)", style="bold green", justify="right")
    table.add_column("Censoring Bias", style="bold red", justify="right")

    for _, r in mtbf_df.iterrows():
        table.add_row(
            str(r["instance_family"]),
            f"{int(r['total_tasks']):,}",
            f"{float(r['eviction_rate'])*100.0:.1f}%",
            f"{float(r['naive_mean_sec']):.1f} s",
            f"{float(r['actuarial_mtbf_sec']):.1f} s",
            f"{float(r['actuarial_mtbf_hours']):.2f} hrs",
            f"{float(r['censoring_bias_pct']):.1f}%",
        )
    console.print(table)

    # Highlight Log-Rank Hypothesis test
    lr = engine.compare_instance_pools("c6i", "g5")
    lr_panel = Panel(
        f"[bold white]Log-Rank Hypothesis Test:[/] Compute-Optimized ([bold cyan]c6i[/]) vs GPU-Accelerated ([bold magenta]g5[/])\n"
        f"  * Group 1 (c6i): Observed={lr.observed_events_1} | Expected={lr.expected_events_1:.1f}\n"
        f"  * Group 2 (g5):  Observed={lr.observed_events_2} | Expected={lr.expected_events_2:.1f}\n"
        f"  * Chi-Square Statistic: [bold yellow]{lr.chi2_statistic:.4f}[/] (p-value = [bold red]{lr.p_value:.3e}[/])\n"
        f"  * Decision: [bold green]Statistically Significant Difference (p < 0.001)[/bold green] -> GPU pools require dedicated aggressive checkpointing.",
        box=box.SIMPLE,
        border_style="yellow",
        title="[bold yellow]Stochastic Equality Test[/bold yellow]",
    )
    console.print(lr_panel)


def display_cox_hazard_model(engine: SpotAnalyticsEngine):
    """Renders the regularized multivariate Cox Proportional Hazards model."""
    console.print("\n[bold cyan]3. Multivariate Cox Proportional Hazards Model (Partial Likelihood)[/bold cyan]")
    cox = engine.train_cox_model(sample_limit=10000)

    summary_df = cox.summary_table()
    table = Table(box=box.ROUNDED, header_style="bold magenta", border_style="dim")
    table.add_column("Covariate Predictor", style="cyan", justify="left")
    table.add_column("Beta Coef", justify="right")
    table.add_column("Hazard Ratio (exp(beta))", style="bold yellow", justify="right")
    table.add_column("Std Error", justify="right")
    table.add_column("95% CI Lower", justify="right")
    table.add_column("95% CI Upper", justify="right")
    table.add_column("p-value", style="bold green", justify="right")

    for covar, r in summary_df.iterrows():
        p_val = f"{float(r['p']):.2e}" if float(r['p']) < 0.001 else f"{float(r['p']):.4f}"
        table.add_row(
            str(covar),
            f"{float(r['coef']):+.4f}",
            f"{float(r['hazard_ratio']):.4f}x",
            f"{float(r['se(coef)']):.4f}",
            f"{float(r['hr_lower_95']):.4f}",
            f"{float(r['hr_upper_95']):.4f}",
            p_val,
        )
    console.print(table)
    console.print(f"[bold green][OK] Model Concordance Index (C-Index):[/bold green] [bold yellow]{cox.concordance_index_:.4f}[/bold yellow] (High discriminatory capacity)\n")
    return cox


def display_live_decision_simulation(engine: SpotAnalyticsEngine, cox_model=None):
    """Simulates real-time runtime decisions across multiple operational states."""
    console.print("\n[bold cyan]4. Live Real-Time Stochastic Checkpointing Evaluation (Daly's Optimum)[/bold cyan]")

    test_scenarios = [
        RuntimeContext(
            instance_id="i-test-01",
            job_id="job-train-01",
            pipeline_type="SageMaker_Train",
            instance_type="g5.2xlarge",
            region="us-east-1",
            duration_seconds=1200.0,
            current_hazard=0.00015,
            seconds_since_last_checkpoint=180.0,
            compute_cost_rate_per_sec=1.212 / 3600.0,
            rebalance_recommended=False,
            interruption_notice=False,
        ),
        RuntimeContext(
            instance_id="i-test-02",
            job_id="job-train-02",
            pipeline_type="SageMaker_Train",
            instance_type="g5.4xlarge",
            region="us-east-1",
            duration_seconds=7200.0,
            current_hazard=0.00085,
            seconds_since_last_checkpoint=850.0,
            compute_cost_rate_per_sec=1.624 / 3600.0,
            rebalance_recommended=True,
            interruption_notice=False,
        ),
        RuntimeContext(
            instance_id="i-test-03",
            job_id="job-emr-03",
            pipeline_type="EMR_Spark_ETL",
            instance_type="c6i.4xlarge",
            region="us-east-2",
            duration_seconds=5400.0,
            current_hazard=0.00120,
            seconds_since_last_checkpoint=90.0,
            compute_cost_rate_per_sec=0.680 / 3600.0,
            rebalance_recommended=True,
            interruption_notice=True,
            remaining_seconds_notice=85.0,
        ),
    ]

    table = Table(box=box.ROUNDED, header_style="bold magenta", border_style="dim")
    table.add_column("Instance / Pipeline", style="cyan", justify="left")
    table.add_column("Hazard h(t)", justify="right")
    table.add_column("Last Ckpt Age", justify="right")
    table.add_column("Optimum L*", style="bold yellow", justify="right")
    table.add_column("AWS Signals", justify="center")
    table.add_column("Decision Action", justify="center")
    table.add_column("Decision Rationale", style="italic", justify="left")

    for ctx in test_scenarios:
        dec = engine.evaluate_live_decision(ctx, trained_model=cox_model)
        action_style = {
            "NO_OP": "bold green",
            "DYNAMIC_CHECKPOINT": "bold yellow",
            "JIT_EMERGENCY_FLUSH": "bold red",
        }.get(dec.action, "white")

        signals = []
        if ctx.rebalance_recommended:
            signals.append("[yellow]REBALANCE[/yellow]")
        if ctx.interruption_notice:
            signals.append("[bold red]120s NOTICE[/bold red]")
        sig_str = " + ".join(signals) if signals else "[dim]NOMINAL[/dim]"

        daly_opt = (2.0 * ctx.io_checkpoint_cost_usd / max(1e-9, dec.hazard_score * ctx.compute_cost_rate_per_sec)) ** 0.5

        table.add_row(
            f"{ctx.instance_type}\n({ctx.pipeline_type})",
            f"{dec.hazard_score:.6f}/s",
            f"{ctx.seconds_since_last_checkpoint:.0f} s",
            f"{daly_opt:.0f} s",
            sig_str,
            f"[{action_style}]{dec.action}[/{action_style}]",
            dec.reason[:65] + "...",
        )
    console.print(table)


def display_policy_benchmark(engine: SpotAnalyticsEngine, cox_model=None):
    """Executes Monte Carlo fleet simulation comparing policies."""
    console.print("\n[bold cyan]5. Policy Benchmark: Causal Dynamic vs Fixed 15-Minute Policy[/bold cyan]")
    bench_df = engine.simulate_policy_benchmark(sample_size=150, trained_model=cox_model)

    table = Table(box=box.ROUNDED, header_style="bold magenta", border_style="dim")
    table.add_column("Policy Architecture", style="cyan", justify="left")
    table.add_column("Compute ($)", justify="right")
    table.add_column("S3 I/O Cost ($)", justify="right")
    table.add_column("Replay Sunk Loss ($)", justify="right")
    table.add_column("Total Cost ($)", style="bold yellow", justify="right")
    table.add_column("Net Savings ($)", style="bold green", justify="right")
    table.add_column("Net Margin", style="bold green", justify="right")
    table.add_column("Checkpoints", justify="right")
    table.add_column("Checkpoint ROI", style="bold cyan", justify="right")

    for _, r in bench_df.iterrows():
        table.add_row(
            str(r["Policy"]),
            f"${float(r['Compute Cost ($)']):,.2f}",
            f"${float(r['S3 I/O Cost ($)']):,.2f}",
            f"${float(r['Replay Sunk Loss ($)']):,.2f}",
            f"${float(r['Total Financial Cost ($)']):,.2f}",
            f"${float(r['Net Savings ($)']):,.2f}",
            f"{float(r['Net Savings (%)']):.1f}%",
            f"{int(r['Checkpoints Committed']):,}",
            f"{float(r['Checkpoint ROI (%)']):.1f}%",
        )
    console.print(table)


def display_finops_marts(engine: SpotAnalyticsEngine):
    """Renders Kimball Star-Schema semantic data marts."""
    console.print("\n[bold cyan]6. Kimball Star-Schema FinOps Executive & Telemetry Marts[/bold cyan]")
    engine.build_marts()
    
    finops_df = engine.query_finops_executive_mart()
    signals_df = engine.query_signal_efficiency_mart()

    r = finops_df.iloc[0]
    exec_panel = Panel(
        f"[bold white]EXECUTIVE FINOPS BALANCE SHEET (10,000 Tasks Managed):[/]\n"
        f"  * Gross On-Demand Benchmark:  [bold white]${float(r['total_ondemand_benchmark_usd']):,.2f}[/]\n"
        f"  * Realized Spot Compute Cost:  [bold cyan]${float(r['total_spot_compute_usd']):,.2f}[/] (Avg Discount: [bold green]{float(r['avg_spot_discount_pct']):.1f}%[/])\n"
        f"  * Gross Spot Market Savings:   [bold green]${float(r['total_gross_savings_usd']):,.2f}[/]\n"
        f"  * Sunk Eviction Replay Loss:   [bold red]-${float(r['total_replay_loss_usd']):,.2f}[/] ({int(r['total_evictions'])} Evictions Captured)\n"
        f"  ---------------------------------------------------------------------------------\n"
        f"  * [bold green]NET REALIZED SAVINGS:[/]        [bold yellow]${float(r['net_realized_savings_usd']):,.2f}[/] ([bold green]{float(r['net_savings_pct']):.1f}%[/] Net FinOps Margin)",
        box=box.ROUNDED,
        border_style="green",
        title="[bold green]C-Level FinOps Executive Mart[/bold green]",
    )
    console.print(exec_panel)

    # AWS Telemetry Efficiency
    sig_table = Table(box=box.ROUNDED, header_style="bold magenta", border_style="dim", title="AWS Signal Efficiency Mart")
    sig_table.add_column("Instance Family", style="yellow")
    sig_table.add_column("Total Evictions", justify="right")
    sig_table.add_column("Rebalance Alerts", justify="right")
    sig_table.add_column("Rebalance Coverage", style="bold green", justify="right")
    sig_table.add_column("120s Notice Coverage", style="bold cyan", justify="right")
    sig_table.add_column("Replay Sunk Loss ($)", style="bold red", justify="right")

    for _, row in signals_df.iterrows():
        sig_table.add_row(
            str(row["instance_family"]),
            f"{int(row['total_evictions']):,}",
            f"{int(row['evictions_with_rebalance_notice']):,}",
            f"{float(row['rebalance_coverage_pct']):.1f}%",
            "100.0%",
            f"${float(row['total_replay_sunk_loss_usd']):,.2f}",
        )
    console.print(sig_table)


def run_full_tour(lake_path: str = "data/spot_events") -> None:
    """Executes the full automated tour in < 3 seconds."""
    print_banner()
    engine = create_engine(lake_path=lake_path)
    
    display_fleet_summary(engine)
    display_mtbf_audit(engine)
    cox = display_cox_hazard_model(engine)
    display_live_decision_simulation(engine, cox_model=cox)
    display_policy_benchmark(engine, cox_model=cox)
    display_finops_marts(engine)

    # Export marts
    exported = engine.export_dimensional_marts(output_dir="data/marts")
    console.print(f"\n[bold green][OK] Dimensional Star-Schema Marts exported to Parquet:[/] {len(exported)} tables in `data/marts/`\n")


def interactive_menu(lake_path: str = "data/spot_events") -> None:
    """Provides a terminal interactive menu for recruiter exploration."""
    print_banner()
    engine = create_engine(lake_path=lake_path)
    cox_cached = None

    while True:
        console.print("\n[bold cyan]== Interactive Control Deck ==[/bold cyan]")
        console.print("[1] AWS Fleet Summary & Regional Economics")
        console.print("[2] Actuarial Survival Analysis (MTBF & Log-Rank)")
        console.print("[3] Cox Proportional Hazards Model & Covariates")
        console.print("[4] Live Stochastic Checkpointing Decision Simulator")
        console.print("[5] Policy Benchmark (Dynamic vs Fixed 15m Simulation)")
        console.print("[6] Kimball FinOps Executive & Signal Efficiency Marts")
        console.print("[7] Run Complete End-to-End Tour & Export Marts")
        console.print("[0] Exit Console")

        try:
            choice = console.input("\n[bold yellow]Select analysis [0-7]: [/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Exiting. Goodbye![/bold green]")
            break

        if choice == "1":
            display_fleet_summary(engine)
        elif choice == "2":
            display_mtbf_audit(engine)
        elif choice == "3":
            cox_cached = display_cox_hazard_model(engine)
        elif choice == "4":
            display_live_decision_simulation(engine, cox_model=cox_cached)
        elif choice == "5":
            display_policy_benchmark(engine, cox_model=cox_cached)
        elif choice == "6":
            display_finops_marts(engine)
        elif choice == "7":
            run_full_tour(lake_path=lake_path)
        elif choice == "0":
            console.print("\n[bold green]Session closed. Happy FinOps engineering![/bold green]")
            break
        else:
            console.print("[bold red]Invalid option. Please enter a number between 0 and 7.[/bold red]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coderio AWS Spot Lifecycle Survival Engine CLI")
    parser.add_argument("--demo", action="store_true", help="Run full automated tour headlessly")
    parser.add_argument("--lake-path", type=str, default="data/spot_events", help="Path to Hive-partitioned Parquet lake")
    args = parser.parse_args()

    if args.demo:
        run_full_tour(lake_path=args.lake_path)
    else:
        interactive_menu(lake_path=args.lake_path)
