#!/usr/bin/env python3
"""
============================================================================
EMPIRICIST EXPERIMENT - MAIN RUNNER
============================================================================
This script runs the complete experiment in one go.

Usage:
    python run_experiment.py

Or run steps individually:
    python step2_load_data.py
    python step3_fast_tokenization.py
    python step4_hypothesis_test.py

Hypothesis:
    "FAST+ universal tokenizer achieves ≥80% of the compression ratio
     of a custom FAST tokenizer on cable manipulation data."

============================================================================
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script_name: str):
    """Run a step script."""
    print()
    print("#" * 70)
    print(f"# Running: {script_name}")
    print("#" * 70)
    print()
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with return code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run FAST+ experiment pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-step2", action="store_true", help="Skip data loading step.")
    group.add_argument("--force-step2", action="store_true", help="Force re-run of data loading step.")
    args = parser.parse_args()

    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " Testing FAST+ Generalization to Cable Manipulation ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    print("Hypothesis:")
    print('  "FAST+ universal tokenizer achieves ≥80% of the compression')
    print('   ratio of a custom FAST tokenizer on cable manipulation data."')
    print()
    print("This experiment will:")
    print("  1. Load cable manipulation data (lerobot/berkeley_cable_routing)")
    print("  2. Compare FAST (custom) vs FAST+ (universal) vs Naive tokenization")
    print("  3. Test the hypothesis and provide recommendations")
    print()
    input("Press Enter to start...")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    # Create directories
    Path("./data").mkdir(exist_ok=True)
    Path("./results").mkdir(exist_ok=True)
    
    # Decide whether to run step2
    data_path = Path("./data/cable_trajectories.npz")
    data_exists = data_path.exists()
    if args.skip_step2:
        if not data_exists:
            print(f"ERROR: {data_path} not found. Cannot skip step2.")
            sys.exit(1)
        run_step2 = False
    elif args.force_step2:
        run_step2 = True
    else:
        run_step2 = not data_exists
        if data_exists:
            print(f"Found {data_path}; skipping step2. Use --force-step2 to rerun.")

    # Run each step
    if run_step2:
        run_step("step2_load_data.py")
    run_step("step3_fast_tokenization.py")
    run_step("step4_hypothesis_test.py")
    
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " EXPERIMENT COMPLETE ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")


if __name__ == "__main__":
    main()
