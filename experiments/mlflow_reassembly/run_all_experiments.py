"""
Main script to run all 8 experiments.

Runs the full experimental matrix:
[LowRank (quantile 0.5, one_time)] x [Adam, AdamW] x [FlatLLM, TransMLA] x [sLM, mLM]

Usage:
    python run_all_experiments.py                    # All 8 experiments
    python run_all_experiments.py --first 2          # First 2
    python run_all_experiments.py --experiment 0     # Specific experiment
    python run_all_experiments.py --dry-run          # Show list without running
"""

# Configure matplotlib backend
import os
os.environ['MPLBACKEND'] = 'Agg'

import argparse
import sys
import time
from pathlib import Path
import torch

# CRITICAL: Fix PyTorch memory fragmentation
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from mlflow_experiments_config import generate_experiment_configs, print_experiment_summary
from mlflow_experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description="Run FedCore compression experiments")
    parser.add_argument('--first', type=int, help="Run only first N experiments")
    parser.add_argument('--experiment', type=int, help="Run specific experiment by index (0-7)")
    parser.add_argument('--dry-run', action='store_true', help="Show experiments without running")
    parser.add_argument('--continue-on-error', action='store_true', help="Continue if experiment fails")
    
    args = parser.parse_args()
    
    # Generate all configurations
    configs = generate_experiment_configs()
    
    # Show summary
    print_experiment_summary(configs)
    
    # Dry run
    if args.dry_run:
        print("Dry run mode - not executing experiments")
        return 0
    
    # Determine which experiments to run
    if args.experiment is not None:
        if 0 <= args.experiment < len(configs):
            configs_to_run = [configs[args.experiment]]
            print(f"\nRunning single experiment #{args.experiment}: {configs[args.experiment].experiment_name}")
        else:
            print(f"Error: Experiment index {args.experiment} out of range (0-{len(configs)-1})")
            return 1
    elif args.first:
        configs_to_run = configs[:args.first]
        print(f"\nRunning first {args.first} experiments")
    else:
        configs_to_run = configs
        print(f"\nRunning all {len(configs)} experiments")
    
    print()
    input("Press Enter to start experiments (or Ctrl+C to cancel)...")
    print()
    
    # Run experiments
    total_start_time = time.time()
    results = []
    
    for i, config in enumerate(configs_to_run, 1):
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {i}/{len(configs_to_run)}")
        print("=" * 80)
        print()
        
        # CRITICAL: Clear GPU memory between experiments
        if i > 1:  # Not before the first experiment
            print("Clearing GPU cache between experiments...")
            if torch.cuda.is_available():
                import gc
                torch.cuda.empty_cache()
                gc.collect()
            print()
        
        try:
            runner = ExperimentRunner(config)
            runner.run()
            
            results.append({
                'experiment': config.experiment_name,
                'status': 'success'
            })
            
        except KeyboardInterrupt:
            print("\n\n Interrupted by user!")
            print("Stopping experiments...")
            break
            
        except Exception as e:
            print(f"\n Experiment failed: {e}")
            
            results.append({
                'experiment': config.experiment_name,
                'status': 'failed',
                'error': str(e)
            })
            
            # CRITICAL: Clear memory even after failure
            if torch.cuda.is_available():
                print("Cleaning GPU memory after error...")
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            if not args.continue_on_error:
                print("\nStopping on error (use --continue-on-error to continue)")
                break
            else:
                print("\nContinuing to next experiment...")
    
    # Final statistics
    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()
    print(f"Total experiments run: {len(results)}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print()
    
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    
    if results:
        print("Detailed results:")
        for i, result in enumerate(results, 1):
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            print(f"  {i}. {status_emoji} {result['experiment']}")
            if result['status'] == 'failed':
                print(f"      Error: {result.get('error', 'Unknown')}")
        print()
    
    print("=" * 80)
    print()
    print(f"View all results at: http://localhost:5000")
    print()
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

