#!/usr/bin/env python3
"""
Test script to verify cached encoding optimization and profile performance.

Usage:
    python test_cache_performance.py --with-cache    # Test with caching enabled
    python test_cache_performance.py --no-cache      # Test without caching (baseline)
    python test_cache_performance.py --validate      # Test with validation enabled
"""

import argparse
import time
import os
import sys

# Set cache directory
os.environ['METAMON_CACHE_DIR'] = '/home/eddie/metamon_cache'


def run_training_steps(num_steps=10, with_validation=False):
    """Run a few training steps and measure time."""
    import torch
    from metamon.rl.finetune_from_hf import main as finetune_main

    # Enable validation if requested
    if with_validation:
        os.environ['METAMON_VALIDATE_CACHE'] = '1'
        print("[INFO] Cache validation ENABLED")
    else:
        os.environ['METAMON_VALIDATE_CACHE'] = '0'

    # Prepare arguments for training
    sys.argv = [
        'test_cache_performance.py',
        '--model_config', 'synthetic_multitaskagent.gin',
        '--train_config', 'vanilla_selfplay_damped.gin',
        '--init_from_checkpoint', 'SyntheticRLV2',
        '--battle_format', 'gen1ou',
        '--team_set', 'modern_replays_v2',
        '--num_envs', '5',
        '--parsed_replay_dir', '/home/eddie/metamon_cache/parsed-replays',
        '--formats', 'gen1ou',
        '--buffer_dir', '/home/eddie/metamon/other/nash_phase0/trajectories',
        '--log_to_wandb', 'False',
        '--save_dir', '/tmp/metamon_cache_test',
        '--val_interval', '9999999',  # Disable validation runs
        '--max_steps', str(num_steps),  # Only run a few steps
    ]

    start_time = time.time()

    try:
        finetune_main()
    except SystemExit:
        pass  # Training will exit after max_steps
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    elapsed = time.time() - start_time
    return elapsed


def test_with_cache():
    """Test with caching enabled (default MetamonMultiTaskAgent)."""
    print("\n" + "="*70)
    print("Testing WITH caching (MetamonMultiTaskAgent)")
    print("="*70)

    elapsed = run_training_steps(num_steps=10)
    if elapsed:
        print(f"\n[RESULT] Time with caching: {elapsed:.2f}s ({elapsed/10:.3f}s per step)")
    return elapsed


def test_without_cache():
    """Test without caching by temporarily modifying the config."""
    print("\n" + "="*70)
    print("Testing WITHOUT caching (reverting to MultiTaskAgent)")
    print("="*70)

    # Temporarily modify the gin config to use base MultiTaskAgent
    import shutil
    config_path = '/home/eddie/repos/metamon/metamon/rl/configs/models/synthetic_multitaskagent.gin'
    backup_path = config_path + '.backup'

    try:
        # Backup original config
        shutil.copy(config_path, backup_path)

        # Read config and replace MetamonMultiTaskAgent with MultiTaskAgent
        with open(config_path, 'r') as f:
            content = f.read()

        modified = content.replace('MetamonMultiTaskAgent', 'agent.MultiTaskAgent')

        with open(config_path, 'w') as f:
            f.write(modified)

        # Run training
        elapsed = run_training_steps(num_steps=10)
        if elapsed:
            print(f"\n[RESULT] Time without caching: {elapsed:.2f}s ({elapsed/10:.3f}s per step)")

        return elapsed

    finally:
        # Restore original config
        shutil.move(backup_path, config_path)
        print("[INFO] Restored original config")


def test_with_validation():
    """Test with cache validation enabled."""
    print("\n" + "="*70)
    print("Testing WITH validation (checking cache correctness)")
    print("="*70)

    elapsed = run_training_steps(num_steps=5, with_validation=True)
    if elapsed:
        print(f"\n[RESULT] Time with validation: {elapsed:.2f}s ({elapsed/5:.3f}s per step)")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description='Test cache performance optimization')
    parser.add_argument('--with-cache', action='store_true', help='Test with caching')
    parser.add_argument('--no-cache', action='store_true', help='Test without caching')
    parser.add_argument('--validate', action='store_true', help='Test with validation')
    parser.add_argument('--compare', action='store_true', help='Compare cached vs uncached')

    args = parser.parse_args()

    # If no args, run comparison
    if not any([args.with_cache, args.no_cache, args.validate, args.compare]):
        args.compare = True

    results = {}

    if args.with_cache or args.compare:
        results['cached'] = test_with_cache()

    if args.no_cache or args.compare:
        results['uncached'] = test_without_cache()

    if args.validate:
        results['validated'] = test_with_validation()

    # Print comparison
    if 'cached' in results and 'uncached' in results:
        if results['cached'] and results['uncached']:
            speedup = results['uncached'] / results['cached']
            print("\n" + "="*70)
            print("PERFORMANCE COMPARISON")
            print("="*70)
            print(f"Time WITH caching:    {results['cached']:.2f}s")
            print(f"Time WITHOUT caching: {results['uncached']:.2f}s")
            print(f"Speedup:              {speedup:.2f}x faster")
            print("="*70)


if __name__ == '__main__':
    main()
