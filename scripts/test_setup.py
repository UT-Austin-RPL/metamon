#!/usr/bin/env python3
"""
Quick test script to verify the self-play pipeline setup.

Tests:
1. Model registration
2. Checkpoint loading
3. Basic evaluation capability

Usage:
    python scripts/test_setup.py
"""

import os
import sys
from pathlib import Path

# Add metamon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing self-play pipeline setup...\n")

# Test 1: Import modules
print("[1/4] Testing imports...")
try:
    from metamon.rl.gen1_binary_models import *
    from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
    from metamon.rl.evaluate import pretrained_vs_baselines
    from metamon.env import get_metamon_teams
    print("✓ All imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: List registered models
print("[2/4] Checking registered models...")
try:
    all_models = get_pretrained_model_names()
    gen1_models = [m for m in all_models if "Gen1Binary" in m]

    print(f"Total registered models: {len(all_models)}")
    print(f"Gen1Binary models: {len(gen1_models)}")

    if gen1_models:
        print("Available Gen1Binary checkpoints:")
        for model in sorted(gen1_models):
            print(f"  - {model}")
    else:
        print("⚠ No Gen1Binary models found (training may not be complete yet)")

    print()
except Exception as e:
    print(f"✗ Model registration check failed: {e}\n")
    sys.exit(1)

# Test 3: Check checkpoint files
print("[3/4] Checking checkpoint files...")
try:
    checkpoint_dir = os.path.expanduser("~/metamon/models/metamon_checkpoints/Gen1BinaryRewardV0/ckpts/policy_weights")

    if os.path.exists(checkpoint_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("policy_epoch_")])
        print(f"Found {len(checkpoints)} checkpoint files:")
        for ckpt in checkpoints:
            size_mb = os.path.getsize(os.path.join(checkpoint_dir, ckpt)) / (1024 * 1024)
            print(f"  - {ckpt} ({size_mb:.0f} MB)")
    else:
        print(f"⚠ Checkpoint directory not found: {checkpoint_dir}")
        print("  Training may still be in progress")

    print()
except Exception as e:
    print(f"✗ Checkpoint check failed: {e}\n")
    sys.exit(1)

# Test 4: Try loading a model (if checkpoints exist)
print("[4/4] Testing model loading...")
try:
    if gen1_models and os.path.exists(checkpoint_dir) and checkpoints:
        test_model_name = gen1_models[0]
        print(f"Attempting to load: {test_model_name}")

        model = get_pretrained_model(test_model_name)
        print(f"✓ Model loaded successfully")
        print(f"  - Observation space: {model.observation_space.__class__.__name__}")
        print(f"  - Action space: {model.action_space.__class__.__name__}")
        print(f"  - Reward function: {model.reward_function.__class__.__name__}")
        print(f"  - Default checkpoint: {model.default_checkpoint}")
    else:
        print("⚠ Skipping model load test (no checkpoints available yet)")

    print()
except Exception as e:
    print(f"✗ Model loading failed: {e}\n")
    print("This may indicate an issue with checkpoint paths or model configuration")
    sys.exit(1)

# Summary
print("="*80)
print("SETUP TEST COMPLETE")
print("="*80)
print()

if gen1_models and os.path.exists(checkpoint_dir) and checkpoints:
    print("✓ All systems operational!")
    print()
    print("Next steps:")
    print("  1. Wait for training to complete (or use existing checkpoints)")
    print("  2. Run tournament: python scripts/self_play_tournament.py --help")
    print("  3. Calculate ELO: python scripts/calculate_elo.py --help")
    print("  4. Visualize: python scripts/visualize_results.py --help")
else:
    print("⚠ Setup partially complete")
    print()
    print("Action items:")
    if not gen1_models:
        print("  - Gen1Binary models not registered yet")
    if not os.path.exists(checkpoint_dir):
        print("  - Wait for training to create checkpoints")
    if not checkpoints:
        print("  - No checkpoint files found (training in progress?)")

print()
sys.exit(0)
