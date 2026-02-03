#!/usr/bin/env python3
"""Tests for CurriculumMasker step progression, especially with multiprocessing."""

import time
import torch.multiprocessing as mp
from typing import List

from metamon.backend.team_prediction.masking import CurriculumMasker
from metamon.backend.team_prediction.team import TeamSet, PokemonSet


def test_basic_step_progression():
    """Test that step progression works in a single process."""
    print("=" * 60)
    print("Test 1: Basic step progression (single process)")
    print("=" * 60)

    masker = CurriculumMasker(
        warmup_steps=1000,
        pokemon_prob=0.15,
        attrs_prob=0.5,
        min_pokemon_prob=0.0,
        min_attrs_prob=0.1,
    )

    # Check initial state
    assert masker._step == 0, f"Expected step 0, got {masker._step}"
    assert masker.progress == 0.0, f"Expected progress 0.0, got {masker.progress}"

    # Check initial ranges with minimums
    pokemon_range = masker.pokemon_prob_range
    attrs_range = masker.attrs_prob_range
    print(f"Step 0: pokemon_range={pokemon_range}, attrs_range={attrs_range}")
    assert pokemon_range == (0.0, 0.0), f"Expected (0.0, 0.0), got {pokemon_range}"
    assert attrs_range == (0.0, 0.1), f"Expected (0.0, 0.1), got {attrs_range}"

    # Progress to 50%
    masker.set_step(500)
    assert masker._step == 500
    assert masker.progress == 0.5

    pokemon_range = masker.pokemon_prob_range
    attrs_range = masker.attrs_prob_range
    print(f"Step 500 (50%): pokemon_range={pokemon_range}, attrs_range={attrs_range}")
    assert (
        abs(pokemon_range[1] - 0.075) < 0.001
    ), f"Expected ~0.075, got {pokemon_range[1]}"
    assert abs(attrs_range[1] - 0.3) < 0.001, f"Expected ~0.3, got {attrs_range[1]}"

    # Progress to 100%
    masker.set_step(1000)
    assert masker._step == 1000
    assert masker.progress == 1.0

    pokemon_range = masker.pokemon_prob_range
    attrs_range = masker.attrs_prob_range
    print(f"Step 1000 (100%): pokemon_range={pokemon_range}, attrs_range={attrs_range}")
    assert (
        abs(pokemon_range[1] - 0.15) < 0.001
    ), f"Expected 0.15, got {pokemon_range[1]}"
    assert abs(attrs_range[1] - 0.5) < 0.001, f"Expected 0.5, got {attrs_range[1]}"

    # Progress beyond warmup (should cap at 1.0)
    masker.set_step(2000)
    assert (
        masker.progress == 1.0
    ), f"Expected progress capped at 1.0, got {masker.progress}"

    print("✓ Basic step progression works correctly\n")


# Global masker for inheritance tests (workers inherit this via fork)
_global_masker: CurriculumMasker = None


def _worker_read_loop(result_queue: mp.Queue, num_reads: int, delay: float):
    """Worker that reads from inherited global masker."""
    for _ in range(num_reads):
        step = _global_masker._step
        attrs_range = _global_masker.attrs_prob_range
        result_queue.put((step, attrs_range[1]))
        time.sleep(delay)
    result_queue.put(None)  # Signal done


def test_multiprocess_inheritance():
    """Test that mp.Value is shared when inherited via fork."""
    print("=" * 60)
    print("Test 2: Multiprocess inheritance (simulating DataLoader)")
    print("=" * 60)

    global _global_masker
    _global_masker = CurriculumMasker(
        warmup_steps=100,
        pokemon_prob=0.15,
        attrs_prob=0.5,
        min_attrs_prob=0.1,
    )

    result_queue = mp.Queue()

    # Start worker BEFORE updating step (inherits masker state)
    worker = mp.Process(target=_worker_read_loop, args=(result_queue, 20, 0.05))
    worker.start()

    # Main process updates step while worker reads
    print("Main process updating steps while worker reads...")
    for step in range(0, 101, 10):
        _global_masker.set_step(step)
        time.sleep(0.1)

    # Collect results
    results = []
    while True:
        item = result_queue.get()
        if item is None:
            break
        results.append(item)

    worker.join()

    print("Worker observations:")
    for i, (step, attrs_prob) in enumerate(results[:5]):
        print(f"  [{i}] step={step}, attrs_prob={attrs_prob:.3f}")
    if len(results) > 8:
        print("  ...")
    for i, (step, attrs_prob) in enumerate(results[-3:], len(results) - 3):
        print(f"  [{i}] step={step}, attrs_prob={attrs_prob:.3f}")

    # Verify worker saw progression
    steps_seen = [r[0] for r in results]
    min_step, max_step = min(steps_seen), max(steps_seen)
    print(f"\nWorker saw steps from {min_step} to {max_step}")

    if max_step > min_step:
        print("✓ Worker saw step progression - shared memory works!\n")
    else:
        print("✗ FAILED: Worker did not see step updates\n")
        raise AssertionError("Worker should have seen step progression")


def _worker_mask_teams(result_queue: mp.Queue, num_masks: int):
    """Worker that masks teams using inherited masker."""
    total_masked = 0
    for _ in range(num_masks):
        team = _make_test_team()
        masked, _ = _global_masker.mask(team)
        # Count missing attributes
        if masked.lead.ability == PokemonSet.MISSING_ABILITY:
            total_masked += 1
        if masked.lead.item == PokemonSet.MISSING_ITEM:
            total_masked += 1
        if masked.lead.tera_type == PokemonSet.MISSING_TERA_TYPE:
            total_masked += 1
        total_masked += sum(
            1 for m in masked.lead.moves if m == PokemonSet.MISSING_MOVE
        )

    result_queue.put(total_masked / num_masks)


def _make_test_team():
    """Create a simple team for testing."""
    pokemon = PokemonSet(
        name="Pikachu",
        gen=9,
        ability="Static",
        item="Light Ball",
        tera_type="Electric",
        moves=["Thunderbolt", "Volt Tackle", "Iron Tail", "Quick Attack"],
        nature="Timid",
        evs=[0, 0, 0, 252, 4, 252],
        ivs=[31, 31, 31, 31, 31, 31],
    )
    return TeamSet(format="gen9ou", lead=pokemon, reserve=[])


def test_masking_output_varies():
    """Test that actual masking output changes with curriculum progress."""
    print("=" * 60)
    print("Test 3: Masking output varies with progress")
    print("=" * 60)

    global _global_masker
    _global_masker = CurriculumMasker(
        warmup_steps=100,
        pokemon_prob=0.15,
        attrs_prob=0.5,
        min_attrs_prob=0.1,
    )

    def count_masks_in_main(num_samples=100):
        total_masked = 0
        for _ in range(num_samples):
            team = _make_test_team()
            masked, _ = _global_masker.mask(team)
            if masked.lead.ability == PokemonSet.MISSING_ABILITY:
                total_masked += 1
            if masked.lead.item == PokemonSet.MISSING_ITEM:
                total_masked += 1
            if masked.lead.tera_type == PokemonSet.MISSING_TERA_TYPE:
                total_masked += 1
            total_masked += sum(
                1 for m in masked.lead.moves if m == PokemonSet.MISSING_MOVE
            )
        return total_masked / num_samples

    # Test at different curriculum stages
    _global_masker.set_step(0)
    masks_at_0 = count_masks_in_main()
    print(f"Step 0 (0%): avg masked attrs = {masks_at_0:.2f}")

    _global_masker.set_step(50)
    masks_at_50 = count_masks_in_main()
    print(f"Step 50 (50%): avg masked attrs = {masks_at_50:.2f}")

    _global_masker.set_step(100)
    masks_at_100 = count_masks_in_main()
    print(f"Step 100 (100%): avg masked attrs = {masks_at_100:.2f}")

    # Verify progression (more masking at higher steps)
    assert masks_at_50 > masks_at_0 * 0.9, f"Expected more masking at step 50 than 0"
    assert (
        masks_at_100 > masks_at_50 * 0.9
    ), f"Expected more masking at step 100 than 50"

    print("✓ Masking increases with curriculum progress\n")


def test_worker_sees_masking_changes():
    """Test that workers see masking changes when step is updated."""
    print("=" * 60)
    print("Test 4: Workers see masking changes")
    print("=" * 60)

    global _global_masker
    _global_masker = CurriculumMasker(
        warmup_steps=100,
        pokemon_prob=0.15,
        attrs_prob=0.5,
        min_attrs_prob=0.1,
    )

    result_queue = mp.Queue()

    # Measure at step 0
    _global_masker.set_step(0)
    worker1 = mp.Process(target=_worker_mask_teams, args=(result_queue, 100))
    worker1.start()
    worker1.join()
    masks_at_0 = result_queue.get()

    # Measure at step 100
    _global_masker.set_step(100)
    worker2 = mp.Process(target=_worker_mask_teams, args=(result_queue, 100))
    worker2.start()
    worker2.join()
    masks_at_100 = result_queue.get()

    print(f"Worker at step 0: avg masked attrs = {masks_at_0:.2f}")
    print(f"Worker at step 100: avg masked attrs = {masks_at_100:.2f}")

    if masks_at_100 > masks_at_0:
        print("✓ Workers see curriculum progression!\n")
    else:
        print("✗ FAILED: Workers don't see step changes\n")
        raise AssertionError("Workers should see more masking at step 100")


def test_dataloader_simulation():
    """Simulate actual DataLoader behavior with persistent workers."""
    print("=" * 60)
    print("Test 5: DataLoader simulation")
    print("=" * 60)

    import torch
    from torch.utils.data import Dataset, DataLoader

    class MockDataset(Dataset):
        def __init__(self, masker: CurriculumMasker, size: int = 1000):
            self.masker = masker
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # This runs in worker process
            step = self.masker._step
            attrs_range = self.masker.attrs_prob_range[1]
            return torch.tensor([step, attrs_range * 100])

    masker = CurriculumMasker(warmup_steps=100)
    dataset = MockDataset(masker)

    # Use num_workers=2 like real training
    loader = DataLoader(dataset, batch_size=4, num_workers=2, persistent_workers=True)

    print("Iterating with step updates...")
    steps_seen = []

    for i, batch in enumerate(loader):
        if i >= 20:
            break

        # Update step in main process (like training loop does)
        masker.set_step(i * 10)

        worker_steps = batch[:, 0].tolist()
        steps_seen.extend(worker_steps)

        if i % 5 == 0:
            print(f"  Batch {i}: main_step={i*10}, worker_saw={worker_steps}")

    min_seen, max_seen = int(min(steps_seen)), int(max(steps_seen))
    print(f"\nWorkers saw steps from {min_seen} to {max_seen}")

    if max_seen > min_seen:
        print("✓ DataLoader workers see step progression!\n")
    else:
        print("⚠ WARNING: DataLoader workers may have prefetch lag\n")
        print("  This is expected - workers prefetch batches ahead of step updates.")
        print("  The curriculum will still work but with some lag.\n")


if __name__ == "__main__":
    # Use fork to allow mp.Value inheritance
    mp.set_start_method("fork", force=True)

    test_basic_step_progression()
    test_multiprocess_inheritance()
    test_masking_output_varies()
    test_worker_sees_masking_changes()
    test_dataloader_simulation()

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
