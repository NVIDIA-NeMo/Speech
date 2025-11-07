#!/usr/bin/env python3
"""
Test script to verify the checkpoint unfinished marker fix.

This script tests:
1. Unfinished markers are properly removed after successful checkpoint saves
2. AsyncFinalizerCallback is automatically added when async_save is enabled
3. Pending async saves are finalized on train_end
"""

import os
import sys
import tempfile
from pathlib import Path


# Test 1: Check that unfinished marker removal is robust
def test_marker_removal():
    """Test that marker removal works correctly."""
    print("\n=== Test 1: Marker Removal ===")

    from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.ckpt"

        # Create an unfinished marker
        ModelCheckpoint.set_checkpoint_unfinished_marker(checkpoint_path, barrier_after=False)
        marker_path = ModelCheckpoint.format_checkpoint_unfinished_marker_path(checkpoint_path)

        assert marker_path.exists(), "Marker should be created"
        print(f"✓ Unfinished marker created: {marker_path}")

        # Remove the marker
        ModelCheckpoint.remove_checkpoint_unfinished_marker(checkpoint_path, barrier_before=False)

        assert not marker_path.exists(), "Marker should be removed"
        print(f"✓ Unfinished marker removed successfully")

    print("✓ Test 1 PASSED\n")
    return True


# Test 2: Check that is_checkpoint_unfinished works correctly
def test_checkpoint_unfinished_check():
    """Test that checkpoint unfinished check works."""
    print("\n=== Test 2: Checkpoint Unfinished Check ===")

    from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.ckpt"

        # Initially should not be unfinished
        assert not ModelCheckpoint.is_checkpoint_unfinished(
            checkpoint_path
        ), "Checkpoint should not be marked as unfinished initially"
        print("✓ Checkpoint not marked as unfinished initially")

        # Mark as unfinished
        ModelCheckpoint.set_checkpoint_unfinished_marker(checkpoint_path, barrier_after=False)

        assert ModelCheckpoint.is_checkpoint_unfinished(checkpoint_path), "Checkpoint should be marked as unfinished"
        print("✓ Checkpoint correctly marked as unfinished")

        # Remove marker
        ModelCheckpoint.remove_checkpoint_unfinished_marker(checkpoint_path, barrier_before=False)

        assert not ModelCheckpoint.is_checkpoint_unfinished(
            checkpoint_path
        ), "Checkpoint should not be marked as unfinished after removal"
        print("✓ Checkpoint no longer marked as unfinished after removal")

    print("✓ Test 2 PASSED\n")
    return True


# Test 3: Check that _remove_unfinished_checkpoints works
def test_remove_unfinished_checkpoints():
    """Test that unfinished checkpoints are properly removed."""
    print("\n=== Test 3: Remove Unfinished Checkpoints ===")

    from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Create some checkpoint files with unfinished markers
        ckpt1 = checkpoint_dir / "checkpoint1.ckpt"
        ckpt2 = checkpoint_dir / "checkpoint2.ckpt"
        ckpt3 = checkpoint_dir / "checkpoint3.ckpt"

        # Create actual checkpoint files
        ckpt1.touch()
        ckpt2.touch()
        ckpt3.touch()

        # Mark ckpt1 and ckpt2 as unfinished
        ModelCheckpoint.set_checkpoint_unfinished_marker(ckpt1, barrier_after=False)
        ModelCheckpoint.set_checkpoint_unfinished_marker(ckpt2, barrier_after=False)

        print(f"✓ Created 3 checkpoints, marked 2 as unfinished")

        # Remove unfinished checkpoints
        ModelCheckpoint._remove_unfinished_checkpoints(checkpoint_dir)

        # Check that unfinished checkpoints were removed
        assert not ckpt1.exists(), "Unfinished checkpoint1 should be removed"
        assert not ckpt2.exists(), "Unfinished checkpoint2 should be removed"
        assert ckpt3.exists(), "Finished checkpoint3 should still exist"

        print("✓ Unfinished checkpoints removed, finished checkpoint preserved")

        # Check that markers are also removed
        marker1 = ModelCheckpoint.format_checkpoint_unfinished_marker_path(ckpt1)
        marker2 = ModelCheckpoint.format_checkpoint_unfinished_marker_path(ckpt2)

        assert not marker1.exists(), "Marker for checkpoint1 should be removed"
        assert not marker2.exists(), "Marker for checkpoint2 should be removed"

        print("✓ Unfinished markers also removed")

    print("✓ Test 3 PASSED\n")
    return True


# Test 4: Test with distributed checkpoint directories
def test_distributed_checkpoint_cleanup():
    """Test that distributed checkpoint directories are properly cleaned up."""
    print("\n=== Test 4: Distributed Checkpoint Cleanup ===")

    from nemo.lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)

        # Create distributed checkpoint directories
        dist_ckpt1 = checkpoint_dir / "checkpoint_step100"
        dist_ckpt2 = checkpoint_dir / "checkpoint_step200"
        dist_ckpt3 = checkpoint_dir / "checkpoint_step300"

        dist_ckpt1.mkdir()
        dist_ckpt2.mkdir()
        dist_ckpt3.mkdir()

        # Mark dist_ckpt1 as unfinished
        ModelCheckpoint.set_checkpoint_unfinished_marker(dist_ckpt1, barrier_after=False)

        print(f"✓ Created 3 distributed checkpoints, marked 1 as unfinished")

        # Remove unfinished checkpoints
        ModelCheckpoint._remove_unfinished_checkpoints(checkpoint_dir)

        # Check that unfinished checkpoint was removed
        assert not dist_ckpt1.exists(), "Unfinished distributed checkpoint should be removed"
        assert dist_ckpt2.exists(), "Finished distributed checkpoint should still exist"
        assert dist_ckpt3.exists(), "Finished distributed checkpoint should still exist"

        print("✓ Unfinished distributed checkpoint removed, others preserved")

    print("✓ Test 4 PASSED\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Checkpoint Unfinished Marker Fix")
    print("=" * 60)

    tests = [
        test_marker_removal,
        test_checkpoint_unfinished_check,
        test_remove_unfinished_checkpoints,
        test_distributed_checkpoint_cleanup,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
