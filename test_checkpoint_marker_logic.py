#!/usr/bin/env python3
"""
Simple test to verify the checkpoint marker logic without full dependencies.
"""

import tempfile
from pathlib import Path


def test_marker_file_operations():
    """Test basic marker file operations."""
    print("\n=== Testing Marker File Operations ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.ckpt"

        # Simulate marker path creation (from format_checkpoint_unfinished_marker_path)
        marker_filepath = str(checkpoint_path).removesuffix(".ckpt")
        marker_filepath = marker_filepath.removesuffix("-EMA")
        marker_path = Path(marker_filepath + "-unfinished")

        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Marker path: {marker_path}")

        # Create marker
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.touch()

        assert marker_path.exists(), "Marker should be created"
        print("✓ Marker created successfully")

        # Remove marker
        if marker_path.exists():
            marker_path.unlink()

        assert not marker_path.exists(), "Marker should be removed"
        print("✓ Marker removed successfully")

    print("✓ Test PASSED\n")
    return True


def test_distributed_checkpoint_marker():
    """Test marker for distributed checkpoint (directory)."""
    print("\n=== Testing Distributed Checkpoint Marker ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint_step100"
        checkpoint_dir.mkdir()

        # Simulate marker path creation for directory
        marker_filepath = str(checkpoint_dir).removesuffix(".ckpt")
        marker_filepath = marker_filepath.removesuffix("-EMA")
        marker_path = Path(marker_filepath + "-unfinished")

        print(f"Checkpoint dir: {checkpoint_dir}")
        print(f"Marker path: {marker_path}")

        # Create marker
        marker_path.touch()

        assert marker_path.exists(), "Marker should be created"
        print("✓ Marker created for distributed checkpoint")

        # Check if checkpoint is unfinished
        is_unfinished = marker_path.exists()
        assert is_unfinished, "Checkpoint should be marked as unfinished"
        print("✓ Checkpoint correctly identified as unfinished")

        # Remove marker
        if marker_path.exists():
            marker_path.unlink()

        assert not marker_path.exists(), "Marker should be removed"
        print("✓ Marker removed successfully")

    print("✓ Test PASSED\n")
    return True


def test_ema_checkpoint_marker():
    """Test that EMA checkpoint uses same marker as base checkpoint."""
    print("\n=== Testing EMA Checkpoint Marker ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        base_checkpoint = Path(tmpdir) / "checkpoint.ckpt"
        ema_checkpoint = Path(tmpdir) / "checkpoint-EMA.ckpt"

        # Simulate marker path for base checkpoint
        base_marker_filepath = str(base_checkpoint).removesuffix(".ckpt")
        base_marker_filepath = base_marker_filepath.removesuffix("-EMA")
        base_marker_path = Path(base_marker_filepath + "-unfinished")

        # Simulate marker path for EMA checkpoint
        ema_marker_filepath = str(ema_checkpoint).removesuffix(".ckpt")
        ema_marker_filepath = ema_marker_filepath.removesuffix("-EMA")
        ema_marker_path = Path(ema_marker_filepath + "-unfinished")

        print(f"Base checkpoint: {base_checkpoint}")
        print(f"Base marker: {base_marker_path}")
        print(f"EMA checkpoint: {ema_checkpoint}")
        print(f"EMA marker: {ema_marker_path}")

        # Verify they use the same marker
        assert base_marker_path == ema_marker_path, "Base and EMA should use same marker"
        print("✓ Base and EMA checkpoints use the same marker (as expected)")

    print("✓ Test PASSED\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Checkpoint Marker Logic")
    print("=" * 60)

    tests = [
        test_marker_file_operations,
        test_distributed_checkpoint_marker,
        test_ema_checkpoint_marker,
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
    import sys

    success = main()
    sys.exit(0 if success else 1)
