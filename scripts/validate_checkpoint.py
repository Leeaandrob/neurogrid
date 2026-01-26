#!/usr/bin/env python3
"""
Validate NeuroGrid tensor checkpoints against PyTorch golden reference.

This script loads SafeTensors checkpoints created by NeuroGrid and compares
them against a PyTorch reference model to validate numerical correctness.

Usage:
    python scripts/validate_checkpoint.py ./checkpoints/req_* --tolerance 0.01
    python scripts/validate_checkpoint.py ./checkpoints/req_123 --layer 5 --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import load_file
except ImportError:
    print("Error: safetensors package required. Install with: pip install safetensors")
    sys.exit(1)


class CheckpointValidator:
    """Validates NeuroGrid tensor checkpoints."""

    def __init__(self, tolerance: float = 0.01, verbose: bool = False):
        self.tolerance = tolerance
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_checkpoint_dir(self, checkpoint_dir: Path) -> bool:
        """Validate all tensors in a checkpoint directory."""
        if not checkpoint_dir.exists():
            self.errors.append(f"Checkpoint directory not found: {checkpoint_dir}")
            return False

        # Find all safetensors files
        safetensor_files = list(checkpoint_dir.rglob("*.safetensors"))
        if not safetensor_files:
            self.errors.append(f"No SafeTensors files found in {checkpoint_dir}")
            return False

        all_valid = True
        for st_file in safetensor_files:
            if not self._validate_safetensors_file(st_file):
                all_valid = False

        return all_valid

    def _validate_safetensors_file(self, st_file: Path) -> bool:
        """Validate a single SafeTensors file."""
        if self.verbose:
            print(f"\nValidating: {st_file}")

        try:
            tensors = load_file(str(st_file))
        except Exception as e:
            self.errors.append(f"Failed to load {st_file}: {e}")
            return False

        all_valid = True
        for name, tensor in tensors.items():
            if not self._validate_tensor(st_file, name, tensor):
                all_valid = False

        return all_valid

    def _validate_tensor(self, st_file: Path, name: str, tensor: np.ndarray) -> bool:
        """Validate a single tensor for numerical issues."""
        issues = []

        # Check for NaN
        nan_count = np.isnan(tensor).sum()
        if nan_count > 0:
            issues.append(f"NaN values: {nan_count} ({100*nan_count/tensor.size:.2f}%)")

        # Check for Inf
        inf_count = np.isinf(tensor).sum()
        if inf_count > 0:
            issues.append(f"Inf values: {inf_count} ({100*inf_count/tensor.size:.2f}%)")

        # Check for extreme values
        if tensor.size > 0:
            valid_mask = np.isfinite(tensor)
            if valid_mask.any():
                valid_tensor = tensor[valid_mask]
                mean_val = np.mean(valid_tensor)
                std_val = np.std(valid_tensor)
                max_val = np.max(np.abs(valid_tensor))

                # FP16 overflow threshold
                if max_val > 65504:  # Max FP16 value
                    issues.append(f"Values exceed FP16 range: max={max_val}")

                if self.verbose:
                    print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
                          f"mean={mean_val:.4f}, std={std_val:.4f}, max_abs={max_val:.4f}")

        if issues:
            self.warnings.append(f"{st_file}/{name}: {', '.join(issues)}")
            return False

        return True

    def compare_with_reference(
        self, checkpoint_dir: Path, reference_dir: Path
    ) -> Tuple[bool, Dict[str, float]]:
        """Compare checkpoint tensors against reference tensors."""
        diffs = {}
        all_within_tolerance = True

        checkpoint_files = list(checkpoint_dir.rglob("*.safetensors"))
        for cp_file in checkpoint_files:
            # Find corresponding reference file
            rel_path = cp_file.relative_to(checkpoint_dir)
            ref_file = reference_dir / rel_path

            if not ref_file.exists():
                self.warnings.append(f"No reference found for {rel_path}")
                continue

            try:
                cp_tensors = load_file(str(cp_file))
                ref_tensors = load_file(str(ref_file))
            except Exception as e:
                self.errors.append(f"Failed to load for comparison: {e}")
                continue

            for name in cp_tensors:
                if name not in ref_tensors:
                    self.warnings.append(f"Tensor {name} not in reference")
                    continue

                cp_tensor = cp_tensors[name].astype(np.float32)
                ref_tensor = ref_tensors[name].astype(np.float32)

                if cp_tensor.shape != ref_tensor.shape:
                    self.errors.append(
                        f"Shape mismatch for {name}: {cp_tensor.shape} vs {ref_tensor.shape}"
                    )
                    continue

                # Compute max absolute difference
                max_diff = np.max(np.abs(cp_tensor - ref_tensor))
                rel_path_name = f"{rel_path}/{name}"
                diffs[rel_path_name] = max_diff

                if max_diff > self.tolerance:
                    all_within_tolerance = False
                    self.errors.append(
                        f"{rel_path_name}: max diff {max_diff:.6f} > tolerance {self.tolerance}"
                    )
                elif self.verbose:
                    print(f"  {rel_path_name}: max diff {max_diff:.6f} ✓")

        return all_within_tolerance, diffs

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All validations passed!")

        return len(self.errors) == 0


def list_checkpoints(base_dir: Path) -> List[Path]:
    """List all checkpoint directories."""
    if not base_dir.exists():
        return []

    checkpoints = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            checkpoints.append(entry)

    return sorted(checkpoints)


def main():
    parser = argparse.ArgumentParser(
        description="Validate NeuroGrid tensor checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ./checkpoints/req_001
  %(prog)s ./checkpoints/req_* --tolerance 0.001
  %(prog)s ./checkpoints/req_001 --reference ./golden --compare
  %(prog)s ./checkpoints --list
        """,
    )

    parser.add_argument(
        "checkpoint_dirs",
        nargs="*",
        type=Path,
        help="Checkpoint directories to validate",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed difference for comparison (default: 0.01)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="Reference directory for comparison",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against reference (requires --reference)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checkpoints",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Only validate specific layer",
    )

    args = parser.parse_args()

    # List mode
    if args.list:
        for cp_dir in args.checkpoint_dirs or [Path("./checkpoints")]:
            checkpoints = list_checkpoints(cp_dir)
            print(f"\nCheckpoints in {cp_dir}:")
            for cp in checkpoints:
                print(f"  - {cp.name}")
        return 0

    # Validate mode
    if not args.checkpoint_dirs:
        parser.error("No checkpoint directories specified")

    validator = CheckpointValidator(
        tolerance=args.tolerance,
        verbose=args.verbose,
    )

    all_valid = True
    for cp_dir in args.checkpoint_dirs:
        print(f"\n{'='*60}")
        print(f"Validating: {cp_dir}")
        print("=" * 60)

        if not validator.validate_checkpoint_dir(cp_dir):
            all_valid = False

        if args.compare and args.reference:
            print(f"\nComparing against reference: {args.reference}")
            within_tolerance, diffs = validator.compare_with_reference(
                cp_dir, args.reference
            )
            if not within_tolerance:
                all_valid = False

            if diffs:
                print("\nDifference summary:")
                for name, diff in sorted(diffs.items(), key=lambda x: -x[1]):
                    status = "❌" if diff > args.tolerance else "✓"
                    print(f"  {status} {name}: {diff:.6f}")

    # Print summary
    if not validator.print_summary():
        all_valid = False

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
