#!/usr/bin/env python3
"""Benchmark script comparing numox.Matrix performance against NumPy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
from importlib import metadata
import json
import os
import platform
import statistics
import sys
import time
from types import ModuleType
from typing import Any, Callable, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required to run this benchmark. Install it via: pip install numpy", file=sys.stderr)
    sys.exit(1)

nx: Any = None


def _numox_version(mod: ModuleType) -> str:
    ver = getattr(mod, "__version__", None)
    if ver:
        return str(ver)
    try:
        return metadata.version("numox")
    except metadata.PackageNotFoundError:
        return "local"


def _has_inplace_core(src_dir: str) -> bool:
    pkg_dir = os.path.join(src_dir, "numox")
    if not os.path.isdir(pkg_dir):
        return False
    return any(
        name.startswith("_core") and name.endswith((".pyd", ".so"))
        for name in os.listdir(pkg_dir)
    )


def _load_numox(use_installed: bool) -> ModuleType:
    saved_path = list(sys.path)
    repo_dir = os.path.abspath(os.path.dirname(__file__))
    src_dir = os.path.join(repo_dir, "src")
    try:
        if use_installed:
            local_dirs = {
                os.path.abspath("."),
                repo_dir,
                src_dir,
            }
            sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in local_dirs]
        elif _has_inplace_core(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)
        import numox as mod

        return mod
    except ImportError as exc:
        print(f"Error: failed to import numox ({exc}).", file=sys.stderr)
        if not use_installed:
            print("  If a local build in this directory shadows site-packages, retry with --installed", file=sys.stderr)
        print("  Or install/build it via: pip install numox   OR   python setup.py build_ext --inplace", file=sys.stderr)
        sys.exit(1)
    finally:
        sys.path = saved_path


@dataclass
class BenchResult:
    size: int
    op: str
    iters: int
    repeats: int
    numpy_median_ms: float
    numpy_min_ms: float
    numox_median_ms: float
    numox_min_ms: float
    speedup_median: float
    metric_name: str
    numpy_metric: Optional[float]
    numox_metric: Optional[float]
    verified: Optional[bool]


def _make_pair(n: int, seed: int, scale: float = 1.0) -> Tuple[np.ndarray, "nx.Matrix"]:
    rng = np.random.default_rng(seed)
    arr = rng.uniform(-scale, scale, size=(n, n))
    mat = nx.Matrix(n, n, 0.0)
    np.copyto(np.asarray(mat).reshape(n, n), arr)
    return arr, mat


def _calibrate_iters(fn: Callable[[], object], target_sec: float, max_iters: int = 2000) -> int:
    t0 = time.perf_counter()
    _ = fn()
    dt = time.perf_counter() - t0
    if dt <= 0.0:
        return max_iters
    est = int(target_sec / dt)
    return max(1, min(max_iters, est))


def _measure(
    fn: Callable[[], object],
    iters: int,
    repeats: int,
    warmup: int,
) -> List[float]:
    for _ in range(warmup):
        _ = fn()

    samples_ms: List[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn()
        t1 = time.perf_counter()
        del out
        samples_ms.append(((t1 - t0) / iters) * 1000.0)
    return samples_ms


def _to_numpy(obj: object) -> np.ndarray:
    return np.asarray(obj)


def _format_metric(val: Optional[float], unit: str) -> str:
    if val is None or not unit:
        return "-"
    return f"{val:7.2f} {unit}"


def run_suite(
    sizes: List[int],
    selected_ops: List[str],
    repeats: int,
    warmup: int,
    target_sec: float,
    fixed_iters: Optional[int],
    verify: bool,
) -> List[BenchResult]:
    results: List[BenchResult] = []
    rng_np = np.random.default_rng(42)

    for n in sizes:
        np_a, nx_a = _make_pair(n, seed=101, scale=1.0)
        np_b, nx_b = _make_pair(n, seed=202, scale=1.0)
        # Scaled matrix for power so A^10 stays well-conditioned in float64
        pow_scale = 1.0 / max(1.0, float(n) ** 0.5)
        np_p, nx_p = _make_pair(n, seed=303, scale=pow_scale)

        elem_bytes = float(n) * float(n) * 8.0
        matmul_flops = 2.0 * float(n) * float(n) * float(n)

        specs: List[
            Tuple[str, Callable[[], object], Callable[[], object], str, Optional[float]]
        ] = [
            ("fill", lambda: np.full((n, n), 1.5, dtype=np.float64), lambda: nx.Matrix(n, n, 1.5), "GB/s", elem_bytes),
            ("rand", lambda: rng_np.uniform(0.0, 1.0, size=(n, n)), lambda: nx.Matrix(n, n, rand=True, low=0.0, high=1.0, seed=42), "GB/s", elem_bytes),
            ("add", lambda: np_a + np_b, lambda: nx_a + nx_b, "GB/s", 3.0 * elem_bytes),
            ("sub", lambda: np_a - np_b, lambda: nx_a - nx_b, "GB/s", 3.0 * elem_bytes),
            ("neg", lambda: -np_a, lambda: -nx_a, "GB/s", 2.0 * elem_bytes),
            ("abs", lambda: np.abs(np_a), lambda: abs(nx_a), "GB/s", 2.0 * elem_bytes),
            ("matmul", lambda: np_a @ np_b, lambda: nx_a @ nx_b, "GFLOPS", matmul_flops),
            ("pow10", lambda: np.linalg.matrix_power(np_p, 10), lambda: nx_p ** 10, "ms", None),
        ]

        if n >= 4:
            np_va = np_a[1 : n - 1, 1 : n - 1]
            np_vb = np_b[1 : n - 1, 1 : n - 1]
            nx_va = nx_a[1 : n - 1, 1 : n - 1]
            nx_vb = nx_b[1 : n - 1, 1 : n - 1]
            view_bytes = 3.0 * float(n - 2) * float(n - 2) * 8.0
            specs.append(
                ("slice_add", lambda: np_va + np_vb, lambda: nx_va + nx_vb, "GB/s", view_bytes)
            )

        for op_name, np_fn, nx_fn, metric_unit, work_amount in specs:
            if "all" not in selected_ops and op_name not in selected_ops:
                continue

            verified: Optional[bool] = None
            if verify and op_name != "rand":
                ref = _to_numpy(np_fn())
                got = _to_numpy(nx_fn()).reshape(ref.shape)
                verified = bool(np.allclose(got, ref, rtol=1e-6, atol=1e-8))
                if not verified:
                    max_diff = float(np.max(np.abs(got - ref)))
                    print(
                        f"WARNING: verification mismatch for {op_name} ({n}x{n}), max diff = {max_diff:.3e}",
                        file=sys.stderr,
                    )

            if fixed_iters is not None:
                iters = fixed_iters
            else:
                iters_np = _calibrate_iters(np_fn, target_sec)
                iters_nx = _calibrate_iters(nx_fn, target_sec)
                iters = max(1, min(iters_np, iters_nx))

            np_samples = _measure(np_fn, iters=iters, repeats=repeats, warmup=warmup)
            nx_samples = _measure(nx_fn, iters=iters, repeats=repeats, warmup=warmup)

            np_med = statistics.median(np_samples)
            np_min = min(np_samples)
            nx_med = statistics.median(nx_samples)
            nx_min = min(nx_samples)
            speedup = np_med / nx_med if nx_med > 0.0 else float("inf")

            np_metric: Optional[float] = None
            nx_metric: Optional[float] = None
            if work_amount is not None:
                np_metric = work_amount / (np_med * 1e6)
                nx_metric = work_amount / (nx_med * 1e6)

            results.append(
                BenchResult(
                    size=n,
                    op=op_name,
                    iters=iters,
                    repeats=repeats,
                    numpy_median_ms=np_med,
                    numpy_min_ms=np_min,
                    numox_median_ms=nx_med,
                    numox_min_ms=nx_min,
                    speedup_median=speedup,
                    metric_name=metric_unit if work_amount is not None else "",
                    numpy_metric=np_metric,
                    numox_metric=nx_metric,
                    verified=verified,
                )
            )
    return results


def print_report(results: List[BenchResult], nx_mod: ModuleType) -> None:
    print("=" * 98)
    print("numox vs NumPy Performance Benchmark")
    print(
        f"Python {platform.python_version()} | NumPy {np.__version__} | "
        f"numox {_numox_version(nx_mod)} | "
        f"OS {platform.system()} {platform.machine()} | "
        f"CPU cores: {os.cpu_count()} | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'auto')}"
    )
    print(f"numox module: {getattr(nx_mod, '__file__', '<unknown>')}")
    print("=" * 98)

    header = (
        f"{'Size':>9}  {'Op':<10}  {'Iters':>5}  "
        f"{'NumPy med(ms)':>13}  {'numox med(ms)':>13}  {'Speedup':>8}  "
        f"{'NumPy rate':>14}  {'numox rate':>14}  {'OK':>3}"
    )
    current_size: Optional[int] = None
    for r in results:
        if r.size != current_size:
            current_size = r.size
            print(f"\n--- {r.size} x {r.size} (float64) ---")
            print(header)
            print("-" * 98)

        ok_str = "-" if r.verified is None else ("yes" if r.verified else "NO")
        np_rate = _format_metric(r.numpy_metric, r.metric_name)
        nx_rate = _format_metric(r.numox_metric, r.metric_name)
        print(
            f"{f'{r.size}x{r.size}':>9}  {r.op:<10}  {r.iters:>5}  "
            f"{r.numpy_median_ms:>13.4f}  {r.numox_median_ms:>13.4f}  {r.speedup_median:>7.2f}x  "
            f"{np_rate:>14}  {nx_rate:>14}  {ok_str:>3}"
        )

    print()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark numox vs NumPy performance.")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[128, 512, 1024],
        help="Square matrix sizes N to benchmark (default: 128 512 1024).",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=["all"],
        choices=["all", "fill", "rand", "add", "sub", "neg", "abs", "matmul", "pow10", "slice_add"],
        help="Operations to benchmark (default: all).",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Number of timed rounds per test (default: 5).")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before timing (default: 2).")
    parser.add_argument("--target-time", type=float, default=0.15, help="Target seconds per round when auto-calibrating iters (default: 0.15).")
    parser.add_argument("--iters", type=int, default=None, help="Fixed inner loop iterations (overrides --target-time).")
    parser.add_argument("--no-verify", action="store_true", help="Skip numerical equivalence check against NumPy.")
    parser.add_argument("--installed", action="store_true", help="Ignore current directory and import pip-installed numox from site-packages.")
    parser.add_argument("--json", type=str, default=None, help="Optional path to write JSON benchmark results.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    global nx
    args = parse_args(argv)
    if any(s <= 0 for s in args.sizes):
        print("Error: all sizes must be positive integers.", file=sys.stderr)
        return 2
    if args.repeats <= 0 or args.warmup < 0 or args.target_time <= 0:
        print("Error: repeats and target-time must be > 0, warmup >= 0.", file=sys.stderr)
        return 2
    if args.iters is not None and args.iters <= 0:
        print("Error: iters must be positive.", file=sys.stderr)
        return 2

    nx = _load_numox(args.installed)

    results = run_suite(
        sizes=args.sizes,
        selected_ops=args.ops,
        repeats=args.repeats,
        warmup=args.warmup,
        target_sec=args.target_time,
        fixed_iters=args.iters,
        verify=not args.no_verify,
    )
    print_report(results, nx)

    if args.json:
        payload = {
            "env": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "numox_version": _numox_version(nx),
                "numox_file": getattr(nx, "__file__", None),
                "system": platform.system(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            },
            "results": [asdict(r) for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"Saved JSON results to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
