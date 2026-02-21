#!/usr/bin/env python3
"""
Sweep FAST settings to find configs where custom FAST beats FAST+.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from step3_fast_tokenization import load_trajectories, run_comparison


def _parse_list(value: str, cast_fn):
    return [cast_fn(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep FAST tokenizer settings.")
    parser.add_argument("--chunk-seconds", default="1.0,2.0", help="Comma-separated list, e.g. 1.0,2.0")
    parser.add_argument("--strides", default="10,5,2,1", help="Comma-separated list of strides in steps")
    parser.add_argument("--vocab-sizes", default="1024,2048,4096", help="Comma-separated list, e.g. 1024,2048,4096")
    parser.add_argument("--scales", default="8.0,10.0,12.0", help="Comma-separated list, e.g. 8.0,10.0,12.0")
    parser.add_argument("--output", default="results/fast_sweep.csv", help="CSV output path")
    parser.add_argument("--top-k", type=int, default=5, help="How many top configs to print")
    parser.add_argument("--reuse", action="store_true", help="Reuse saved custom tokenizers if present")
    args = parser.parse_args()

    chunk_seconds_list = _parse_list(args.chunk_seconds, float)
    strides_list = _parse_list(args.strides, int)
    vocab_sizes = _parse_list(args.vocab_sizes, int)
    scales = _parse_list(args.scales, float)

    trajectories = load_trajectories()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    best = None
    total_runs = (
        len(chunk_seconds_list)
        * len(strides_list)
        * len(vocab_sizes)
        * len(scales)
    )
    print(f"Planned sweep runs: {total_runs}")

    for chunk_seconds in chunk_seconds_list:
        chunk_steps = int(round(10 * chunk_seconds))
        if chunk_steps <= 0:
            continue
        for stride in strides_list:
            if stride <= 0:
                continue
            for vocab_size in vocab_sizes:
                for scale in scales:
                    save_dir = (
                        Path("results")
                        / "custom_fast_sweep"
                        / f"cs{chunk_seconds}_s{stride}_v{vocab_size}_k{scale}"
                    )
                    print()
                    print("=" * 70)
                    print(f"Sweep config: chunk_seconds={chunk_seconds} stride={stride} vocab={vocab_size} scale={scale}")
                    print("=" * 70)

                    results = run_comparison(
                        trajectories,
                        chunk_steps=chunk_steps,
                        stride=stride,
                        vocab_size=vocab_size,
                        scale=scale,
                        custom_save_dir=save_dir,
                        force_retrain=not args.reuse,
                    )

                    avg_naive = float(np.mean(results["naive_tokens"]))
                    avg_fast = float(np.mean(results["fast_tokens"]))
                    avg_custom = float(np.mean(results["custom_tokens"]))
                    compression_fast = avg_naive / avg_fast
                    compression_custom = avg_naive / avg_custom
                    generalization_ratio = (
                        compression_fast / compression_custom if compression_custom > 0 else 0.0
                    )

                    rows.append(
                        {
                            "chunk_seconds": chunk_seconds,
                            "chunk_steps": chunk_steps,
                            "stride": stride,
                            "vocab_size": vocab_size,
                            "scale": scale,
                            "total_chunks": results.get("total_chunks", 0),
                            "avg_naive_tokens": avg_naive,
                            "avg_fast_tokens": avg_fast,
                            "avg_custom_tokens": avg_custom,
                            "compression_fast": compression_fast,
                            "compression_custom": compression_custom,
                            "generalization_ratio": generalization_ratio,
                        }
                    )

                    if best is None or compression_custom > best["compression_custom"]:
                        best = rows[-1]

    # Write CSV
    if rows:
        header = list(rows[0].keys())
        with out_path.open("w") as f:
            f.write(",".join(header) + "\n")
            for row in rows:
                f.write(",".join(str(row[h]) for h in header) + "\n")

    print()
    print("=" * 70)
    print(f"Saved sweep results to {out_path}")
    if best:
        print("Best custom FAST compression:")
        print(best)

    if rows:
        # Top configs by custom compression
        top_custom = sorted(rows, key=lambda r: r["compression_custom"], reverse=True)[: args.top_k]
        print()
        print("Top configs by custom FAST compression:")
        for row in top_custom:
            print(
                f"  cs={row['chunk_seconds']} stride={row['stride']} vocab={row['vocab_size']} scale={row['scale']} "
                f"custom={row['compression_custom']:.2f}x fast+={row['compression_fast']:.2f}x "
                f"gen_ratio={row['generalization_ratio']:.2f}"
            )

        # Top configs where custom beats FAST+ (gen_ratio < 1.0)
        custom_better = [r for r in rows if r["generalization_ratio"] < 1.0]
        print()
        if custom_better:
            top_better = sorted(custom_better, key=lambda r: r["generalization_ratio"])[: args.top_k]
            print("Top configs where custom FAST beats FAST+:")
            for row in top_better:
                print(
                    f"  cs={row['chunk_seconds']} stride={row['stride']} vocab={row['vocab_size']} scale={row['scale']} "
                    f"custom={row['compression_custom']:.2f}x fast+={row['compression_fast']:.2f}x "
                    f"gen_ratio={row['generalization_ratio']:.2f}"
                )
        else:
            print("No configs where custom FAST beats FAST+ in this sweep.")


if __name__ == "__main__":
    main()
