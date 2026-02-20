#!/usr/bin/env python3
"""
============================================================================
STEP 4: Hypothesis Test
============================================================================
This script tests the main hypothesis:

    "FAST+ universal tokenizer achieves ≥80% of the compression ratio 
     of a custom FAST tokenizer on cable manipulation data."

Usage:
    python step4_hypothesis_test.py

Output:
    - Hypothesis test result
    - Recommendations for the Intrinsic challenge
============================================================================
"""

import numpy as np
from pathlib import Path

# Paths
RESULTS_DIR = Path("./results")


def load_results():
    """Load tokenization results."""
    results_path = RESULTS_DIR / "tokenization_results.npy"
    
    if not results_path.exists():
        print(f"ERROR: {results_path} not found!")
        print("Run step3_fast_tokenization.py first.")
        exit(1)
    
    return np.load(results_path, allow_pickle=True).item()


def run_hypothesis_test(results: dict):
    """
    Test the main hypothesis.
    
    Hypothesis: FAST+ achieves ≥80% of custom FAST compression
    """
    
    print()
    print("=" * 60)
    print("HYPOTHESIS TEST")
    print("=" * 60)
    print()
    print("Hypothesis:")
    print('  "FAST+ universal tokenizer achieves ≥80% of the')
    print('   compression ratio of a custom FAST tokenizer')
    print('   on cable manipulation data."')
    print()
    
    # Get compression ratios (handle older results format gracefully)
    comp_custom = results.get('compression_custom')
    comp_plus = results.get('compression_plus')
    if comp_custom is None or comp_plus is None:
        print("WARNING: tokenization_results.npy is missing custom/FAST+ compression fields.")
        print("Please re-run step3_fast_tokenization.py to regenerate results.")
        return {
            'result': "UNKNOWN",
            'generalization_ratio': None,
            'threshold': 0.80,
            'compression_custom': comp_custom,
            'compression_plus': comp_plus,
        }
    
    # Generalization ratio = FAST+ compression / Custom compression
    generalization_ratio = comp_plus / comp_custom
    
    print("Results:")
    print(f"  Custom FAST compression:  {comp_custom:.2f}x")
    print(f"  FAST+ compression:        {comp_plus:.2f}x")
    print()
    print(f"  Generalization Ratio:     {generalization_ratio:.1%}")
    print(f"  (FAST+ compression / Custom FAST compression)")
    print()
    
    # Hypothesis test
    THRESHOLD = 0.80
    
    if generalization_ratio >= THRESHOLD:
        result = "SUPPORTED"
        print("  ╔══════════════════════════════════════╗")
        print("  ║     ✅ HYPOTHESIS SUPPORTED          ║")
        print("  ╚══════════════════════════════════════╝")
        print()
        print(f"  FAST+ achieves {generalization_ratio:.1%} of custom FAST compression,")
        print(f"  which exceeds the 80% threshold.")
    elif generalization_ratio >= 0.60:
        result = "PARTIALLY SUPPORTED"
        print("  ╔══════════════════════════════════════╗")
        print("  ║   ⚠️  HYPOTHESIS PARTIALLY SUPPORTED ║")
        print("  ╚══════════════════════════════════════╝")
        print()
        print(f"  FAST+ achieves {generalization_ratio:.1%} of custom FAST compression,")
        print(f"  which is below 80% but above 60%.")
    else:
        result = "NOT SUPPORTED"
        print("  ╔══════════════════════════════════════╗")
        print("  ║     ❌ HYPOTHESIS NOT SUPPORTED      ║")
        print("  ╚══════════════════════════════════════╝")
        print()
        print(f"  FAST+ achieves only {generalization_ratio:.1%} of custom FAST compression,")
        print(f"  which is below the 80% threshold.")
    
    return {
        'result': result,
        'generalization_ratio': generalization_ratio,
        'threshold': THRESHOLD,
        'compression_custom': comp_custom,
        'compression_plus': comp_plus,
    }


def print_recommendations(hypothesis_result: dict, results: dict):
    """Print recommendations based on results."""
    
    print()
    print("=" * 60)
    print("IMPLICATIONS FOR INTRINSIC CHALLENGE")
    print("=" * 60)
    print()
    
    result = hypothesis_result['result']
    ratio = hypothesis_result['generalization_ratio']
    
    if result == "UNKNOWN":
        print("RECOMMENDATION: Re-run step3_fast_tokenization.py")
        print()
        print("  - Current results file is missing required fields")
        print("  - This prevents a valid FAST+ vs custom FAST comparison")
        print()
        print("Next steps:")
        print("  1. python step3_fast_tokenization.py")
        print("  2. python step4_hypothesis_test.py")
        return
    elif result == "SUPPORTED":
        print("RECOMMENDATION: Use FAST+ directly for the challenge")
        print()
        print("  ✓ No need to train a custom tokenizer")
        print("  ✓ Focus engineering effort on policy architecture")
        print(f"  ✓ Expected training speedup: ~{hypothesis_result['compression_plus']:.1f}x vs naive")
        print()
        print("Implementation steps:")
        print("  1. Use pre-trained FAST+ from HuggingFace")
        print("  2. Fine-tune π0 or similar VLA on cable data")
        print("  3. Generate additional data in Gazebo simulation")
        
    elif result == "PARTIALLY SUPPORTED":
        print("RECOMMENDATION: Start with FAST+, consider custom training")
        print()
        print("  ✓ FAST+ is a reasonable starting point")
        print("  ⚠ Train custom tokenizer if performance plateaus")
        print(f"  ✓ Expected training speedup: ~{hypothesis_result['compression_plus']:.1f}x vs naive")
        print()
        print("Implementation steps:")
        print("  1. Start with FAST+ universal tokenizer")
        print("  2. Evaluate policy performance")
        print("  3. If needed, train custom FAST on challenge data")
        
    else:
        print("RECOMMENDATION: Train custom FAST tokenizer")
        print()
        print("  ✗ FAST+ does not generalize well to cable manipulation")
        print("  ✓ Must train custom tokenizer on challenge data")
        print(f"  ✓ Expected training speedup with custom: ~{hypothesis_result['compression_custom']:.1f}x vs naive")
        print()
        print("Implementation steps:")
        print("  1. Collect cable manipulation data (or use berkeley_cable_routing)")
        print("  2. Train custom FAST tokenizer (BPE vocabulary)")
        print("  3. Use custom tokenizer for policy training")
    
    # Reconstruction quality check
    mse = results.get('reconstruction_mse', 0)
    rmse = results.get('reconstruction_rmse', np.sqrt(mse))
    
    print()
    print("-" * 60)
    print("RECONSTRUCTION QUALITY CHECK")
    print("-" * 60)
    print()
    print(f"  RMSE: {rmse:.6f}")
    print(f"  Relative Error: {rmse * 100:.3f}%")
    print()
    
    if rmse < 0.01:
        print("  ✅ Reconstruction quality is EXCELLENT (<1% error)")
        print("     FAST will not degrade policy performance.")
    elif rmse < 0.05:
        print("  ✅ Reconstruction quality is GOOD (<5% error)")
        print("     FAST should work well for most tasks.")
    else:
        print("  ⚠️ Reconstruction quality may be insufficient (>5% error)")
        print("     Consider increasing quantization scale or using custom tokenizer.")


def print_summary_for_presentation():
    """Print a summary suitable for the 15-min empiricist presentation."""
    
    print()
    print("=" * 60)
    print("SUMMARY FOR PRESENTATION (15 min)")
    print("=" * 60)
    print()
    print("SLIDE 1: Hypothesis")
    print('  "FAST+ universal tokenizer achieves ≥80% of custom')
    print('   FAST compression on cable manipulation data."')
    print()
    print("SLIDE 2: Why This Matters")
    print("  - Paper shows FAST works; we test if FAST+ generalizes")
    print("  - Cable manipulation is challenge domain (Intrinsic)")
    print("  - Determines: use FAST+ directly vs train custom?")
    print()
    print("SLIDE 3: Method")
    print("  - Dataset: lerobot/berkeley_cable_routing (1,647 eps)")
    print("  - Compare: FAST (custom) vs FAST+ (universal) vs Naive")
    print("  - Metric: Compression ratio, reconstruction error")
    print()
    print("SLIDE 4: Results")
    print("  [Show results table from above]")
    print()
    print("SLIDE 5: Conclusion & Recommendation")
    print("  [Based on hypothesis test result]")


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " STEP 4: Hypothesis Test ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Load results
    results = load_results()
    
    # Run hypothesis test
    hypothesis_result = run_hypothesis_test(results)
    
    # Print recommendations
    print_recommendations(hypothesis_result, results)
    
    # Print presentation summary
    print_summary_for_presentation()
    
    # Save final results
    final_results_path = RESULTS_DIR / "hypothesis_test_results.npy"
    np.save(final_results_path, {**hypothesis_result, **results})
    print()
    print(f"Final results saved to {final_results_path}")
    
    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE!")
    print("=" * 60)
    print()
    print("Files generated:")
    print(f"  ./data/cable_trajectories.npz")
    print(f"  ./results/tokenization_results.npy")
    print(f"  ./results/hypothesis_test_results.npy")


if __name__ == "__main__":
    main()
