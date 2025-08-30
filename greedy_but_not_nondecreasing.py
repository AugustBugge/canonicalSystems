#!/usr/bin/env python3
"""
Find coin systems (subsets of {1..n}) where:
  - Greedy is optimal for all reachable amounts in 1..max_amount
  - BUT the system does NOT have nondecreasing consecutive gaps

Saves such systems to CSV.

Definitions:
- Greedy always takes the largest coin ≤ remaining amount.
- Optimal is computed by DP (minimum coins).
- "Reachable" means the amount can be formed exactly by the coin system.
- Nondecreasing gaps: for sorted coins c1<...<ck, let gaps di = c_{i+1}-c_i.
  The rule is "d1 <= d2 <= ...". We want systems that VIOLATE this.

Usage examples:
  python greedy_but_not_nondecreasing.py --n 15 --max-amount 200
  python greedy_but_not_nondecreasing.py --n 18 --max-amount 250 --no-require-1
  python greedy_but_not_nondecreasing.py --n 16 --max-amount 300 --progress 200000

Notes:
- Enumeration is exponential in n (2^n subsets). Keep n modest.
- By default we require coin 1 to be present (common for currency systems). Toggle off with --no-require-1.
- Systems with fewer than 3 coins cannot violate the nondecreasing-gap rule and are skipped.
- Systems with ZERO reachable amounts in 1..max_amount are skipped (to avoid vacuous "successes").
"""

from __future__ import annotations
import argparse
import csv
import itertools
from typing import List, Tuple, Optional


# ---------------- Greedy / Optimal ----------------

def greedy_num_coins(amount: int, coins_desc: List[int]) -> Optional[int]:
    """Greedy: take as many as possible of the largest coin, then next, etc.
    Returns coin count or None if not exact.
    """
    remaining = amount
    count = 0
    for c in coins_desc:
        if remaining <= 0:
            break
        k = remaining // c
        if k:
            count += k
            remaining -= k * c
    return count if remaining == 0 else None


def dp_min_coins_up_to(max_amount: int, coins_asc: List[int]) -> List[int]:
    """Unbounded coin-change DP for min coins up to max_amount.
    dp[x] = min #coins to make x, or INF if unreachable.
    """
    INF = 10**9
    dp = [INF] * (max_amount + 1)
    dp[0] = 0
    for x in range(1, max_amount + 1):
        best = dp[x]
        for c in coins_asc:
            if c <= x:
                cand = dp[x - c] + 1
                if cand < best:
                    best = cand
        dp[x] = best
    return dp


def test_greedy_optimal_for_reachable(coins: List[int], max_amount: int) -> Tuple[bool, int]:
    """Return (greedy_ok_for_all_reachable, reachable_count) for amounts 1..max_amount."""
    coins_asc = sorted(coins)
    coins_desc = list(reversed(coins_asc))
    dp = dp_min_coins_up_to(max_amount, coins_asc)
    INF = 10**9

    reachable_count = 0
    for a in range(1, max_amount + 1):
        if dp[a] >= INF:
            continue
        reachable_count += 1
        g = greedy_num_coins(a, coins_desc)
        if g != dp[a]:
            return False, reachable_count
    return True, reachable_count


# ---------------- Gap Rule ----------------

def is_nondecreasing_gaps(coins_sorted: List[int]) -> bool:
    """True if consecutive gaps are nondecreasing; vacuously True for len < 3."""
    if len(coins_sorted) < 3:
        return True
    gaps = [coins_sorted[i+1] - coins_sorted[i] for i in range(len(coins_sorted)-1)]
    # Check d1 <= d2 <= ...:
    for i in range(len(gaps) - 1):
        if gaps[i] > gaps[i+1]:
            return False
    return True


# ---------------- Systems Generator ----------------

def generate_all_systems(n: int, require_1: bool) -> List[Tuple[int, ...]]:
    """All subsets of {1..n} (excluding empty), optionally requiring 1 ∈ system."""
    universe = list(range(1, n + 1))
    start_r = 1 if not require_1 else 0  # with require_1 we enforce inclusion below
    systems = []
    if require_1:
        rest = universe[1:]  # must include 1
        for r in range(0, len(rest) + 1):
            for combo in itertools.combinations(rest, r):
                systems.append((1, *combo))
    else:
        for r in range(1, n + 1):
            for combo in itertools.combinations(universe, r):
                systems.append(combo)
    return systems


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Find coin systems where greedy is optimal but gaps are NOT nondecreasing."
    )
    ap.add_argument("--n", type=int, required=True, help="Max coin denomination; systems are subsets of {1..n}.")
    ap.add_argument("--max-amount", type=int, default=100, help="Test amounts 1..MAX (default: 100).")
    ap.add_argument("--require-1", dest="require_1", action="store_true", default=True,
                    help="Require coin 1 in every system (default: on).")
    ap.add_argument("--no-require-1", dest="require_1", action="store_false",
                    help="Allow systems without coin 1.")
    ap.add_argument("--outfile", type=str, default="greedy_but_not_nondecreasing.csv",
                    help="CSV output path (default: greedy_but_not_nondecreasing.csv)")
    ap.add_argument("--progress", type=int, default=0,
                    help="Print a progress line after this many systems (0 = silent).")
    args = ap.parse_args()

    n = args.n
    max_amount = args.max_amount
    require_1 = args.require_1

    print(f"Enumerating all systems from {{1..{n}}}{' (requiring coin 1)' if require_1 else ''}...")
    systems = generate_all_systems(n, require_1)
    total = len(systems)
    print(f"Total systems to consider: {total}")

    kept = []  # systems where greedy OK AND gaps not nondecreasing

    for idx, coins in enumerate(systems, 1):
        # Need at least 3 coins to violate the rule
        if len(coins) < 3:
            if args.progress and (idx % args.progress == 0):
                print(f"Processed {idx}/{total} systems...")
            continue

        coins_sorted = sorted(coins)
        if is_nondecreasing_gaps(coins_sorted):
            # We only want systems that violate the rule
            if args.progress and (idx % args.progress == 0):
                print(f"Processed {idx}/{total} systems...")
            continue

        greedy_ok, reachable_cnt = test_greedy_optimal_for_reachable(coins_sorted, max_amount)
        # Skip degenerate systems with zero reachable amounts to avoid vacuous truth
        if greedy_ok and reachable_cnt > 0:
            kept.append(coins_sorted)

        if args.progress and (idx % args.progress == 0):
            print(f"Processed {idx}/{total} systems...")

    # Write results
    with open(args.outfile, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coins"])
        for coins in kept:
            w.writerow([" ".join(map(str, coins))])

    print("\n=== Summary ===")
    print(f"Total systems considered:         {total}")
    print(f"Systems violating gap rule:       "
          f"{sum(1 for c in systems if len(c)>=3 and not is_nondecreasing_gaps(sorted(c)))}")
    print(f"Greedy-OK & NOT nondecreasing:    {len(kept)}")
    print(f"Wrote: {args.outfile}")


if __name__ == "__main__":
    main()
