#!/usr/bin/env python3
"""
Classify coin systems by:
  - Canonical (greedy optimal on all reachable amounts in 1..max_amount)
  - Rule 2 (no violations found up to max_amount)

Outputs percentages for:
  1) canonical & rule2
  2) non-canonical & rule2
  3) canonical & NOT rule2
  4) non-canonical & NOT rule2

Rule 2 (one-step greedy test):
  For each adjacent pair (c_i, c_{i+1}) in sorted coins:
    Consider amounts A where greedy would take EXACTLY ONE c_{i+1} and no larger coin:
      A ∈ [c_{i+1}, min(2*c_{i+1}-1, c_{i+2}-1, max_amount)], with c_{i+2}=+∞ if none.
    Let r = A - c_{i+1}.
    Compute:
      opt_≤ci(r): optimal #coins to make r using coins ≤ c_i (tiny DP up to c_{i+1}-1)
      opt(A):     optimal #coins to make A using full system (global DP)
    If 1 + opt_≤ci(r) > opt(A) for any such A, Rule 2 FAILS.

Usage examples:
  python canonical_rule2_percentages.py --n 12 --max-amount 200
  python canonical_rule2_percentages.py --n 14 --max-amount 300 --no-require-1 --progress 100000

Notes:
  - Search space is exponential in n (2^n, or 2^(n-1) if --require-1).
  - By default, a system with ZERO reachable amounts in 1..max_amount is NOT considered canonical
    (to avoid vacuous truth). Toggle with --canonical-allows-empty if you prefer.
"""

from __future__ import annotations
import argparse
import itertools
from typing import List, Optional

INF = 10**9


# ---------- Greedy / DP ----------

def greedy_num_coins(amount: int, coins_desc: List[int]) -> Optional[int]:
    """Greedy: take as many as possible of the largest coin, then next, etc."""
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
    """Unbounded coin-change DP for minimum coins up to max_amount."""
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


def tiny_dp_up_to(limit: int, coins_asc: List[int]) -> List[int]:
    """DP limited to 0..limit with given coin set."""
    dp = [INF] * (limit + 1)
    dp[0] = 0
    for x in range(1, limit + 1):
        best = dp[x]
        for c in coins_asc:
            if c <= x:
                cand = dp[x - c] + 1
                if cand < best:
                    best = cand
        dp[x] = best
    return dp


# ---------- Property checks ----------

def is_canonical(coins_sorted: List[int], max_amount: int, require_reachable: bool) -> bool:
    """Canonical ⇔ greedy matches optimal on all reachable amounts in 1..max_amount."""
    dp_full = dp_min_coins_up_to(max_amount, coins_sorted)
    coins_desc = list(reversed(coins_sorted))

    reachable = 0
    for a in range(1, max_amount + 1):
        if dp_full[a] >= INF:
            continue
        reachable += 1
        g = greedy_num_coins(a, coins_desc)
        if g != dp_full[a]:
            return False
    if require_reachable and reachable == 0:
        return False
    return True


def rule2_holds(coins_sorted: List[int], max_amount: int) -> bool:
    """
    Rule 2 as described in the header comment.
    Returns True if NO violation is found up to max_amount.
    """
    k = len(coins_sorted)
    if k < 2:
        return True  # vacuously

    dp_full = dp_min_coins_up_to(max_amount, coins_sorted)

    for i in range(k - 1):
        ci = coins_sorted[i]
        cnext = coins_sorted[i + 1]
        c_larger = coins_sorted[i + 2] if (i + 2) < k else INF

        A_lo = cnext
        A_hi = min(2 * cnext - 1, c_larger - 1, max_amount)
        if A_lo > A_hi:
            continue

        # tiny DP for amounts up to cnext-1 using coins ≤ ci
        dp_le_ci = tiny_dp_up_to(cnext - 1, coins_sorted[: i + 1])

        for A in range(A_lo, A_hi + 1):
            if dp_full[A] >= INF:
                # If A is unreachable at all, skip this A
                continue
            r = A - cnext
            opt_r = dp_le_ci[r]
            if 1 + opt_r > dp_full[A]:
                return False  # violation found
    return True


# ---------- Systems enumeration ----------

def generate_all_systems(n: int, require_1: bool):
    """Yield all non-empty subsets of {1..n}; enforce inclusion of 1 if require_1."""
    universe = list(range(1, n + 1))
    if require_1:
        rest = universe[1:]
        for r in range(0, len(rest) + 1):
            for combo in itertools.combinations(rest, r):
                yield tuple([1, *combo])
    else:
        for r in range(1, n + 1):
            for combo in itertools.combinations(universe, r):
                yield tuple(combo)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Percentages for (canonical, rule2) categories over coin systems up to n.")
    ap.add_argument("--n", type=int, required=True, help="Max coin denomination; systems are subsets of {1..n}.")
    ap.add_argument("--max-amount", type=int, default=100, help="Test amounts 1..MAX (default: 100).")
    ap.add_argument("--require-1", dest="require_1", action="store_true", default=True,
                    help="Require coin 1 in every system (default: on).")
    ap.add_argument("--no-require-1", dest="require_1", action="store_false",
                    help="Allow systems without coin 1.")
    ap.add_argument("--canonical-requires-reachable", dest="require_reachable", action="store_true", default=True,
                    help="Canonical requires at least one reachable amount (default: on).")
    ap.add_argument("--canonical-allows-empty", dest="require_reachable", action="store_false",
                    help="Allow systems with zero reachable amounts to count as canonical.")
    ap.add_argument("--progress", type=int, default=0, help="Print progress every K systems (0 = silent).")
    args = ap.parse_args()

    n = args.n
    max_amount = args.max_amount
    require_1 = args.require_1
    require_reachable = args.require_reachable

    # Counters for 4 categories
    cnt_can_r2 = 0
    cnt_noncan_r2 = 0
    cnt_can_notr2 = 0
    cnt_noncan_notr2 = 0
    total = 0

    for idx, coins in enumerate(generate_all_systems(n, require_1), start=1):
        total += 1
        coins_sorted = list(coins)  # already sorted
        canonical = is_canonical(coins_sorted, max_amount, require_reachable)
        r2_ok = rule2_holds(coins_sorted, max_amount)

        if canonical and r2_ok:
            cnt_can_r2 += 1
        elif (not canonical) and r2_ok:
            cnt_noncan_r2 += 1
        elif canonical and (not r2_ok):
            cnt_can_notr2 += 1
        else:
            cnt_noncan_notr2 += 1

        if args.progress and (idx % args.progress == 0):
            print(f"Processed {idx} systems...")

    def pct(x: int, denom: int) -> str:
        return f"{(100.0 * x / denom):.2f}%" if denom > 0 else "n/a"

    print("\n=== Results ===")
    print(f"Universe: subsets of {{1..{n}}}{' with 1 required' if require_1 else ''}")
    print(f"Amounts tested: 1..{max_amount}")
    print(f"Canonical requires at least one reachable amount: {require_reachable}")
    print(f"Total systems considered: {total}\n")

    print(f"Canonical & Rule 2:             {cnt_can_r2:8d}  ({pct(cnt_can_r2, total)})")
    print(f"Non-canonical & Rule 2:         {cnt_noncan_r2:8d}  ({pct(cnt_noncan_r2, total)})")
    print(f"Canonical & NOT Rule 2:         {cnt_can_notr2:8d}  ({pct(cnt_can_notr2, total)})")
    print(f"Non-canonical & NOT Rule 2:     {cnt_noncan_notr2:8d}  ({pct(cnt_noncan_notr2, total)})")


if __name__ == "__main__":
    main()
