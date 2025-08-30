#!/usr/bin/env python3
"""
Classify coin systems from subsets of {1..n} by:
  - Canonical (greedy optimal on all reachable amounts in 1..max_amount) vs non-canonical
  - Nondecreasing consecutive gaps vs not

Outputs counts and percentages for four categories:
  1) canonical & nondecreasing
  2) canonical & NOT nondecreasing
  3) non-canonical & nondecreasing
  4) non-canonical & NOT nondecreasing

Also prints:
  - % of canonical systems that follow the rule
  - % of systems that follow the rule that are canonical
"""

from __future__ import annotations
import argparse
import csv
import itertools
from typing import List, Tuple, Optional


# ---------- Greedy / DP ----------

def greedy_num_coins(amount: int, coins_desc: List[int]) -> Optional[int]:
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


def check_canonical(coins_sorted: List[int], max_amount: int, require_reachable: bool) -> Tuple[bool, int]:
    coins_desc = list(reversed(coins_sorted))
    dp = dp_min_coins_up_to(max_amount, coins_sorted)
    INF = 10**9
    reachable_count = 0
    for a in range(1, max_amount + 1):
        if dp[a] >= INF:
            continue
        reachable_count += 1
        g = greedy_num_coins(a, coins_desc)
        if g != dp[a]:
            return (False, reachable_count)
    if require_reachable and reachable_count == 0:
        return (False, reachable_count)
    return (True, reachable_count)


# ---------- Gap Rule ----------

def is_nondecreasing_gaps(coins_sorted: List[int]) -> bool:
    if len(coins_sorted) < 3:
        return True
    gaps = [coins_sorted[i+1] - coins_sorted[i] for i in range(len(coins_sorted)-1)]
    return all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))


# ---------- Systems Enumeration ----------

def generate_all_systems(n: int, require_1: bool):
    universe = list(range(1, n + 1))
    if require_1:
        rest = universe[1:]  # enforce inclusion of 1
        for r in range(0, len(rest) + 1):
            for combo in itertools.combinations(rest, r):
                yield tuple([1, *combo])
    else:
        for r in range(1, n + 1):
            for combo in itertools.combinations(universe, r):
                yield tuple(combo)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Classify coin systems by canonicality and nondecreasing gaps.")
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
    ap.add_argument("--write-csv", action="store_true", default=False,
                    help="Write CSVs listing systems in each category.")
    args = ap.parse_args()

    n = args.n
    max_amount = args.max_amount
    require_1 = args.require_1
    require_reachable = args.require_reachable

    cat1 = []  # canonical & nondecreasing
    cat2 = []  # canonical & NOT nondecreasing
    cat3 = []  # non-canonical & nondecreasing
    cat4 = []  # non-canonical & NOT nondecreasing

    total = 0
    for idx, coins in enumerate(generate_all_systems(n, require_1), start=1):
        total += 1
        coins_sorted = list(coins)  # already sorted
        nd = is_nondecreasing_gaps(coins_sorted)
        canonical, _ = check_canonical(coins_sorted, max_amount, require_reachable)

        if canonical and nd:
            cat1.append(coins_sorted)
        elif canonical and not nd:
            cat2.append(coins_sorted)
        elif (not canonical) and nd:
            cat3.append(coins_sorted)
        else:
            cat4.append(coins_sorted)

        if args.progress and (idx % args.progress == 0):
            print(f"Processed {idx} systems...")

    # Helper for percent strings
    def pct(x: int, denom: int) -> str:
        return f"{(100.0 * x / denom):.2f}%" if denom > 0 else "n/a"

    # Base counts
    c1, c2, c3, c4 = len(cat1), len(cat2), len(cat3), len(cat4)
    canonical_total = c1 + c2
    nd_total = c1 + c3  # systems that follow the rule

    print("\n=== Classification Summary ===")
    print(f"Universe: subsets of {{1..{n}}}{' with 1 required' if require_1 else ''}")
    print(f"Amounts tested: 1..{max_amount}")
    print(f"Canonical requires at least one reachable amount: {require_reachable}")
    print(f"Total systems considered: {total}\n")

    print(f"1) Canonical & Nondecreasing gaps:       {c1}  ({pct(c1, total)})")
    print(f"2) Canonical & NOT nondecreasing gaps:   {c2}  ({pct(c2, total)})")
    print(f"3) Non-canonical & Nondecreasing gaps:   {c3}  ({pct(c3, total)})")
    print(f"4) Non-canonical & NOT nondecreasing:    {c4}  ({pct(c4, total)})\n")

    # New conditional percentages
    print("— Conditional breakdown —")
    print(f"% of canonical systems that follow the rule: {pct(c1, canonical_total)} "
          f"(= {c1}/{canonical_total})")
    print(f"% of systems that follow the rule that are canonical: {pct(c1, nd_total)} "
          f"(= {c1}/{nd_total})")

    # Optional CSVs
    if args.write_csv:
        def write_csv(path: str, systems: list[list[int]]):
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["coins"])
                for s in systems:
                    w.writerow([" ".join(map(str, s))])
        write_csv("cat1_canonical_nondecreasing.csv", cat1)
        write_csv("cat2_canonical_not_nondecreasing.csv", cat2)
        write_csv("cat3_noncanonical_nondecreasing.csv", cat3)
        write_csv("cat4_noncanonical_not_nondecreasing.csv", cat4)
        print("\nWrote CSVs:")
        print("  cat1_canonical_nondecreasing.csv")
        print("  cat2_canonical_not_nondecreasing.csv")
        print("  cat3_noncanonical_nondecreasing.csv")
        print("  cat4_noncanonical_not_nondecreasing.csv")


if __name__ == "__main__":
    main()
