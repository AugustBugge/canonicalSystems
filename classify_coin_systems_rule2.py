#!/usr/bin/env python3
"""
Classify coin systems from subsets of {1..n} by three binary properties:

  A) Canonical (greedy optimal on all reachable amounts in 1..max_amount)
  B) Nondecreasing consecutive gaps
  C) Rule 2 holds (no Rule 2 violations found up to max_amount)

Rule 2 (pairwise one-step greedy test):
  For each adjacent pair (c_i, c_{i+1}) with coins sorted strictly increasing:
    Consider all amounts A where greedy would take EXACTLY ONE c_{i+1} and no larger coin:
        A ∈ [c_{i+1}, min(2*c_{i+1}-1, c_{i+2}-1, max_amount)]   (with c_{i+2}=+∞ if none)
      Let r = A - c_{i+1}. Compute:
        opt_≤ci(r): optimal coins to make r using only coins ≤ c_i  (tiny DP up to r ≤ c_{i+1}-1)
        opt(A):     optimal coins to make A using the full system   (from global DP)
      If 1 + opt_≤ci(r) > opt(A), flag a Rule 2 violation.
  (This catches classic failures like {1,2,5,13,23} at 26; {1,2,5,13,27} at 39.)

Outputs:
  - Counts and percentages for 8 categories (canonical × ND-gaps × Rule2).
  - The original 4-category summary (canonical × ND-gaps).
  - Conditional percentages:
      * % of canonical systems that follow ND-gaps
      * % of ND-gaps systems that are canonical
      * % of canonical systems that satisfy Rule 2
      * % of Rule-2 systems that are canonical

Options:
  --n N                     Max coin denomination; systems are subsets of {1..n}. (required)
  --max-amount M            Test amounts 1..M (default: 100)
  --require-1 / --no-require-1
                            Require coin 1 to be present (default: on)
  --canonical-requires-reachable / --canonical-allows-empty
                            If on (default), a system with zero reachable amounts is NOT canonical.
                            If off, such a system counts as canonical (vacuously).
  --progress K              Print progress every K systems (default: 0 = silent)
  --write-csv               If set, write CSVs listing systems in each of the 8 categories.

Usage:
  python classify_coin_systems_rule2.py --n 12 --max-amount 200 --write-csv
"""

from __future__ import annotations
import argparse
import csv
import itertools
from typing import List, Tuple, Optional


INF = 10**9


# ---------- Greedy / DP ----------

def greedy_num_coins(amount: int, coins_desc: List[int]) -> Optional[int]:
    """Greedy: take as many as possible of the largest coin, then next, etc.
    Returns coin count, or None if not exact.
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


def check_canonical(coins_sorted: List[int], max_amount: int, require_reachable: bool) -> Tuple[bool, int]:
    """Canonical ⇔ greedy matches optimal on all reachable amounts in 1..max_amount."""
    coins_desc = list(reversed(coins_sorted))
    dp_full = dp_min_coins_up_to(max_amount, coins_sorted)
    reachable = 0
    for a in range(1, max_amount + 1):
        if dp_full[a] >= INF:
            continue
        reachable += 1
        g = greedy_num_coins(a, coins_desc)
        if g != dp_full[a]:
            return (False, reachable)
    if require_reachable and reachable == 0:
        return (False, reachable)
    return (True, reachable)


# ---------- Gap Rule ----------

def is_nondecreasing_gaps(coins_sorted: List[int]) -> bool:
    """True if consecutive gaps are nondecreasing; vacuously True for len < 3."""
    if len(coins_sorted) < 3:
        return True
    gaps = [coins_sorted[i+1] - coins_sorted[i] for i in range(len(coins_sorted)-1)]
    return all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1))


# ---------- Rule 2 ----------

def tiny_dp_up_to(limit: int, coins_asc: List[int]) -> List[int]:
    """DP limited to amounts 0..limit with given coin set."""
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


def rule2_holds(coins_sorted: List[int], max_amount: int, dp_full: Optional[List[int]] = None) -> bool:
    """
    Implements Rule 2 check as described in the prompt.
    Returns True if NO Rule 2 violation is found for amounts up to max_amount.
    """
    k = len(coins_sorted)
    if k < 2:
        return True  # vacuously
    coins_asc = coins_sorted

    # If caller didn't provide dp_full for opt(A), compute it once.
    if dp_full is None:
        dp_full = dp_min_coins_up_to(max_amount, coins_asc)

    for i in range(k - 1):
        ci = coins_asc[i]
        cnext = coins_asc[i + 1]
        # Next larger coin (if any)
        c_larger = coins_asc[i + 2] if (i + 2) < k else INF

        # Amounts where greedy would take exactly ONE cnext and no larger coin:
        #   A ∈ [cnext, min(2*cnext - 1, c_larger - 1, max_amount)]
        A_lo = cnext
        A_hi = min(2 * cnext - 1, c_larger - 1, max_amount)
        if A_lo > A_hi:
            continue

        # Tiny DP for opt_≤ci(r) only needs amounts up to cnext-1
        dp_le_ci = tiny_dp_up_to(cnext - 1, coins_asc[: i + 1])

        for A in range(A_lo, A_hi + 1):
            r = A - cnext  # remainder after taking exactly one cnext
            opt_r = dp_le_ci[r]
            opt_A = dp_full[A]
            if opt_A >= INF:
                # If A is unreachable at all, skip; Rule 2 is meant for reachable A.
                continue
            # Violation if taking one cnext then optimal remainder is worse than true optimum:
            if 1 + opt_r > opt_A:
                return False  # Rule 2 violated

    return True  # No violations found


# ---------- Systems Enumeration ----------

def generate_all_systems(n: int, require_1: bool):
    """Yield all non-empty subsets of {1..n}; enforce inclusion of 1 if require_1."""
    universe = list(range(1, n + 1))
    if require_1:
        rest = universe[1:]  # must include 1
        for r in range(0, len(rest) + 1):
            for combo in itertools.combinations(rest, r):
                yield tuple([1, *combo])
    else:
        for r in range(1, n + 1):
            for combo in itertools.combinations(universe, r):
                yield tuple(combo)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Classify coin systems by canonicality, nondecreasing gaps, and Rule 2.")
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
                    help="Write CSVs listing systems in each of the 8 categories.")
    args = ap.parse_args()

    n = args.n
    max_amount = args.max_amount
    require_1 = args.require_1
    require_reachable = args.require_reachable

    # 8 categories: canonical/non × ND-gap yes/no × Rule2 yes/no
    # Indexing map: (canonical, nd, rule2) -> list
    cats = {
        (True,  True,  True ): [],
        (True,  True,  False): [],
        (True,  False, True ): [],
        (True,  False, False): [],
        (False, True,  True ): [],
        (False, True,  False): [],
        (False, False, True ): [],
        (False, False, False): [],
    }

    total = 0
    for idx, coins in enumerate(generate_all_systems(n, require_1), start=1):
        total += 1
        coins_sorted = list(coins)  # already sorted
        # Precompute DPs once per system to share between checks
        dp_full = dp_min_coins_up_to(max_amount, coins_sorted)

        # Canonical
        coins_desc = list(reversed(coins_sorted))
        reachable = 0
        canonical = True
        for a in range(1, max_amount + 1):
            if dp_full[a] >= INF:
                continue
            reachable += 1
            g = greedy_num_coins(a, coins_desc)
            if g != dp_full[a]:
                canonical = False
                break
        if require_reachable and reachable == 0:
            canonical = False

        # Nondecreasing gaps
        nd = is_nondecreasing_gaps(coins_sorted)

        # Rule 2
        r2_ok = rule2_holds(coins_sorted, max_amount, dp_full=dp_full)

        cats[(canonical, nd, r2_ok)].append(coins_sorted)

        if args.progress and (idx % args.progress == 0):
            print(f"Processed {idx} systems...")

    # Helper for percent strings
    def pct(x: int, denom: int) -> str:
        return f"{(100.0 * x / denom):.2f}%" if denom > 0 else "n/a"

    # Tally
    counts = {k: len(v) for k, v in cats.items()}

    # Original 4-category rollups (by ND only)
    c_nd_can = counts[(True, True, True)] + counts[(True, True, False)]
    c_nd_non = counts[(False, True, True)] + counts[(False, True, False)]
    c_vio_can = counts[(True, False, True)] + counts[(True, False, False)]
    c_vio_non = counts[(False, False, True)] + counts[(False, False, False)]

    canonical_total = c_nd_can + c_vio_can
    nd_total        = c_nd_can + c_nd_non
    rule2_total     = (
        counts[(True, True, True)] + counts[(True, False, True)] +
        counts[(False, True, True)] + counts[(False, False, True)]
    )

    print("\n=== Classification Summary (n={}, max_amount={}, require_1={}, require_reachable={}) ==="
          .format(n, max_amount, require_1, require_reachable))
    print(f"Total systems considered: {total}\n")

    # 8-category table
    rows = [
        ("Canonical & ND-gaps & Rule2",              counts[(True,  True,  True )]),
        ("Canonical & ND-gaps & NOT Rule2",          counts[(True,  True,  False)]),
        ("Canonical & NOT ND-gaps & Rule2",          counts[(True,  False, True )]),
        ("Canonical & NOT ND-gaps & NOT Rule2",      counts[(True,  False, False)]),
        ("Non-canonical & ND-gaps & Rule2",          counts[(False, True,  True )]),
        ("Non-canonical & ND-gaps & NOT Rule2",      counts[(False, True,  False)]),
        ("Non-canonical & NOT ND-gaps & Rule2",      counts[(False, False, True )]),
        ("Non-canonical & NOT ND-gaps & NOT Rule2",  counts[(False, False, False)]),
    ]
    print("— 8-way breakdown —")
    for name, cnt in rows:
        print(f"{name:45s} {cnt:8d}  ({pct(cnt, total)})")
    print()

    # Original 4-category summary (by ND-gaps only)
    print("— 4-way breakdown (canonical × ND-gaps) —")
    print(f"Canonical & ND-gaps:            {c_nd_can:8d}  ({pct(c_nd_can, total)})")
    print(f"Canonical & NOT ND-gaps:        {c_vio_can:8d}  ({pct(c_vio_can, total)})")
    print(f"Non-canonical & ND-gaps:        {c_nd_non:8d}  ({pct(c_nd_non, total)})")
    print(f"Non-canonical & NOT ND-gaps:    {c_vio_non:8d}  ({pct(c_vio_non, total)})\n")

    # Conditional percentages (requested + Rule 2 analogs)
    print("— Conditional percentages —")
    print(f"% of canonical systems that follow ND-gaps: {pct(c_nd_can, canonical_total)} "
          f"(= {c_nd_can}/{canonical_total})")
    print(f"% of ND-gaps systems that are canonical:   {pct(c_nd_can, nd_total)} "
          f"(= {c_nd_can}/{nd_total})")
    print(f"% of canonical systems that satisfy Rule2: {pct(counts[(True, True, True)] + counts[(True, False, True)], canonical_total)} "
          f"(= {counts[(True, True, True)] + counts[(True, False, True)]}/{canonical_total})")
    print(f"% of Rule2 systems that are canonical:     {pct(counts[(True, True, True)] + counts[(True, False, True)], rule2_total)} "
          f"(= {counts[(True, True, True)] + counts[(True, False, True)]}/{rule2_total})")

    # Optional CSVs
    if args.write_csv:
        name_map = {
            (True,  True,  True ):  "cat1_canonical_nd_rule2.csv",
            (True,  True,  False):  "cat2_canonical_nd_not_rule2.csv",
            (True,  False, True ):  "cat3_canonical_notnd_rule2.csv",
            (True,  False, False):  "cat4_canonical_notnd_not_rule2.csv",
            (False, True,  True ):  "cat5_noncanonical_nd_rule2.csv",
            (False, True,  False):  "cat6_noncanonical_nd_not_rule2.csv",
            (False, False, True ):  "cat7_noncanonical_notnd_rule2.csv",
            (False, False, False):  "cat8_noncanonical_notnd_not_rule2.csv",
        }
        for key, systems in cats.items():
            path = name_map[key]
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["coins"])
                for s in systems:
                    w.writerow([" ".join(map(str, s))])
        print("\nWrote CSVs for all 8 categories.")


if __name__ == "__main__":
    main()
