#!/usr/bin/env python3
"""
Enumerate coin systems with nondecreasing consecutive gaps and test greedy optimality.

A coin system here is an increasing sequence of coin values within {1,2,...,n}.
We only include systems whose consecutive differences (gaps) are nondecreasing:
if the coins are c1 < c2 < ... < ck, then (c2-c1) <= (c3-c2) <= ... (nondecreasing).

Options:
- Require coin '1' to be present (default on).
- Test greedy-vs-optimal for all reachable amounts in 1..max_amount.
- Output CSVs for systems where greedy succeeds vs fails (with first counterexample).

Usage:
  python enumerate_convex_coin_systems.py --n 15 --max-amount 200
  python enumerate_convex_coin_systems.py --n 12 --max-amount 150 --no-require-1 --progress 10000
"""

from __future__ import annotations
import argparse
import csv
from typing import List, Tuple, Optional, Iterable


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
    """Classic unbounded-coin DP for min coins up to max_amount.
    dp[x] = min number of coins to make x, or INF if unreachable.
    """
    INF = 10**9
    dp = [INF] * (max_amount + 1)
    dp[0] = 0
    for x in range(1, max_amount + 1):
        best = INF
        for c in coins_asc:
            if c <= x and dp[x - c] + 1 < best:
                best = dp[x - c] + 1
        dp[x] = best
    return dp


def test_system(coins: List[int], max_amount: int) -> Tuple[bool, Optional[int], Optional[int], Optional[int]]:
    """Test one system.
    Returns (all_greedy_ok, first_bad_amount, greedy_at_bad, optimal_at_bad).
    Only checks reachable amounts (by DP).
    """
    coins_asc = sorted(coins)
    coins_desc = list(reversed(coins_asc))
    dp = dp_min_coins_up_to(max_amount, coins_asc)
    INF = 10**9
    for a in range(1, max_amount + 1):
        if dp[a] >= INF:  # unreachable
            continue
        g = greedy_num_coins(a, coins_desc)
        if g != dp[a]:
            return (False, a, g if g is not None else -1, dp[a])
    return (True, None, None, None)


# ---------- Generator for nondecreasing-gap systems ----------

def generate_nondecreasing_gap_systems(n: int, require_1: bool = True) -> Iterable[List[int]]:
    """Yield all increasing sequences of coins within {1..n} whose consecutive gaps are nondecreasing.
       Sequences may have length 1 or more. If require_1, every sequence must start at 1.
    """
    if require_1:
        starts = [1]
    else:
        starts = list(range(1, n + 1))

    for start in starts:
        # Yield the singleton system [start]
        yield [start]

        # Now extend with nondecreasing gaps.
        # We define a recursive builder that keeps: last coin value, minimum allowed next gap.
        def extend(seq: List[int], last: int, min_gap: int) -> None:
            # Next gap g must be >= min_gap and ensure last+g <= n
            g = min_gap
            while last + g <= n:
                next_coin = last + g
                new_seq = seq + [next_coin]
                yield new_seq
                # Recurse: next gap must be >= g (nondecreasing)
                yield from extend(new_seq, next_coin, g)
                g += 1  # try larger gap

        # Start extensions with any first gap g >= 1
        yield from extend([start], start, 1)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Check greedy optimality for coin systems with nondecreasing gaps.")
    ap.add_argument("--n", type=int, required=True, help="Max coin denomination; coins are within {1..n}.")
    ap.add_argument("--max-amount", type=int, default=100, help="Test amounts 1..MAX (default: 100).")
    ap.add_argument("--require-1", dest="require_1", action="store_true", default=True,
                    help="Require coin 1 to be included (default: on).")
    ap.add_argument("--no-require-1", dest="require_1", action="store_false",
                    help="Allow systems that do not include coin 1.")
    ap.add_argument("--successes-out", type=str, default="successes.csv",
                    help="CSV for systems where greedy succeeds.")
    ap.add_argument("--failures-out", type=str, default="failures.csv",
                    help="CSV for systems where greedy fails.")
    ap.add_argument("--progress", type=int, default=0,
                    help="Print a progress line after this many systems (0 = silent).")
    args = ap.parse_args()

    n = args.n
    max_amount = args.max_amount
    require_1 = args.require_1

    print(f"Generating systems in {{1..{n}}} with nondecreasing gaps"
          f"{' and requiring coin 1' if require_1 else ''}...")

    successes = []
    failures = []

    # Enumerate and test
    for idx, coins in enumerate(generate_nondecreasing_gap_systems(n, require_1=require_1), start=1):
        ok, bad_amt, g_bad, opt_bad = test_system(coins, max_amount)
        if ok:
            successes.append(tuple(coins))
        else:
            failures.append((tuple(coins), bad_amt, g_bad, opt_bad))

        if args.progress and (idx % args.progress == 0):
            print(f"Processed {idx} systems...")

    # Write outputs
    with open(args.successes_out, "w", newline="") as f_ok:
        w = csv.writer(f_ok)
        w.writerow(["coins"])
        for coins in successes:
            w.writerow([" ".join(map(str, coins))])

    with open(args.failures_out, "w", newline="") as f_bad:
        w = csv.writer(f_bad)
        w.writerow(["coins", "first_counterexample_amount", "greedy_num_coins", "optimal_num_coins"])
        for coins, bad_amt, g_bad, opt_bad in failures:
            w.writerow([" ".join(map(str, coins)), bad_amt, g_bad, opt_bad])

    # Summary
    total = len(successes) + len(failures)
    print("\n=== Summary ===")
    print(f"Total systems (nondecreasing gaps): {total}")
    print(f"Greedy OK:                         {len(successes)}")
    print(f"Greedy fails:                      {len(failures)}")
    if failures:
        show = min(5, len(failures))
        print(f"Examples of failures (showing {show}):")
        for (coins, bad_amt, g_bad, opt_bad) in failures[:show]:
            print(f"  coins={coins}, first bad amount={bad_amt}, greedy={g_bad}, optimal={opt_bad}")
    print(f"Wrote successes to: {args.successes_out}")
    print(f"Wrote failures to:  {args.failures_out}")


if __name__ == "__main__":
    main()
