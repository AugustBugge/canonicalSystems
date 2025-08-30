#!/usr/bin/env python3
"""
Check 3-coin systems with nondecreasing gaps for greedy optimality.

We enumerate coin triples (a < b < c) within {1..n} such that:
    (b - a) <= (c - b)   (i.e., nondecreasing consecutive gaps)

Options:
- Require coin 1 (default on): a = 1; otherwise a can vary.
- Test greedy-vs-optimal for all reachable amounts in 1..max_amount.
- Output CSVs for systems where greedy succeeds vs fails (with first counterexample).

Usage examples:
  python check_3coin_nondecreasing.py --n 50 --max-amount 200
  python check_3coin_nondecreasing.py --n 40 --max-amount 150 --no-require-1
  python check_3coin_nondecreasing.py --n 60 --max-amount 300 --progress 100000
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
        best = dp[x]
        for c in coins_asc:
            if c <= x:
                cand = dp[x - c] + 1
                if cand < best:
                    best = cand
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
        if dp[a] >= INF:   # unreachable amount
            continue
        g = greedy_num_coins(a, coins_desc)
        if g != dp[a]:
            return (False, a, g if g is not None else -1, dp[a])
    return (True, None, None, None)


# ---------- Generate 3-coin systems with nondecreasing gaps ----------

def generate_3coin_nondecreasing(n: int, require_1: bool = True) -> Iterable[List[int]]:
    """
    Yield all triples (a < b < c) within {1..n} with (b-a) <= (c-b).
    If require_1: a = 1; else a ranges.
    """
    if require_1:
        a = 1
        for b in range(a + 1, n):
            d1 = b - a
            # Need c >= b + d1 and c <= n
            c_start = b + d1
            if c_start > n:
                continue
            for c in range(c_start, n + 1):
                yield [a, b, c]
    else:
        for a in range(1, n - 1):
            for b in range(a + 1, n):
                d1 = b - a
                c_start = b + d1
                if c_start > n:
                    continue
                for c in range(c_start, n + 1):
                    yield [a, b, c]


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Check greedy optimality for 3-coin systems with nondecreasing gaps.")
    ap.add_argument("--n", type=int, required=True, help="Max coin denomination; coins lie in {1..n}.")
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

    successes = []
    failures = []

    print(f"Generating 3-coin systems in {{1..{n}}} with nondecreasing gaps"
          f"{' and requiring coin 1' if require_1 else ''}...")
    count = 0
    for idx, coins in enumerate(generate_3coin_nondecreasing(n, require_1=require_1), start=1):
        count += 1
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
    print("\n=== Summary ===")
    print(f"Total 3-coin systems (nondecreasing gaps): {count}")
    print(f"Greedy OK:                               {len(successes)}")
    print(f"Greedy fails:                            {len(failures)}")
    if failures:
        show = min(5, len(failures))
        print(f"Examples of failures (showing {show}):")
        for (coins, bad_amt, g_bad, opt_bad) in failures[:show]:
            print(f"  coins={coins}, first bad amount={bad_amt}, greedy={g_bad}, optimal={opt_bad}")
    print(f"Wrote successes to: {args.successes_out}")
    print(f"Wrote failures to:  {args.failures_out}")


if __name__ == "__main__":
    main()
