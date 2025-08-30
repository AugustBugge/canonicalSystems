#!/usr/bin/env python3
"""
Check whether the greedy coin-change algorithm is optimal for a given currency system.

Given a list of coin denominations, this script:
  - Runs greedy and optimal (DP/brute-force) coin change for all amounts 1..n
  - Records for each amount whether greedy matches the true optimum
  - Saves results to a CSV
  - Prints a short summary

Usage examples:
  python check_greedy_coin_change.py --coins 1,5,10,25 --n 100
  python check_greedy_coin_change.py --coins 1,3,4 --n 100 --outfile results.csv
"""

from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ResultRow:
    amount: int
    reachable: bool
    greedy_num_coins: Optional[int]
    optimal_num_coins: Optional[int]
    greedy_is_optimal: Optional[bool]


def parse_coins(s: str) -> List[int]:
    try:
        coins = sorted({int(x) for x in s.split(",") if x.strip() != ""})
    except ValueError:
        raise argparse.ArgumentTypeError("Coins must be integers separated by commas, e.g. 1,3,4,10")

    if any(c <= 0 for c in coins):
        raise argparse.ArgumentTypeError("All coin denominations must be positive integers.")
    return coins


def greedy_change(amount: int, coins_desc: List[int]) -> Optional[int]:
    """
    Greedy: repeatedly take the largest coin <= remaining amount.
    Returns the number of coins used, or None if it cannot make exact change.
    """
    remaining = amount
    count = 0
    for c in coins_desc:
        if remaining <= 0:
            break
        take = remaining // c
        if take > 0:
            count += take
            remaining -= take * c
    return count if remaining == 0 else None


def optimal_change_min_coins(amount: int, coins: List[int]) -> Optional[int]:
    """
    Brute-force via dynamic programming:
    dp[x] = minimum number of coins to make x, or inf if impossible.
    Returns None if unreachable.
    """
    INF = 10**9
    dp = [INF] * (amount + 1)
    dp[0] = 0
    for x in range(1, amount + 1):
        best = INF
        for c in coins:
            if c <= x and dp[x - c] + 1 < best:
                best = dp[x - c] + 1
        dp[x] = best
    return None if dp[amount] >= INF else dp[amount]


def analyze(coins: List[int], n: int) -> Tuple[List[ResultRow], bool]:
    coins_asc = sorted(coins)
    coins_desc = sorted(coins, reverse=True)
    rows: List[ResultRow] = []

    all_reachable_cases_greedy_ok = True

    for amt in range(1, n + 1):
        opt = optimal_change_min_coins(amt, coins_asc)
        gr = greedy_change(amt, coins_desc)
        reachable = opt is not None
        if reachable:
            greedy_ok = (gr == opt)
            all_reachable_cases_greedy_ok &= greedy_ok
        else:
            greedy_ok = None  # N/A if unreachable

        rows.append(
            ResultRow(
                amount=amt,
                reachable=reachable,
                greedy_num_coins=gr,
                optimal_num_coins=opt,
                greedy_is_optimal=greedy_ok,
            )
        )

    return rows, all_reachable_cases_greedy_ok


def save_csv(rows: List[ResultRow], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["amount", "reachable", "greedy_num_coins", "optimal_num_coins", "greedy_is_optimal"])
        for r in rows:
            writer.writerow([
                r.amount,
                r.reachable,
                "" if r.greedy_num_coins is None else r.greedy_num_coins,
                "" if r.optimal_num_coins is None else r.optimal_num_coins,
                "" if r.greedy_is_optimal is None else r.greedy_is_optimal,
            ])


def main():
    parser = argparse.ArgumentParser(description="Check greedy optimality for a coin system.")
    parser.add_argument("--coins", required=True, type=parse_coins,
                        help="Comma-separated coin denominations (e.g., 1,5,10,25)")
    parser.add_argument("--n", type=int, default=100, help="Test amounts 1..n (default: 100)")
    parser.add_argument("--outfile", type=str, default="greedy_check_results.csv",
                        help="CSV output path (default: greedy_check_results.csv)")
    args = parser.parse_args()

    coins = args.coins
    n = args.n

    # Basic sanity notes for the user
    if 1 not in coins:
        print("Note: 1 is not in the coin system; some amounts may be unreachable.")

    rows, all_ok = analyze(coins, n)
    save_csv(rows, args.outfile)

    # Summary
    reachable_counts = sum(1 for r in rows if r.reachable)
    mismatches = [r.amount for r in rows if r.reachable and r.greedy_is_optimal is False]
    print(f"Analyzed amounts 1..{n} for coins {coins}.")
    print(f"Reachable amounts: {reachable_counts}/{n}")
    if mismatches:
        print(f"Greedy failed on {len(mismatches)} reachable amount(s): {mismatches}")
        print("=> The currency system is NOT greedy-optimal over the tested range.")
    else:
        if reachable_counts == 0:
            print("No reachable amounts in the tested range.")
        else:
            print("Greedy matched optimal for all reachable amounts in the tested range.")
    print(f"Results saved to: {args.outfile}")


if __name__ == "__main__":
    main()
