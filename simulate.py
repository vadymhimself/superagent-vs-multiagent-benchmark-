#!/usr/bin/env python3
"""Reproducible illustrative Monte Carlo comparison, not an Anthropic API emulator."""
import argparse
import csv
import numpy as np

def simulate(seed=20261113, runs=1000):
    rng = np.random.default_rng(seed)
    output = []
    for run in range(runs):
        times = np.sort(rng.uniform(0, 480, 200))
        skew = rng.uniform(0.5, 1.4)
        weights = (np.arange(1, 1001) + 20) ** (-skew)
        weights /= weights.sum()
        clients = rng.choice(1000, 200, p=weights)
        header_last = -1e9
        client_last, client_hist = {}, {}
        shared_last, shared_hist = -1e9, 0
        tok = np.zeros((2, 5))  # input, cache_read, write_5m, write_1h, output
        for t, c in zip(times, clients):
            if t - header_last <= 5:
                tok[:, 1] += 30000
            else:
                tok[:, 2] += 30000
            header_last = t
            tok[:, 0] += 200
            tok[:, 4] += 700
            tail = 1200 + client_hist.get(c, 0)
            if t - client_last.get(c, -1e9) <= 5:
                tok[0, 1] += tail
            else:
                tok[0, 2] += tail
            client_last[c] = t
            client_hist[c] = client_hist.get(c, 0) + 500
            tok[1, 0] += 1200
            if shared_hist:
                if t - shared_last <= 60:
                    tok[1, 1] += shared_hist
                else:
                    tok[1, 3] += shared_hist
            shared_last = t
            shared_hist += 500
        prices = np.array([4, 0.2, 5, 8, 20])
        cost = (tok @ prices) / 1e6
        output.append((run, float(cost[0]), float(cost[1])))
    return output

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=20261113)
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--output", default="results.csv")
    a = p.parse_args()
    data = simulate(a.seed, a.runs)
    with open(a.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "separate_usd", "shared_usd"])
        w.writerows(data)
    x = np.array(data)[:, 1:]
    print("median:", np.median(x, axis=0))
    print("p05:", np.quantile(x, 0.05, axis=0))
    print("p95:", np.quantile(x, 0.95, axis=0))
    print("shared cheaper fraction:", np.mean(x[:, 1] < x[:, 0]))
