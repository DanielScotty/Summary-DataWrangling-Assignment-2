"""
generate_streams.py - M, D and C streams from
Dasu, Krishnan, Venkatasubramanian, Yi: "An Information-Theoretic Approach to
Detecting Changes in Multi-Dimensional Data Streams" (Section 6.2).

Everything needed is configured in the CONFIG block below. Just run:

    python generate_streams.py        (or press Run in VS Code)

Each stream: 5,000,000 2D Gaussian points, split into 100 groups of 50,000.
Parameters are constant within a group and change between groups, giving
5,000,000 / 50,000 - 1 = 99 changes.

  mean  -> M(delta): mu1, mu2 random-walk in [0.2, 0.8] from 0.5; sigma=0.2, rho=0.5
  sigma -> D(delta): sigma1, sigma2 random-walk in [0, 0.4] from 0.2; mu=0.5, rho=0.5
  corr  -> C(delta): rho random-walk in [-1, 1] from 0; mu=0.5, sigma=0.2

For each stream two files are written to OUTPUT_DIR:
  <name>.csv        one record per line, "x,y" (readable by PCA_CD.py)
  <name>_truth.txt  the 99 true change-point indices (50000, 100000, ...)
"""

import os
import numpy as np

# =====================================================================
# CONFIG - edit this block
# =====================================================================
OUTPUT_DIR = "generated_streams"      # created if it does not exist

# (kind, delta) pairs to generate. kind is "mean", "sigma" or "corr".
# These are the nine streams used in Table 2 of the paper (~100 MB each).
# Delete lines to generate fewer files.
STREAMS = [
    ("mean",  0.01), ("mean",  0.02), ("mean",  0.05),   # M(0.01), M(0.02), M(0.05)
    ("sigma", 0.01), ("sigma", 0.02), ("sigma", 0.05),   # D(0.01), D(0.02), D(0.05)
    ("corr",  0.10), ("corr",  0.15), ("corr",  0.20),   # C(0.1),  C(0.15), C(0.2)
]

SEED = 42             # set to None for a different stream on every run
N_POINTS = 5_000_000  # total points per stream
GROUP = 50_000        # points per group (a change happens every GROUP points)

EPS_SIGMA = 0.02      # paper range for sigma is [0, 0.4]; sigma = 0 is degenerate
MAX_RHO = 0.99        # paper range for rho is [-1, 1]; |rho| = 1 is a singular covariance
# =====================================================================

N_GROUPS = N_POINTS // GROUP          # 100 groups -> 99 changes
NAMES = {"mean": "M", "sigma": "D", "corr": "C"}


def random_walk(start, delta, lo, hi, n, rng):
    """Path of length n. Each step is uniform in [-delta,-delta/2] U [delta/2,delta];
    if the new value leaves [lo, hi], another step is drawn."""
    path = [start]
    for _ in range(n - 1):
        while True:
            step = rng.uniform(delta / 2, delta) * rng.choice([-1, 1])
            new = path[-1] + step
            if lo <= new <= hi:
                break
        path.append(new)
    return path


def gauss2d(rng, size, mu1, mu2, s1, s2, rho):
    """Sample `size` points from a 2D normal with correlation rho."""
    z1 = rng.standard_normal(size)
    z2 = rng.standard_normal(size)
    x = mu1 + s1 * z1
    y = mu2 + s2 * (rho * z1 + np.sqrt(1 - rho ** 2) * z2)
    return np.column_stack([x, y])


def group_params(kind, delta, rng):
    """List of (mu1, mu2, sigma1, sigma2, rho), one entry per group."""
    if kind == "mean":
        m1 = random_walk(0.5, delta, 0.2, 0.8, N_GROUPS, rng)
        m2 = random_walk(0.5, delta, 0.2, 0.8, N_GROUPS, rng)
        return [(a, b, 0.2, 0.2, 0.5) for a, b in zip(m1, m2)]
    if kind == "sigma":
        s1 = random_walk(0.2, delta, EPS_SIGMA, 0.4, N_GROUPS, rng)
        s2 = random_walk(0.2, delta, EPS_SIGMA, 0.4, N_GROUPS, rng)
        return [(0.5, 0.5, a, b, 0.5) for a, b in zip(s1, s2)]
    if kind == "corr":
        rho = random_walk(0.0, delta, -MAX_RHO, MAX_RHO, N_GROUPS, rng)
        return [(0.5, 0.5, 0.2, 0.2, r) for r in rho]
    raise ValueError(f"Unknown stream kind: {kind!r} (use 'mean', 'sigma' or 'corr')")


def generate(kind, delta, rng):
    name = f"{NAMES[kind]}_{delta:g}"
    data_path = os.path.join(OUTPUT_DIR, f"{name}.csv")
    truth_path = os.path.join(OUTPUT_DIR, f"{name}_truth.txt")

    params = group_params(kind, delta, rng)
    changes = []
    with open(data_path, "w") as f:
        for g, (m1, m2, s1, s2, r) in enumerate(params):
            if g > 0:
                changes.append(g * GROUP)
            np.savetxt(f, gauss2d(rng, GROUP, m1, m2, s1, s2, r),
                       fmt="%.6f", delimiter=",")

    with open(truth_path, "w") as t:
        t.write("\n".join(map(str, changes)) + "\n")
    print(f"{name}: {N_POINTS} points, {len(changes)} true changes -> {data_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for kind, delta in STREAMS:
        generate(kind, delta, rng)
    print("Done.")


if __name__ == "__main__":
    main()