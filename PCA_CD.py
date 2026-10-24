"""
cd.py  –  Change Detection via PCA + Histogram Divergence + Page-Hinkley Test
Converted from CD.cpp / CD.h / ReportResult.h (original author: Hakim, Feb 2015)

Usage:
    python cd.py <data_file> <window_size> <threshold_factor> <output_change_file>
                 <data_dimension> <delta> <divergence_metric (1 or 2)>

Divergence metrics:
    1 – Intersection (histogram overlap)
    2 – Symmetric KL divergence
"""

import sys
import math
import os
import time
from typing import List, Tuple
from collections import deque
import numpy as np
import re

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
MAX_DIM = 100
CUM_EIG = 0.995          # cumulative eigenvalue threshold for PCA truncation
BIN_WIDTH_FACTOR = 2.0
CHANGE_INTERVAL = 50_000  # used only in results evaluation


# ─────────────────────────────────────────────────────────────
# Statistics helpers
# ─────────────────────────────────────────────────────────────

def std_from_sums(tot_sum: float, tot_sqsum: float, n: int) -> float:
    """Population-corrected standard deviation from running sums."""
    if n <= 1:
        return 0.0
    variance = (tot_sqsum - tot_sum ** 2 / n) / (n - 1)
    if variance < 0 or math.isnan(variance) or math.isinf(variance):
        return 0.0
    return math.sqrt(variance)


def bandwidth_nr(tot_sum: float, tot_sqsum: float, n: int) -> Tuple[float, float]:
    """Normal-reference bandwidth (Silverman's rule) + sigma."""
    sigma = std_from_sums(tot_sum, tot_sqsum, n)
    if sigma == 0.0:
        return 1.0, sigma
    return 1.06 * sigma * n ** (-0.2), sigma


# ─────────────────────────────────────────────────────────────
# Histogram helpers
# ─────────────────────────────────────────────────────────────

def compute_score(h_bins: List[dict], idx: int, ref_num_pnts: int, dm: int) -> float:
    """Score contribution of a single histogram bin."""
    r_epdf = h_bins[idx]["count"][0]
    t_epdf = h_bins[idx]["count"][1]
    if dm == 1:
        return min(r_epdf, t_epdf)
    # dm == 2: symmetric KL
    r_norm = r_epdf / ref_num_pnts if ref_num_pnts else 0.0
    t_norm = t_epdf / ref_num_pnts if ref_num_pnts else 0.0
    s1 = s2 = 0.0
    if r_norm > 0.0 and t_norm > 0.0:
        s1 = t_norm * math.log(t_norm / r_norm)
        s2 = r_norm * math.log(r_norm / t_norm)
    return max(s1, s2)


def _bin_index(h_bins: List[dict], point: float, lo: float) -> int:
    """Return the histogram bin index for *point* (linear scan, matches C++)."""
    for i, b in enumerate(h_bins):
        if b["lo"] <= point < b["hi"]:
            return i
    if point == lo:
        return 0
    return -1


def create_hist(
    data: List[float],
    lo: float,
    hi: float,
    n_bins: int,
    channel: int,          # 0 = reference, 1 = test
    h_bins: List[dict],    # mutated in-place when channel == 1
    ) -> List[dict]:
    """
    Build (channel 0) or fill (channel 1) a histogram.
    Returns the bin list.
    """
    if channel == 0:
        h_bins.clear()
        for k in range(n_bins):
            h_bins.append({
                "lo": lo + k * (hi - lo) / n_bins,
                "hi": lo + (k + 1) * (hi - lo) / n_bins,
                "count": [0.0, 0.0],
            })
    for val in data:
        idx = _bin_index(h_bins, val, lo)
        if 0 <= idx < len(h_bins):
            h_bins[idx]["count"][channel] += 1
    return h_bins


def update_hist(
    h_bins: List[dict],
    new_point: float,
    rem_point: float,
    lo: float,
    channel_score: float,
    ref_num_pnts: int,
    dm: int,
    ) -> float:
    """
    Sliding-window histogram update: add *new_point*, remove *rem_point*.
    Returns updated cumulative score.
    """
    score = channel_score

    idx = _bin_index(h_bins, new_point, lo)
    if 0 <= idx < len(h_bins):
        score -= compute_score(h_bins, idx, ref_num_pnts, dm)
        h_bins[idx]["count"][1] += 1
        score += compute_score(h_bins, idx, ref_num_pnts, dm)

    idx = _bin_index(h_bins, rem_point, lo)
    if 0 <= idx < len(h_bins):
        score -= compute_score(h_bins, idx, ref_num_pnts, dm)
        h_bins[idx]["count"][1] -= 1
        score += compute_score(h_bins, idx, ref_num_pnts, dm)

    return score


def compare_dist(h_bins: List[dict], ref_num_pnts: int, dm: int) -> float:
    return sum(compute_score(h_bins, i, ref_num_pnts, dm) for i in range(len(h_bins)))


# ─────────────────────────────────────────────────────────────
# PCA helper (replaces stats::pca)
# ─────────────────────────────────────────────────────────────

class IncrementalPCA:
    """Thin wrapper around NumPy SVD-based PCA (batch, not truly incremental)."""

    def __init__(self, dim: int):
        self.dim = dim
        self._records: List[List[float]] = []
        self.eigenvalues: List[float] = []
        self.eigenvectors: List[List[float]] = []  # shape: [component][feature]
        self.principals: List[List[float]] = []    # shape: [component][sample]

    def add_record(self, record: List[float]) -> None:
        self._records.append(list(record))

    def solve(self) -> None:
        X = np.array(self._records, dtype=float)      # (n_samples, dim)
        X_c = X - X.mean(axis=0)
        cov = np.cov(X_c, rowvar=False)               # (dim, dim)
        vals, vecs = np.linalg.eigh(cov)              # ascending order
        # Flip to descending
        vals = vals[::-1]
        vecs = vecs[:, ::-1]
        total = vals.sum() if vals.sum() > 0 else 1.0
        self.eigenvalues = (vals / total).tolist()
        self.eigenvectors = [vecs[:, i].tolist() for i in range(self.dim)]
        # Principal components: project mean-centred data
        self.principals = [(X_c @ vecs[:, i]).tolist() for i in range(self.dim)]

    def get_eigenvalue(self, i: int) -> float:
        return self.eigenvalues[i] if i < len(self.eigenvalues) else 0.0

    def get_eigenvector(self, i: int) -> List[float]:
        return self.eigenvectors[i] if i < len(self.eigenvectors) else [0.0] * self.dim

    def get_principal(self, i: int) -> List[float]:
        return self.principals[i] if i < len(self.principals) else []


# ─────────────────────────────────────────────────────────────
# Results reporting (ReportResult.h)
# ─────────────────────────────────────────────────────────────

def report_results(change_file: str, num_samples: int, W: int, elapsed: float) -> None:
    correct = wrong = late = 0
    start = 0
    try:
        with open(change_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counter = int(line)
                if counter % CHANGE_INTERVAL < 2 * W:
                    correct += 1
                elif counter // CHANGE_INTERVAL > start // CHANGE_INTERVAL:
                    late += 1
                else:
                    wrong += 1
                start = counter
    except FileNotFoundError:
        print("Cannot open change points file.")
        return

    print(f"NUMBER of SAMPLES = {num_samples}\nCHANGE INTERVAL = {CHANGE_INTERVAL}")
    missed = num_samples // CHANGE_INTERVAL - 1 - correct - late
    print(
        f"Correct = {correct}\nLate = {late}\nWrong = {wrong}\n"
        f"Missed = {missed}\nRun time = {elapsed:.4f}"
    )


# ─────────────────────────────────────────────────────────────
# PCA fit helper (used twice: initial + after each change)
# ─────────────────────────────────────────────────────────────

def fit_pca_model(
    records: List[List[float]],
    dim: int,
) -> Tuple[
    "IncrementalPCA",
    int,                       # new_dim
    List[List[float]],         # eig_vectors[feature][component]
    List[List[float]],         # MD  – per-dim sliding windows
    List[float],               # min_val
    List[float],               # max_val
    List[float],               # R_sum
    List[float],               # R_sqsum
    List[float],               # R_mean
]:
    mypca = IncrementalPCA(dim)
    R_mean = [0.0] * dim
    for rec in records:
        for i, v in enumerate(rec):
            R_mean[i] += v
        mypca.add_record(rec)
    n = len(records)
    R_mean = [s / n for s in R_mean]

    mypca.solve()

    cum = 0.0
    new_dim = dim
    for i in range(dim):
        cum += mypca.get_eigenvalue(i)
        if cum > CUM_EIG:
            new_dim = i + 1
            break

    eig_vectors = [[0.0] * new_dim for _ in range(dim)]
    MD: List[List[float]] = [[] for _ in range(new_dim)]
    min_val = [0.0] * new_dim
    max_val = [0.0] * new_dim
    R_sum = [0.0] * new_dim
    R_sqsum = [0.0] * new_dim

    for i in range(new_dim):
        ev = mypca.get_eigenvector(i)
        for m in range(dim):
            eig_vectors[m][i] = ev[m]
        pc = mypca.get_principal(i)
        for k, dp in enumerate(pc):
            MD[i].append(dp)
            if k == 0:
                min_val[i] = max_val[i] = dp
            else:
                min_val[i] = min(min_val[i], dp)
                max_val[i] = max(max_val[i], dp)
            R_sum[i] += dp
            R_sqsum[i] += dp * dp

    return mypca, new_dim, eig_vectors, MD, min_val, max_val, R_sum, R_sqsum, R_mean


def project_record(record: List[float], R_mean: List[float],
                   eig_vectors: List[List[float]], dim: int, new_dim: int) -> List[float]:
    """Mean-centre then project onto *new_dim* principal components."""
    centred = [record[k] - R_mean[k] for k in range(dim)]
    projected = [
        sum(centred[k] * eig_vectors[k][m] for k in range(dim))
        for m in range(new_dim)
    ]
    return projected


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    sys.argv = [
        "PCA_CD.py",
        r"C:\Users\Daniel\OneDrive\Documenten\UU\Data_Wrangling\Assignments\Assignment 2\generated_streams",       # data_file
        "10000",            # window_size
        "500",            # threshold_factor
        r"C:\Users\Daniel\OneDrive\Documenten\UU\Data_Wrangling\Assignments\Assignment 2\changes.txt",    # output_change_file
        "2",              # data_dimension
        "0.005",          # delta
        "1",              # divergence_metric
    ]

    
    win_size = int(sys.argv[2])
    f_factor = float(sys.argv[3])
    out_file = sys.argv[4]
    dim = int(sys.argv[5])
    delta = float(sys.argv[6])
    div_metric = int(sys.argv[7])
    
    for file in os.listdir(sys.argv[1]):
        if not file.endswith(".csv"):
            continue
        else:
            data_file = os.path.join(sys.argv[1], file)

        for div_metric in (1, 2):
            print(f"Processing file: {data_file}, divergence metric: {div_metric}")

            # ── open files ──────────────────────────────────────────
            try:
                infs = open(data_file, "r")
            except OSError:
                print("Error opening data file.")
                sys.exit(3)

            try:
                chfs = open(out_file, "w")
            except OSError:
                print("Error opening output change file.")
                infs.close()
                sys.exit(3)

            def read_record() -> List[float] | None:
                """Read one DIM-valued record from infs. Returns None on EOF."""
                vals = []
                while len(vals) < dim:
                    token = infs.readline()
                    # print(token)
                    if token == "":
                        return None  # EOF
                    for v in re.split(',| |\t', token):
                        # print(v)
                        try:
                            vals.append(float(v))
                        except ValueError:
                            pass
                        if len(vals) == dim:
                            break
                # print (vals)
                return vals if len(vals) == dim else None

            # ── start timer ─────────────────────────────────────────
            t_begin = time.perf_counter()
            data_length = 0

            # ── read initial reference window ───────────────────────
            ref_records: List[List[float]] = []
            while len(ref_records) < win_size:
                rec = read_record() 
                # break       
                if rec is None:
                    break
                ref_records.append(rec)
                data_length += 1

            R_tot_num_pnts = len(ref_records)
            if R_tot_num_pnts == 0:
                print("No data read.")
                infs.close(); chfs.close(); sys.exit(3)

            _, new_dim, eig_vectors, MD_ref, min_val, max_val, R_sum, R_sqsum, R_mean = \
                fit_pca_model(ref_records, dim)

            # Build reference histograms
            Hist: List[List[dict]] = [[] for _ in range(new_dim)]
            num_resamp: List[int] = [0] * new_dim
            for i in range(new_dim):
                bw, _ = bandwidth_nr(R_sum[i], R_sqsum[i], min(data_length, win_size))
                num_resamp[i] = max(1, int(BIN_WIDTH_FACTOR * (max_val[i] - min_val[i]) / bw))
                create_hist(MD_ref[i], min_val[i], max_val[i], num_resamp[i], 0, Hist[i])

            # Sliding test window (deque per component)
            MD: List[deque] = [deque() for _ in range(new_dim)]

            # Scores per component
            score_vec = [0.0] * new_dim
            T_tot_num_pnts = 0

            # Page-Hinkley state
            p_mean_current = 0.0
            m_current = 0.0
            M_current = 0.0
            PH_current = 0.0
            sc_count = 0.0
            start_point = 0.0

            # ── main loop ───────────────────────────────────────────
            while True:
                # Fill test window if not yet full
                if T_tot_num_pnts < win_size:
                    while T_tot_num_pnts < win_size:
                        rec = read_record()
                        if rec is None:
                            break
                        proj = project_record(rec, R_mean, eig_vectors, dim, new_dim)
                        for k in range(new_dim):
                            MD[k].append(proj[k])
                        data_length += 1
                        T_tot_num_pnts += 1

                    if len(MD[0]) < win_size:
                        break  # not enough data

                    # Initial scores for this test window
                    for i in range(new_dim):
                        create_hist(list(MD[i]), min_val[i], max_val[i], num_resamp[i], 1, Hist[i])
                        score_vec[i] = compare_dist(Hist[i], R_tot_num_pnts, div_metric)

                # ── compute MaxScore across components ──────────────
                max_score = None
                for k in range(new_dim):
                    if div_metric == 1:
                        fs = 1.0 - score_vec[k] / R_tot_num_pnts
                    else:
                        fs = score_vec[k]
                    max_score = fs if max_score is None else max(max_score, fs)

                # ── Page-Hinkley update ──────────────────────────────
                p_mean_current = (
                    p_mean_current * (sc_count - start_point) + max_score
                ) / (sc_count - start_point + 1.0)
                m_current = m_current + p_mean_current - max_score + delta
                if abs(m_current) > M_current:
                    M_current = abs(m_current)
                PH_current = M_current - m_current
                sc_count += 1
                lam = f_factor * p_mean_current

                # ── change detected ─────────────────────────────────
                if PH_current > lam:
                    chfs.write(f"{data_length}\n")
                    start_point = sc_count
                    p_mean_current = m_current = M_current = PH_current = 0.0

                    # Re-read reference window
                    ref_records = []
                    while len(ref_records) < win_size:
                        rec = read_record()
                        if rec is None:
                            break
                        ref_records.append(rec)
                        data_length += 1

                    if len(ref_records) < win_size:
                        break

                    R_tot_num_pnts = len(ref_records)
                    _, new_dim, eig_vectors, MD_ref, min_val, max_val, R_sum, R_sqsum, R_mean = \
                        fit_pca_model(ref_records, dim)

                    Hist = [[] for _ in range(new_dim)]
                    num_resamp = [0] * new_dim
                    for i in range(new_dim):
                        bw, _ = bandwidth_nr(R_sum[i], R_sqsum[i], min(data_length, win_size))
                        num_resamp[i] = max(1, int(BIN_WIDTH_FACTOR * (max_val[i] - min_val[i]) / bw))
                        create_hist(MD_ref[i], min_val[i], max_val[i], num_resamp[i], 0, Hist[i])

                    # Re-init test window
                    MD = [deque() for _ in range(new_dim)]
                    score_vec = [0.0] * new_dim
                    T_tot_num_pnts = 0
                    continue

                # ── slide the test window by one point ──────────────
                rec = read_record()
                if rec is None:
                    break
                proj = project_record(rec, R_mean, eig_vectors, dim, new_dim)
                for k in range(new_dim):
                    dp = proj[k]
                    rem = MD[k].popleft()
                    score_vec[k] = update_hist(
                        Hist[k], dp, rem, min_val[k], score_vec[k], R_tot_num_pnts, div_metric,
                    )
                    MD[k].append(dp)
                data_length += 1

            # ── finish ───────────────────────────────────────────────
            elapsed = time.perf_counter() - t_begin
            chfs.close()
            infs.close()
            report_results(out_file, data_length, win_size, elapsed)

if __name__ == "__main__":
    main()