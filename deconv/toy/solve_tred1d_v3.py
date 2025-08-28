#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch solver for grouped (px, py) hits.

Inputs
------
- NPZ file with:
    - "hits_tpc0_batch10"                 : 1D array of charges (hqs)
    - "hits_tpc0_batch10_location"        : 2D array with columns ["px", "py", "pt0", "pt1", ...]
- NPY response kernel file (default: response_44_v2a_100ns.npy)
    * If the array is 3D, we use fr[0, 0, -TAIL:] and scale by SCALE, mirroring your org-file.

Behavior
--------
For each unique (px, py):
  1) Collect group rows and their corresponding hqs.
  2) Build Ae using intervals [pt0, pt1) (end-exclusive by default).
  3) Deduce N from total length T and len(fr) via: N = T - len(fr) + 1.
  4) Build Toeplitz frm, A = Ae @ frm.
  5) Solve min ||A q - hqs||_2^2 + lam ||q||_2^2  (lam >= 0).
  6) (Optional) Compute block-sum x_hat using K blocks (like your example).
  7) Print a compact summary.

Usage
-----
python solve_groups.py --npz path/to/data.npz \
                       --fr path/to/response_44_v2a_100ns.npy \
                       --scale 0.1 --tail 400 \
                       --lam 0.0 --K 4 \
                       --px-col 0 --py-col 1 --pt0-col 2 --pt1-col 3 \
                       --end-exclusive

Notes
-----
- If you want end to be inclusive, pass --end-inclusive instead (or set --end-exclusive off).
- If N <= 0 for a group (too-short windows vs. kernel length), that group is skipped.
"""

import argparse
import numpy as np
import scipy.linalg
import scipy.optimize

import matplotlib.pyplot as plt


# -------------------------
# Loading & basic utilities
# -------------------------
def load_fr(fr_path, scale=0.1, tail=0):
    """
    Load response kernel `fr` similar to your org-file.
    If 3D, uses fr[0, 0, -tail:] * scale.
    If 1D, uses the last `tail` samples (or full length if shorter) and scales.
    """
    if tail != 0:
        raise ValueError
    arr = np.load(fr_path)
    if arr.ndim == 3:
        fr_raw = arr[0, 0, ]
    # elif arr.ndim == 1:
    #     fr_raw = arr[-min(tail, arr.shape[0]):]
    else:
        # Fallback: flatten last axis if needed
        fr_raw = np.ravel(arr)[-min(tail, arr.size):]
    fr = fr_raw.astype(np.float64) * scale
    return fr


def build_frm(fr, N, T=None):
    """
    Toeplitz convolution matrix frm of shape (T, N),
    where T = len(fr) + N - 1.
    """
    if T is None:
        T = len(fr) + N - 1
    c = np.r_[fr, np.zeros(T-len(fr), dtype=np.float64)]
    r = np.r_[fr[0], np.zeros(T-len(fr)-1, dtype=np.float64)]
    frm = scipy.linalg.toeplitz(c, r)   # (T, N)
    return frm


def build_Ae_from_pt(hts_pt0, hts_pt1, T, reset_time=2, end_exclusive=True):
    """
    Build integration/aggregation matrix Ae of shape (M, T),
    marking ones across [pt0, pt1) if end_exclusive else [pt0, pt1+1).
    """
    M = len(hts_pt0) + 1
    Ae = np.zeros((M, T), dtype=np.float64)
    assert hts_pt0[0] >= 0
    Ae[0, 0:hts_pt0[0]] = 1.0  # pre-window zeros
    Ae[1, hts_pt0[0]:hts_pt1[0]] = 1.0  # pre-window zeros
    if end_exclusive:
        for i in range(1, M-1):
            s = int(hts_pt1[i-1]+reset_time)
            e = int(hts_pt1[i])
            s = max(s, 0)
            e = min(e, T)
            assert e < T
            if e > s:
                Ae[i+1, s:e] = 1.0
    else:
        for i in range(1, M):
            s = int(hts_pt1[i-1]+reset_time)
            e = int(hts_pt1[i]) + 1  # inclusive end -> make exclusive
            s = max(s, 0)
            e = min(e, T)
            assert e < T
            if e > s:
                Ae[i+1, s:e] = 1.0
    return Ae


def solve_tikhonov(A, y, lam=0.0):
    """
    Solve min ||A q - y||_2^2 + lam ||q||_2^2.
    Returns q_hat and residual norm ||A q - y||_2.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    M, N = A.shape
    if lam > 0.0:
        A_aug = np.vstack([A, np.sqrt(lam) * np.eye(N, dtype=np.float64)])
        y_aug = np.concatenate([y, np.zeros(N, dtype=np.float64)])
        # non-negative constraint should be added to resolve positive q_hat
        q_hat, *_ = scipy.optimize.nnls(A_aug, y_aug)
        # q_hat, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    else:
        # q_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
        q_hat, *_ = scipy.optimize.nnls(A_aug, y_aug)
    resid = np.linalg.norm(A @ q_hat - y)
    return q_hat, resid


# ---------------------------------
# Block-ops (as in your org-example)
# ---------------------------------
def build_block_ops(N, K):
    """
    Exactly your construction:
      - R: K x N sums over B=N//K slots per block
      - C: N x K expands block values to N
      - W: N x (N-K) null-space completion s.t. R @ W = 0
    """
    assert N % K == 0, "N must be divisible by K"
    B = N // K
    oneB = np.ones((B, 1), dtype=np.float64)
    I_K = np.eye(K, dtype=np.float64)

    R = np.kron(I_K, oneB.T)           # K x N
    C = np.kron(I_K, oneB)             # N x K

    Wb = np.zeros((B, B-1), dtype=np.float64)
    for j in range(B - 1):
        Wb[j, j] = 1.0
        Wb[B - 1, j] = -1.0
    W = np.kron(I_K, Wb)               # N x (N-K)

    # Sanity checks (can be commented out for speed)
    assert np.allclose(R @ C, B * np.eye(K))
    assert np.allclose(R @ W, 0)
    return B, R, C, W


import numpy as np
from scipy.optimize import lsq_linear


def solve_block_ridge_full(A, y, C, W, Proj, lam0, lam1):
    """
    Same algebra as your org-file snippet.
    Returns u, v with x_hat = B * u (block sums).
    """

    A11 = A @ C
    A12 = A @ W
    A21 = np.zeros((C.shape[0], C.shape[1]), dtype=np.float64)
    A22 = np.sqrt(lam0) * W

    rhs1 = y
    rhs2 = np.zeros((W.shape[0], 1))
    M = np.block([[A11, A12],
                  [A21, A22],])
    b = np.vstack([rhs1, rhs2,])

    K = C.shape[1]
    J = W.shape[1]
    # Bounds: u >= 0, v free
    # lb = np.concatenate([np.zeros(K), -np.inf * np.ones(J)])
    # ub = np.concatenate([np.inf * np.ones(K + J)])
    lb = np.concatenate([-np.inf * np.ones(K + J)])
    ub = np.concatenate([np.inf * np.ones(K + J)])

    res = lsq_linear(M, np.squeeze(b), bounds=(lb, ub), method='trf')
    z = res.x.reshape(-1, 1)

    u = z[:K]
    v = z[K:]
    return u


def baseline_identifiable_x(A, R, y):
    A_plus = np.linalg.pinv(A)
    return R @ A_plus @ y


# -------------------------
# Per-group solve function
# -------------------------
def solve_one_group(fr, hqs_group, pt0_group, pt1_group, spacing, lam0=0.0, lam1=0.1, end_exclusive=True):
    """
    Build Ae and frm for a group, solve for q_hat, and (optionally) block x_hat.
    Returns a dict with results and a small summary.
    """
    # Determine total length T from pt1. Assume pt1 is an exclusive end by default.
    post_n = 1  # assume 1 extra spacing after last hit if signal is too small to detect; signal is assumed to be no earlier than first hit.
    pre_n = 1 # assume 1 extra spacing before first hit, which is from suppression of induction

    # truncate time to index

    # FIXME: how to deal with negative?
    start_idx = ((np.min(pt0_group) - 10.431 / 0.16 // 0.05) // spacing - pre_n) * spacing
    end_idx = ((np.max(pt1_group) - 10.431 / 0.16 // 0.05) // spacing + post_n) * spacing
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    N = end_idx - start_idx
    K = N // spacing
    print(N, start_idx, end_idx)
    # print("range of indices", start_idx, end_idx, max_end)
    # Proj = np.eye(N,) - proj @ proj.T
    mask = np.ones((N,), dtype=float)
    Proj = np.diag(1.0 - mask)
    pt0_group = pt0_group - start_idx
    pt1_group = pt1_group - start_idx
    T = N + len(fr)
    Ae  = build_Ae_from_pt(pt0_group, pt1_group, T, reset_time=2, end_exclusive=end_exclusive)  # (M, T)
    frm = build_frm(fr, N, T)                             # (T, N)
    A   = Ae @ frm                                     # (M, N)
    y = np.zeros((len(hqs_group)+1,), dtype=np.float64)
    y[1:] = hqs_group

    thres = 5
    y[0] = thres  # enforce some minimum signal before first hit
    y[1] = y[1] - thres

    # print("A.shape", A.shape, "frm.shape", frm.shape, "Ae.shape", Ae.shape, y.shape)

    # Solve for q
    q_hat, resid = solve_tikhonov(A, y, lam=lam0)

    result = {
        "ok": True,
        "T": T,
        "N": N,
        "M": A.shape[0],
        # "residual_norm": float(resid),
        "q_hat": q_hat,          # length N
        "A" : A,
        "Ae" : Ae,
        "frm" : frm,
    }

    # Optional: block estimate like your example
    if K is not None and K > 0 and (N % K == 0):
        B, R, C, W = build_block_ops(N, K)
        # print('B', B, 'R', R, 'C', C, 'W', W)
        y_vec = y.reshape(-1, 1)
        u = solve_block_ridge_full(A, y_vec, C, W, Proj, lam0=max(lam0, 0.0), lam1=lam1)
        x_hat = (B * u).reshape(-1)
        # x_pinv = baseline_identifiable_x(A, R, y_vec).reshape(-1)
        result.update({
            "K": K,
            "B": B,
            "x_hat": x_hat,       # length K (block sums)
            # "x_pinv": x_pinv,     # baseline identifiable part
            "tstart" : start_idx, # starting time index for this group
            # "residual" : np.linalg.norm(y_vec[:,0] - A @ C @ u - A @ W @ v),
        })
    else:
        if K is not None and (N % K != 0):
            result.update({"block_warning": f"N={N} not divisible by K={K}; skipping block estimate."})

    return result


def condense_effq_by_pxpy(effq, effq_loc, spacing=50, px_col=0, py_col=1, t_col=2, drop_zero_bins=True):
    """
    Group by (px, py), bin time t into [k*spacing, (k+1)*spacing) bins,
    and sum effq weights inside each bin.

    Returns:
      dict keyed by (px, py) -> {
          "bin_left": 1D array of left edges,
          "bin_right": 1D array of right edges,
          "sum": 1D array of accumulated effq per bin,
          "bin_index": 1D array of integer bin indices (for reference)
      }
    """
    effq = np.asarray(effq, dtype=float).reshape(-1)
    effq_loc = np.asarray(effq_loc)

    if effq_loc.ndim != 2 or effq_loc.shape[1] <= max(px_col, py_col, t_col):
        raise ValueError("effq_loc must be a 2D array with at least three columns [px, py, t].")

    if len(effq) != len(effq_loc):
        raise ValueError("effq and effq_loc must have the same length.")

    px = effq_loc[:, px_col]
    py = effq_loc[:, py_col]
    t  = effq_loc[:, t_col]

    # Bin index: k = floor(t / spacing), clamp negatives to 0
    bin_idx_all = np.floor(t / spacing).astype(int)
    bin_idx_all[bin_idx_all < 0] = 0

    # Unique groups by (px, py)
    pairs = np.stack([px, py], axis=1)
    uniq_pairs, inv = np.unique(pairs, axis=0, return_inverse=True)

    result = {}
    for gi, (px_val, py_val) in enumerate(uniq_pairs):
        sel = (inv == gi)
        if not np.any(sel):
            continue

        bins_g = bin_idx_all[sel]
        effq_g = effq[sel]

        nbins = int(np.max(bins_g)) + 1  # cover [0, ..., max_bin]
        sums = np.bincount(bins_g, weights=effq_g, minlength=nbins)

        # Build edges for readability
        bin_left = np.arange(nbins) * spacing
        bin_right = bin_left + spacing

        if drop_zero_bins:
            nz = sums != 0
            bin_left = bin_left[nz]
            bin_right = bin_right[nz]
            sums = sums[nz]
            bins_out = np.arange(nbins)[nz]
        else:
            bins_out = np.arange(nbins)

        result[(px_val, py_val)] = {
            "bin_left": bin_left,
            "bin_right": bin_right,
            "sum": sums,
            "bin_index": bins_out
        }

    return result

# -------------
# Main routine
# -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to NPZ containing hits_tpc0_batch10 and hits_tpc0_batch10_location")
    ap.add_argument("--fr",  default="response_44_v2a_100ns.npy", help="Path to NPY kernel (response)")
    ap.add_argument("--scale", type=float, default=0.05, help="Scale factor for the response kernel")
    ap.add_argument("--tail",  type=int,   default=0, help="Use last `tail` samples of the kernel")
    ap.add_argument("--lam0",   type=float, default=0.0, help="lambda for mean (0 = pure least squares)")
    ap.add_argument("--lam1",   type=float, default=0.0, help="lambda for sparse (0 = pure least squares)")
    ap.add_argument("--K",     type=int,   default=4,   help="Number of blocks for x_hat (like org-file). Requires N %% K == 0.") # sum every 50, then N // 50
    ap.add_argument("--px-col", type=int, default=0)
    ap.add_argument("--py-col", type=int, default=1)
    ap.add_argument("--pt0-col", type=int, default=2, help="Column index for pt0 in location array")
    ap.add_argument("--pt1-col", type=int, default=3, help="Column index for pt1 in location array")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--end-exclusive", action="store_true", default=True, help="Treat pt1 as exclusive end (default).")
    g.add_argument("--end-inclusive", action="store_true", help="Treat pt1 as inclusive end.")

    args = ap.parse_args()

    # Load data
    npz = np.load(args.npz)
    hqs_all = np.asarray(npz["hits_tpc0_batch10"][:,-1], dtype=np.float64).reshape(-1)
    locs    = np.asarray(npz["hits_tpc0_batch10_location"])
    if locs.ndim != 2:
        raise ValueError("hits_tpc0_batch10_location must be a 2D array.")

    # Column extraction
    px_all  = locs[:, args.px_col]
    py_all  = locs[:, args.py_col]
    pt0_all = locs[:, args.pt0_col]
    pt1_all = locs[:, args.pt1_col]

    # Sanity: lengths must match
    if not (len(hqs_all) == len(px_all) == len(py_all) == len(pt0_all) == len(pt1_all)):
        raise ValueError("Mismatched lengths across hqs and location columns.")

    # Load kernel
    fr = load_fr(args.fr, scale=args.scale, tail=args.tail)

    # spacing = 50  # 0.05 ns * 0.16 cm/us * 50 = 0.04 cm
    spacing = 25  # 0.05 ns * 0.16 cm/us * 50 = 0.04 cm
    assert len(fr) % 50 == 0, "length of fr must be dividable by 50 for spacing=50"

    # Group by (px, py)
    pairs = np.stack([px_all, py_all], axis=1)
    uniq_pairs, inverse_idx = np.unique(pairs, axis=0, return_inverse=True)

    end_exclusive = not args.endInclusive if hasattr(args, "endInclusive") else args.end_exclusive
    # (argparse stores --end-inclusive in args.end_inclusive; handle both for safety)
    if hasattr(args, "end_inclusive") and args.end_inclusive:
        end_exclusive = False

    print(f"Loaded: {len(hqs_all)} hits; unique (px, py) groups = {len(uniq_pairs)}")
    print(f"Kernel: len(fr)={len(fr)}, sum(fr)={np.sum(fr):.6g}")
    print(f"Solving with lam={args.lam0}, K={args.K} (block est. only if N divisible by K)")
    print("-" * 80)

    effq = npz["effq_tpc0_batch10"][:,-1]
    effq_loc = npz["effq_tpc0_batch10_location"]  # columns: [px, py, t]
    condensed = condense_effq_by_pxpy(effq, effq_loc, spacing=spacing)
    # condensed = condense_effq_by_pxpy(effq, effq_loc, spacing=1, drop_zero_bins=False)


    hit_pair = {}
    pixel_pair = {}

    # Solve per group
    for gi, (px_val, py_val) in enumerate(uniq_pairs):
        sel = (inverse_idx == gi)
        hqs_g  = hqs_all[sel]
        pt0_g  = pt0_all[sel]
        pt1_g  = pt1_all[sel]

        res = solve_one_group(
            fr,
            hqs_g,
            pt0_g,
            pt1_g,
            spacing = spacing,
            lam0=args.lam0,
            lam1=args.lam1,
            # K=args.K,
            end_exclusive=end_exclusive
        )

        header = f"(px={px_val}, py={py_val})  |  M={len(hqs_g)}"
        print(header)
        if not res["ok"]:
            print(f"  SKIP: {res['reason']}")
            print("-" * 80)
            continue

        # print(f"  T={res['T']}, N={res['N']}, residual ||Aq - y||={res['residual_norm']:.6g}")
        qsum = float(np.sum(res["q_hat"]))
        print(f"  sum(q_hat)={qsum:.6g}")

        R = np.kron(np.eye(res["K"]), np.ones((res["N"]//res["K"], 1)).T)

        if "x_hat" in res:
            xsum = float(np.sum(res["x_hat"]))
            # xpinv_sum = float(np.sum(res["x_pinv"]))
            print(f"  x_hat (K={res['K']}, B={res['B']}): {res['x_hat'].round(6)}")
            print(f"  q_hat {R@res['q_hat']}")
            # print(f"  x_pinv: {res['x_pinv'].round(6)}")
            # print(f"  sum(x_hat)={xsum:.6g}, sum(x_pinv)={xpinv_sum:.6g}")
            print(f"  sum(x_hat)={xsum:.6g}",)
            print(f"  time: {res['tstart']} to {res['tstart'] + res['N'] - 1} (len={res['N']})")
            print(f"  time offset {res['tstart'] + 10.431/0.16//0.05}, offset {10.431/0.16//0.05} ")
            # print(f"  residual ||y - A C u - A W v|| = {res['residual']:.6g}")
        elif "block_warning" in res:
            print(f"  Note: {res['block_warning']}")

        print(f"  effq: {np.sum(condensed.get((px_val, py_val), {'sum': np.array([])})['sum'])}", f"time: {condensed.get((px_val, py_val), {'bin_left': np.array([])})['bin_left']}")
        print(f"  recorded hits: {np.sum(hqs_g)}", hqs_g)

        q = np.r_[5, hqs_g]
        q[1] = q[1] - 5
        # print(res["A"]@np.kron(np.ones(50,), res["x_hat"])/50, hqs_g, q)
        # print(condensed.get((px_val, py_val))['sum'][-len(q)*spacing:].shape, res["A"].shape)
        # print(res["A"]@condensed.get((px_val, py_val))['sum'][-res["A"].shape[1]:], res["A"].shape)
        # plt.plot(res["A"][0])
        # plt.savefig("A0.png")
        print("-" * 80)

        pixel_pair[(px_val, py_val)] = (np.sum(res["x_hat"]), np.sum(res["q_hat"]),
                                        np.sum(condensed.get((px_val, py_val), {'sum': np.array([])})["sum"]),
                                        np.sum(hqs_g))

    for k in condensed.keys():
        if k in pixel_pair.keys():
            continue
        pixel_pair[(px_val, py_val)] = (0, 0, np.sum(condensed[k]['sum']), 0)


    lam0str = str(args.lam0).replace('.', 'p').replace('-', 'm')
    x_hat_vals = [p[0] for p in pixel_pair.values()]
    q_hat_vals = [p[1] for p in pixel_pair.values()]
    effq_vals = [p[2] for p in pixel_pair.values()]
    hitq_vals = [p[3] for p in pixel_pair.values()]
    plt.figure(figsize=(10,6))
    plt.plot(effq_vals, x_hat_vals, 'o')
    plt.plot(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1), '--')
    plt.xlabel("effq per pixel")
    plt.ylabel("x_hat per pixel")
    plt.title("Block sum x_hat vs effq per pixel; lambda=%.1e" % args.lam0)
    plt.savefig("xhat_vs_effq_perpix_lam%s.png" % lam0str)
    plt.figure(figsize=(10,6))
    plt.plot(effq_vals, q_hat_vals, 'o')
    plt.plot(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1), '--')
    plt.xlabel("effq per pixel")
    plt.ylabel("q_hat per pixel")
    plt.title("Total q_hat vs effq per pixel; lambda=%.1e" % args.lam0)
    plt.savefig("qhat_vs_effq_perpix_lam%s.png" % lam0str)
    plt.figure(figsize=(10,6))
    plt.plot(effq_vals, hitq_vals, 'o')
    plt.plot(np.arange(0, 30, 0.1), np.arange(0, 30, 0.1), '--')
    plt.xlabel("effq per pixel")
    plt.ylabel("hitq per pixel")
    plt.savefig("hitq_vs_effq_perpix.png")


if __name__ == "__main__":
    main()
