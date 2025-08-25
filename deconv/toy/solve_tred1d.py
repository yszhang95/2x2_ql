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
        # fr_raw = arr[0, 0, -tail:]
        fr_raw = arr[0, 0, ]
    # elif arr.ndim == 1:
    #     fr_raw = arr[-min(tail, arr.shape[0]):]
    else:
        # Fallback: flatten last axis if needed
        fr_raw = np.ravel(arr)[-min(tail, arr.size):]
    fr = fr_raw.astype(np.float64) * scale
    return fr


def build_frm(fr, N):
    """
    Toeplitz convolution matrix frm of shape (T, N),
    where T = len(fr) + N - 1.
    """
    T = len(fr) + N - 1
    c = np.r_[fr, np.zeros(N - 1, dtype=np.float64)]
    r = np.r_[fr[0], np.zeros(N - 1, dtype=np.float64)]
    frm = scipy.linalg.toeplitz(c, r)   # (T, N)
    return frm


def build_Ae_from_pt(hts_pt0, hts_pt1, T, reset_time=2, end_exclusive=True):
    """
    Build integration/aggregation matrix Ae of shape (M, T),
    marking ones across [pt0, pt1) if end_exclusive else [pt0, pt1+1).
    """
    M = len(hts_pt0) + 1
    Ae = np.zeros((M, T), dtype=np.float64)
    assert hts_pt0[0] > 0
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
        q_hat, *_ = np.linalg.lstsq(A_aug, y_aug, rcond=None)
    else:
        q_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
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


def solve_block_ridge_full(A, y, C, W, lam):
    """
    Same algebra as your org-file snippet.
    Returns u, v with x_hat = B * u (block sums).
    """
    CtAt = C.T @ A.T
    WtAt = W.T @ A.T

    A11 = CtAt @ A @ C
    A12 = CtAt @ A @ W
    A21 = A12.T
    A22 = WtAt @ A @ W + lam * (W.T @ W)

    rhs1 = CtAt @ y
    rhs2 = WtAt @ y
    KKT = np.block([[A11, A12],
                    [A21, A22]])
    rhs = np.vstack([rhs1, rhs2])

    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    u = sol[:C.shape[1]]
    v = sol[C.shape[1]:]
    return u, v


def baseline_identifiable_x(A, R, y):
    A_plus = np.linalg.pinv(A)
    return R @ A_plus @ y


# -------------------------
# Per-group solve function
# -------------------------
def solve_one_group(fr, hqs_group, pt0_group, pt1_group, spacing, lam=0.0, K=None, end_exclusive=True):
    """
    Build Ae and frm for a group, solve for q_hat, and (optionally) block x_hat.
    Returns a dict with results and a small summary.
    """
    # Determine total length T from pt1. Assume pt1 is an exclusive end by default.
    post_n = 1  # assume 1 extra spacing after last hit if signal is too small to detect; signal is assumed to be no earlier than first hit.
    pre_n = 1 # assume 1 extra spacing before first hit, which is from suppression of induction
    max_end = int(np.max(pt1_group)) // spacing * spacing + spacing * post_n + spacing
    min_start = int(np.min(pt0_group)) // spacing * spacing - spacing * pre_n
    # T = max_end if end_exclusive else (max_end + 1)
    # T = max(T, 0)

    # Deduce N from T and len(fr)
    # N = T - len(fr) + 1
    # FIXME:
    N = max_end - min_start  # ensure N is multiple of spacing
    # print(max_end, min_start, N, np.max(pt1_group), np.min(pt0_group))
    K = N // spacing
    T = N + len(fr) - 1  # adjust T accordingly
    assert N % spacing == 0, "N must be divisible by spacing"
    assert K >= len(hqs_group), f"K must be greater than 1 + number of hits in group, K={K}, nhits = {len(hqs_group)}"
    if N <= 0:
        return {
            "ok": False,
            "reason": f"N={N} <= 0 (T={T}, len(fr)={len(fr)}). Likely windows too short for this kernel."
        }

    # truncate time to index
    pt0_group = pt0_group - min_start
    pt1_group = pt1_group - min_start

    # Build matrices
    frm = build_frm(fr, N)                             # (T, N)
    Ae  = build_Ae_from_pt(pt0_group, pt1_group, T, reset_time=2, end_exclusive=end_exclusive)  # (M, T)
    A   = Ae @ frm                                     # (M, N)
    # build y
    y = np.zeros((len(hqs_group)+1,), dtype=np.float64)
    y[1:] = hqs_group

    thres = 5
    y[0] = thres  # enforce some minimum signal before first hit
    y[1] = y[1] - thres

    # print("A.shape", A.shape, "frm.shape", frm.shape, "Ae.shape", Ae.shape, y.shape)

    # Solve for q
    q_hat, resid = solve_tikhonov(A, y, lam=lam)

    result = {
        "ok": True,
        "T": T,
        "N": N,
        "M": A.shape[0],
        "residual_norm": float(resid),
        "q_hat": q_hat,          # length N
    }

    # Optional: block estimate like your example
    if K is not None and K > 0 and (N % K == 0):
        B, R, C, W = build_block_ops(N, K)
        y_vec = y.reshape(-1, 1)
        u, v = solve_block_ridge_full(A, y_vec, C, W, lam=max(lam, 0.0))
        x_hat = (B * u).reshape(-1)
        x_pinv = baseline_identifiable_x(A, R, y_vec).reshape(-1)
        result.update({
            "K": K,
            "B": B,
            "x_hat": x_hat,       # length K (block sums)
            "x_pinv": x_pinv,     # baseline identifiable part
            "tstart" : min_start, # starting time index for this group
        })
    else:
        if K is not None and (N % K != 0):
            result.update({"block_warning": f"N={N} not divisible by K={K}; skipping block estimate."})

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
    ap.add_argument("--lam",   type=float, default=0.0, help="Tikhonov lambda (0 = pure least squares)")
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

    spacing = 50  # 0.05 ns * 0.16 cm/us * 50 = 0.04 cm
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
    print(f"Solving with lam={args.lam}, K={args.K} (block est. only if N divisible by K)")
    print("-" * 80)

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
            lam=args.lam,
            # K=args.K,
            end_exclusive=end_exclusive
        )

        header = f"(px={px_val}, py={py_val})  |  M={len(hqs_g)}"
        print(header)
        if not res["ok"]:
            print(f"  SKIP: {res['reason']}")
            print("-" * 80)
            continue

        print(f"  T={res['T']}, N={res['N']}, residual ||Aq - y||={res['residual_norm']:.6g}")
        qsum = float(np.sum(res["q_hat"]))
        print(f"  sum(q_hat)={qsum:.6g}")

        if "x_hat" in res:
            xsum = float(np.sum(res["x_hat"]))
            xpinv_sum = float(np.sum(res["x_pinv"]))
            print(f"  x_hat (K={res['K']}, B={res['B']}): {res['x_hat'].round(6)}")
            print(f"  sum(x_hat)={xsum:.6g}, sum(x_pinv)={xpinv_sum:.6g}")
        elif "block_warning" in res:
            print(f"  Note: {res['block_warning']}")

        print("-" * 80)


if __name__ == "__main__":
    main()
