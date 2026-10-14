"""Benchmark 2: banded low-rank polynomial matrix, all structures combined.

    min_{x in [-1, 1]^n}  lambda_min(F(x)),   F(x) = sum_{l=1}^r A_l prod_i f_{l,i}(x_i),

with tridiagonal positive semidefinite m x m matrices A_l and univariate
quadratics f_{l,i} in the Bernstein basis of [-1, 1] with coefficients
b_0 = 1, b_1, b_2 in [1 + 1/n, 1 + 2/n] (the well-conditioned instances of
Balada Gaggioli, Henrion & Korda, Sec. 4.2, with matrix weights). Each
factor is >= 1 with equality only at x_i = -1, and F is monotone in every
product since A_l >= 0, so the minimum is lambda_min(sum_l A_l), attained at
x = (-1, ..., -1). F is tridiagonal, of degree 2n and dense in the
variables (every entry depends on every x_i).

Methods (one run per call; prints one JSON line):

  dense      Scherer-Hol relaxation of minimal order n, one dense PSD block
  zf         ... with the chordal (row-clique) decomposition (Zheng-Fantuzzi)
  ts1        ... with term sparsity, first step of TS="MD" (Miller-Wang-Guo)
  liftdense  SL-chord lifting + correlative sparsity, dense m x m certificate
  lift       SL-chord lifting + correlative + matrix sparsity (this paper)
  push       SL-push lifting, matrix sparsity in the last stage (this paper)

Usage: python banded.py N M METHOD [ORDER] [SOLVER]
"""
import sys

import numpy as np

from common import emit, psp, sdp_stats, solver_options, timed


def instance(n, m, r=2, d=2, seed=0):
    rng = np.random.RandomState(seed)
    factors = []
    for _ in range(n):
        F = rng.uniform(1 + 1.0 / n, 1 + 2.0 / n, size=(d + 1, r))
        F[0, :] = 1.0
        factors.append(F)
    mats = []
    for _ in range(r):
        B = np.diag(rng.uniform(0.5, 1.5, m)) + np.diag(rng.uniform(-1, 1, m - 1), 1)
        mats.append(B @ B.T)  # tridiagonal and PSD
    return factors, mats


def bernstein(x, coeffs):
    s = (x + 1) * 0.5
    d = len(coeffs) - 1
    out = psp.Polynomial.zero(x.nvars)
    for j, c in enumerate(coeffs):
        binom = [1, 1] if d == 1 else [1, 2, 1] if d == 2 else None
        out = out + (s ** j) * ((1 - s) ** (d - j)) * (float(c) * binom[j])
    return out


def explicit_matrix(factors, mats):
    n, m, r = len(factors), mats[0].shape[0], len(mats)
    x = psp.polyvar(n)
    prods = []
    for l in range(r):
        p = psp.Polynomial.constant(n, 1.0)
        for i in range(n):
            p = p * bernstein(x[i], [row[l] for row in factors[i]])
        prods.append(p)
    zero = psp.Polynomial.zero(n)
    arr = [[sum((prods[l] * float(mats[l][a, b]) for l in range(r) if mats[l][a, b] != 0), zero)
            for b in range(m)] for a in range(m)]
    return psp.PolyMatrix(arr, nvars=n), x


def main():
    n, m, method = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    order = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    solver = sys.argv[5] if len(sys.argv) > 5 else "CLARABEL"
    factors, mats = instance(n, m)
    exact = float(np.linalg.eigvalsh(sum(mats)).min())
    rec = {"bench": "banded", "n": n, "m": m, "method": method, "solver": solver,
           "exact": exact}

    if method in ("dense", "zf", "ts1"):
        F, x = explicit_matrix(factors, mats)
        res, total, solve = timed(lambda: psp.pmi_optimize(
            F, ineqs=[1 - xi**2 for xi in x], order=order,
            sparse=(method == "zf"), ts="MD" if method == "ts1" else None,
            ts_order=1 if method == "ts1" else None, solver=solver, **solver_options(solver)))
    elif method in ("lift", "liftdense"):
        # ``sparse`` is forwarded to pmi_optimize for the lifted PMI
        extra = {"sparse": False} if method == "liftdense" else {}
        res, total, solve = timed(lambda: psp.cp_lower_bound(
            factors, box=1.0, basis="bernstein", order=order or 2, method="chord",
            matrices=mats, solver=solver, **extra, **solver_options(solver)))
    elif method == "push":
        res, total, solve = timed(lambda: psp.cp_lower_bound(
            factors, box=1.0, basis="bernstein", order=order or 3, method="push",
            matrices=mats, solver=solver, **solver_options(solver)))
    else:
        raise ValueError(method)
    rec.update(sdp_stats(res))
    rec.update({"order": res.order, "bound": res.value, "status": res.status,
                "time_total": total, "time_solve": solve})
    try:
        xs = res.minimizer()
        rec["x_star"] = [float(v) for v in xs]
        rec["gap"] = float(res.gap(np.clip(xs, -1, 1)))
    except Exception as e:
        rec["x_star_error"] = str(e)
    emit(rec)


if __name__ == "__main__":
    main()
