"""Benchmark 1: worst-case gain of a product of parameter-dependent matrices.

    max_{x in [-1, 1]^n}  lambda_max(P(x)' P(x)),   P(x) = M(x_1) ... M(x_n),

with M(x) = [[0.9, 0.5 x], [0.3 x, 0.8]] (examples/matrix_product_gain.py).
P'P is a 2x2 polynomial matrix of degree 2n in n variables, dense in the
variables. Since P is affine in each x_i, lambda_max(P'P) = ||P||^2 is convex
in each x_i, so the maximum is attained at a vertex of the box: vertex
enumeration gives the exact value for moderate n. Since M(-x) = D M(x) D with
D = diag(1, -1), the gain is invariant under x -> -x; all methods restrict
x_1 to [0, 1] (no loss of generality), which makes the maximizer unique so
that it can be read off the pseudo-moments.

Methods (one run per call; prints one JSON line):

  dense   Scherer-Hol / Henrion-Lasserre relaxation of minimal order n
  ts1     same, first step of the term-sparsity hierarchy (TS="MD", s = 1)
  push    SL-push state lifting with matrix-valued last stage (this paper)
  chord   SL-chord state lifting + correlative + matrix sparsity (this paper)

Usage: python gain.py N METHOD [ORDER] [SOLVER]
"""
import sys

import numpy as np

from common import emit, psp, sdp_stats, solver_options, timed


def M(x):
    return [[0.9, 0.5 * x], [0.3 * x, 0.8]]


def congruence(S, x):
    """M(x)' S M(x) for a symmetric 2x2 S stored as [s11, s12, s22]."""
    Mx = M(x)
    Sm = [[S[0], S[1]], [S[1], S[2]]]
    return [
        [sum(Mx[c][a] * Sm[c][d] * Mx[d][b] for c in range(2) for d in range(2))
         for b in range(2)]
        for a in range(2)
    ]


def first(s, x):
    R = congruence([1.0, 0.0, 1.0], x)
    return [R[0][0], R[0][1], R[1][1]]


def step(s, x):
    R = congruence(s, x)
    return [R[0][0], R[0][1], R[1][1]]


def last(s, x):
    return congruence(s, x)  # matrix-valued terminal map


def objective(x):
    P = np.eye(2)
    for xi in x:
        P = P @ np.array(M(xi))
    return np.linalg.eigvalsh(P.T @ P).max()


def vertex_max(n):
    """Exact maximum by enumerating {-1, 1}^n (prefix products, vectorized)."""
    Ms = np.array([M(-1.0), M(1.0)])  # (2, 2, 2)
    P = Ms.copy()
    for _ in range(n - 1):
        P = np.einsum("aij,bjk->abik", P, Ms).reshape(-1, 2, 2)
    G = np.einsum("aji,ajk->aik", P, P)
    return float(np.linalg.eigvalsh(G)[:, -1].max())


def dense_matrix(n):
    x = psp.polyvar(n)
    P = psp.PolyMatrix(M(x[0]), nvars=n)
    for i in range(1, n):
        P = P @ psp.PolyMatrix(M(x[i]), nvars=n)
    PT = psp.PolyMatrix([[P.arr[j, i] for j in range(2)] for i in range(2)])
    return PT @ P, x


def main():
    n = int(sys.argv[1])
    method = sys.argv[2]
    order = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    solver = sys.argv[4] if len(sys.argv) > 4 else "CLARABEL"
    rec = {"bench": "gain", "n": n, "method": method, "order": order, "solver": solver}

    if method in ("dense", "ts1"):
        F, x = dense_matrix(n)
        ts = "MD" if method == "ts1" else None
        box = [x[0] * (1 - x[0])] + [1 - xi**2 for xi in x[1:]]
        res, total, solve = timed(lambda: psp.pmi_optimize(
            F * (-1.0), ineqs=box, order=order or n,
            ts=ts, ts_order=1 if ts else None, solver=solver, **solver_options(solver)))
        value = None if res.value is None else -res.value
    else:
        maps = [first] + [step] * (n - 2) + [last]
        res, total, solve = timed(lambda: psp.composition_lower_bound(
            maps, box=[(0.0, 1.0)] + [1.0] * (n - 1), order=order, method=method,
            sense="max", solver=solver, **solver_options(solver)))
        value = res.value
    rec.update(sdp_stats(res))
    rec.update({"order": res.order, "bound": value, "status": res.status,
                "time_total": total, "time_solve": solve})
    try:
        xs = res.minimizer()
        rec["x_star"] = [float(v) for v in xs]
        rec["f_x_star"] = float(objective(np.clip(xs, -1, 1)))
    except Exception as e:  # moment recovery is optional
        rec["x_star_error"] = str(e)
    if n <= 20:
        rec["exact"] = vertex_max(n)
    emit(rec)


if __name__ == "__main__":
    main()
