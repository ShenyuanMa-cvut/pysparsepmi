"""Shared helpers for the benchmarks of the paper (doc/paper.tex)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import pysparsepmi as psp  # noqa: E402


def sdp_stats(res):
    """Solver-independent size of the SDP behind a result.

    Counts every PSD variable of the compiled CVXPY problem (certificate
    and localizing multipliers alike).
    """
    prob = res.problem.compile()
    sizes = sorted(
        (v.shape[0] for v in prob.variables() if v.attributes.get("PSD")),
        reverse=True,
    )
    sm = prob.size_metrics
    return {
        "psd_blocks": len(sizes),
        "max_block": sizes[0] if sizes else 0,
        "sum_block_sq": int(sum(s * s for s in sizes)),
        "scalar_vars": int(sm.num_scalar_variables),
        "scalar_eqs": int(sm.num_scalar_eq_constr),
    }


def solver_options(solver):
    """Tight tolerances for the interior-point solver used in the paper."""
    if solver == "CLARABEL":
        return {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10}
    return {}


def timed(fn):
    # perf_counter does not advance while the machine sleeps (macOS:
    # mach_absolute_time), so the measured time is pure computation
    t0 = time.perf_counter()
    res = fn()
    total = time.perf_counter() - t0
    stats = res.problem.compile().solver_stats
    solve = getattr(stats, "solve_time", None)
    return res, total, solve


def emit(record):
    print(json.dumps(record), flush=True)
