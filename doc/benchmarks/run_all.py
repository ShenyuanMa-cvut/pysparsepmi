"""Run every benchmark of the paper and append JSON lines to results.jsonl.

Each configuration runs in its own process with a time limit (on a monotonic
clock that does not advance while the machine sleeps) and a resident-memory
cap; a configuration that fails is recorded with its reason and larger
instances of the same method are skipped.

Usage: python run_all.py [gain|banded|all] [TIME_LIMIT_S] [MEM_LIMIT_GB]
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.jsonl")

GAIN_N = [3, 4, 5, 6, 8, 12, 16, 24, 32, 40]
GAIN_METHODS = [("dense", "-"), ("ts1", "-"), ("push", "2"), ("push", "3"), ("chord", "2")]

BANDED_N = [2, 3, 4, 6, 8, 12, 16, 24, 32]  # with m = 8
BANDED_M = [4, 16, 32, 64]  # with n = 8
BANDED_METHODS = ["dense", "zf", "ts1", "liftdense", "lift", "push"]


def rss_kb(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])
        return int(out.strip() or 0)
    except (subprocess.CalledProcessError, ValueError):
        return 0


def run(args, key, time_limit, mem_kb):
    t0 = time.monotonic()
    proc = subprocess.Popen([sys.executable] + args, cwd=HERE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            universal_newlines=True)
    reason, peak = None, 0
    while proc.poll() is None:
        time.sleep(1.0)
        peak = max(peak, rss_kb(proc.pid))
        if time.monotonic() - t0 > time_limit:
            reason = "time limit"
        elif peak > mem_kb:
            reason = "memory limit"
        if reason:
            proc.kill()
            proc.wait()
            break
    out = proc.stdout.read()
    rec = None
    for line in out.splitlines():
        if line.startswith("{"):
            rec = json.loads(line)
    if rec is None:
        rec = dict(key)
        rec["failed"] = reason or "error (exit %s)" % proc.returncode
    rec["peak_rss_gb"] = peak / 2**20
    rec["wall"] = time.monotonic() - t0
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec.get(k) for k in
                      ("bench", "n", "m", "method", "order", "bound", "exact",
                       "max_block", "time_total", "failed")}), flush=True)
    return "failed" not in rec


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
    mem_kb = float(sys.argv[3] if len(sys.argv) > 3 else 12) * 2**20
    if which in ("gain", "all"):
        for method, order in GAIN_METHODS:
            for n in GAIN_N:
                key = {"bench": "gain", "n": n, "method": method,
                       "order": None if order == "-" else int(order)}
                if not run(["gain.py", str(n), method, order], key, time_limit, mem_kb):
                    break
    if which in ("banded", "all"):
        for method in BANDED_METHODS:
            sizes = [(n, 8) for n in BANDED_N]
            if method in ("liftdense", "lift", "push"):
                sizes += [(8, m) for m in BANDED_M]
            failed_n = failed_m = False
            for n, m in sizes:
                if (m == 8 and failed_n) or (m != 8 and failed_m):
                    continue
                key = {"bench": "banded", "n": n, "m": m, "method": method}
                if not run(["banded.py", str(n), str(m), method], key, time_limit, mem_kb):
                    if m == 8:
                        failed_n = True
                    else:
                        failed_m = True


if __name__ == "__main__":
    main()
