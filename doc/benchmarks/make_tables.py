"""Turn results.jsonl into the LaTeX tables of the paper (doc/tables/*.tex)."""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "tables")


def load():
    recs = {}
    with open(os.path.join(HERE, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            key = (r["bench"], r["n"], r.get("m"), r["method"], r.get("order"))
            if r["bench"] == "banded":
                key = (r["bench"], r["n"], r.get("m"), r["method"], None)
            recs[key] = r  # the last run of a configuration wins
    return recs


def sci(v, digits=1):
    if v == 0:
        return "0"
    e = int(math.floor(math.log10(abs(v))))
    m = v / 10 ** e
    return r"$%.*f{\cdot}10^{%d}$" % (digits, m, e)


def secs(r):
    if r is None:
        return ""
    if "failed" in r:
        return "mem" if "memory" in r["failed"] else (
            r"$>$1800" if "time" in r["failed"] else "fail")
    t = r["time_total"]
    return "%.1f" % t if t < 10 else "%.0f" % t


def gap(bound, ref):
    g = (bound - ref) / abs(ref)
    if abs(g) < 1e-6:
        return r"$<\!10^{-6}$"
    return sci(g)


def gain_table(recs):
    ns = sorted({k[1] for k in recs if k[0] == "gain"})
    rows = []
    for n in ns:
        get = lambda meth, order: recs.get(("gain", n, None, meth, order))
        dense, ts1 = get("dense", n), get("ts1", n)
        lifted = [get("push", 2), get("push", 3), get("chord", 2)]
        exact = next((r["exact"] for r in [dense, ts1] + lifted
                      if r and r.get("exact") is not None), None)
        star = ""
        if exact is None:  # best feasible value at a recovered maximizer
            vals = [r["f_x_star"] for r in lifted
                    if r and "failed" not in r and r.get("f_x_star") == r.get("f_x_star")]
            exact = max(vals) if vals else None
            star = r"$^\dagger$"
        dblk = 2 * math.factorial(2 * n) // math.factorial(n) ** 2
        cells = ["%d" % n,
                 ("%.6g" % exact if exact < 1e5 else sci(exact, 5)) + star if exact else "",
                 "%d" % dblk if dblk < 1e5 else sci(dblk),
                 secs(dense) if dense else "",
                 "%d" % ts1["max_block"] if ts1 and "failed" not in ts1 else "",
                 secs(ts1) if ts1 else ""]
        for r in lifted:
            if r is None:
                cells += ["", ""]
            elif "failed" in r:
                cells += [secs(r), ""]
            else:
                cells += [secs(r), gap(r["bound"], exact)]
        rows.append(" & ".join(cells) + r" \\")
    blk = {m: next((recs[k]["max_block"] for k in recs
                    if k[0] == "gain" and k[3] == m[0] and k[4] == m[1]
                    and "failed" not in recs[k]), "")
           for m in [("push", 2), ("push", 3), ("chord", 2)]}
    head = (r"\begin{tabular}{@{}rr rr rr rr rr rr@{}}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{2}{c}{dense} & \multicolumn{2}{c}{TS, $s=1$}"
            r" & \multicolumn{2}{c}{SL-push, $d=2$} & \multicolumn{2}{c}{SL-push, $d=3$}"
            r" & \multicolumn{2}{c}{SL-chord, $d=2$}\\" "\n"
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(l){11-12}"
            "\n"
            r"$n$ & $\lambda^\star$ & block & time & block & time"
            r" & time & gap & time & gap & time & gap\\" "\n"
            r" & & & & & & \multicolumn{2}{c}{block %s} & \multicolumn{2}{c}{block %s}"
            r" & \multicolumn{2}{c}{block %s}\\" "\n" r"\midrule" "\n"
            % (blk[("push", 2)], blk[("push", 3)], blk[("chord", 2)]))
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


BANDED_METHODS = ["dense", "zf", "ts1", "liftdense", "lift", "push"]


def banded_table(recs):
    sizes = sorted({(k[1], k[2]) for k in recs if k[0] == "banded"},
                   key=lambda nm: (nm[1] != 8, nm[1], nm[0]))
    rows = []
    for n, m in sizes:
        cells = ["%d" % n, "%d" % m]
        worst = 0.0
        for meth in BANDED_METHODS:
            r = recs.get(("banded", n, m, meth, None))
            if r is None:
                cells.append("")
            elif "failed" in r:
                cells.append(secs(r))
            else:
                cells.append("%s (%d)" % (secs(r), r["max_block"]))
                if meth != "push":
                    worst = max(worst, abs(r["bound"] - r["exact"]) / abs(r["exact"]))
        push = recs.get(("banded", n, m, "push", None))
        cells.append(gap(push["bound"], push["exact"])
                     if push and "failed" not in push else "")
        cells.append(sci(worst) if worst >= 1e-6 else r"$<\!10^{-6}$")
        rows.append(" & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{@{}rr llllll rr@{}}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{3}{c}{no lifting} & \multicolumn{3}{c}{lifting}"
            r" & \multicolumn{2}{c}{rel.\ gap}\\" "\n"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}" "\n"
            r"$n$ & $m$ & dense & matrix & TS, $s=1$ & SL-chord & SL-chord$\wedge$mat."
            r" & SL-push & push & others\\" "\n" r"\midrule" "\n")
    return head + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n"


def main():
    recs = load()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "gain.tex"), "w") as f:
        f.write(gain_table(recs))
    with open(os.path.join(OUT, "banded.tex"), "w") as f:
        f.write(banded_table(recs))
    print(open(os.path.join(OUT, "gain.tex")).read())
    print(open(os.path.join(OUT, "banded.tex")).read())


if __name__ == "__main__":
    main()
