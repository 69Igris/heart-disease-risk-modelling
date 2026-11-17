"""Write the LaTeX tables used by report/phase2/main.tex directly from results/phase2_*.csv.

No number in the Phase 2 report tables is typed by hand: re-run notebook 02, run this script,
then recompile the report.   Usage (from repo root):  python src/build_phase2_tables.py
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "report" / "phase2" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def ci(r, k, d=3):
    return f"{r[k]:.{d}f} [{r[k + ' lo']:.{d}f}, {r[k + ' hi']:.{d}f}]"


def write(name, header, rows, colspec):
    body = " \\\\\n".join(" & ".join(str(c) for c in row) for row in rows)
    body = re.sub(r"(?<![A-Za-z0-9])-(?=\d)", "$-$", body)   # real minus signs for negative numbers
    tex = (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body} \\\\\n"
           "\\bottomrule\n\\end{tabular}\n")
    (OUT / name).write_text(tex, encoding="utf-8")


# ---- Table: main results (Phase 1 baselines + MLP variants) ---------------------------
m = pd.read_csv(RES / "phase2_metrics.csv").set_index("model")
rows = []
for name in ["Naive Bayes", "Bayesian net", "LogReg (disc.)", "LogReg (cont.)", "Random forest", "MLP", "MLP (binned)"]:
    r = m.loc[name]
    label = name.replace("(disc.)", "(binned)")
    label = f"\\textbf{{{label}}}" if name == "MLP" else label
    rows.append([label, ci(r, "ROC AUC"), f"{r['PR AUC']:.3f}", f"{r['Sensitivity']:.3f}", f"{r['Specificity']:.3f}",
                 ci(r, "Brier"), f"{r['ECE']:.3f}", f"{r['Log loss']:.3f}", f"{r['Cal. slope']:.2f}"])
write("main_results.tex", "Model & ROC AUC [95\\% CI] & AP & Sens. & Spec. & Brier [95\\% CI] & ECE & Log loss & Slope", rows, "lcccccccc")

# ---- Table: paired comparisons ---------------------------------------------------------
p = pd.read_csv(RES / "phase2_paired.csv")
rows = []
for (a, b), g in p.groupby(["A", "B"], sort=False):
    auc = g[g.metric == "ROC AUC"].iloc[0]; ap = g[g.metric == "PR AUC"].iloc[0]; br = g[g.metric == "Brier"].iloc[0]
    rows.append([f"{a} $-$ {b}", f"{auc['mean diff (A-B)']:+.3f} [{auc['lo']:+.3f}, {auc['hi']:+.3f}]", f"{auc['corrected p']:.2f}",
                 f"{ap['mean diff (A-B)']:+.3f} [{ap['lo']:+.3f}, {ap['hi']:+.3f}]",
                 f"{br['mean diff (A-B)']:+.3f} [{br['lo']:+.3f}, {br['hi']:+.3f}]"])
write("paired.tex", "Comparison (A $-$ B) & $\\Delta$AUC [95\\% CI] & $p_{\\mathrm{corr}}$ & $\\Delta$AP [95\\% CI] & $\\Delta$Brier [95\\% CI]", rows, "lcccc")

# ---- Table: without ca/thal --------------------------------------------------------------
b = pd.read_csv(RES / "phase2_scenarioB.csv").set_index("model")
rows = []
for name in ["Naive Bayes", "Bayesian net", "LogReg (cont.)", "Random forest", "MLP"]:
    rb, ra = b.loc[name], m.loc[name]
    rows.append([name, ci(rb, "ROC AUC"), f"{rb['ROC AUC'] - ra['ROC AUC']:+.3f}", f"{rb['Brier']:.3f}", f"{rb['Cal. slope']:.2f}"])
write("no_ca_thal.tex", "Model & AUC [95\\% CI] & $\\Delta$AUC & Brier & Slope", rows, "lcccc")

# ---- Table: chosen MLP configurations ----------------------------------------------------
hp = pd.read_csv(RES / "phase2_mlp_hparams.csv")
g = hp.groupby(["hidden", "weight_decay", "dropout"]).agg(n=("split", "size"), ep=("epochs", "median")).reset_index().sort_values("n", ascending=False)
rows = [[" $\\to$ ".join(x for x in h.strip("()").split(",") if x.strip()).replace(" ", "") .replace("$\\to$", " $\\to$ "), f"{wd:g}", f"{dr:g}", n, f"{ep:.0f}"]
        for h, wd, dr, n, ep in g.itertuples(index=False)]
write("hparams.tex", "Hidden units & Weight decay & Dropout & Times chosen & Median epochs", rows, "lcccc")

# ---- Table: explanation importance ---------------------------------------------------------
imp = pd.read_csv(RES / "phase2_importance.csv", index_col=0).sort_values("MLP |SHAP|", ascending=False)
rows = [[i.replace("risk factors (age, sex, BP, chol, FBS)", "risk factors$^\\dagger$"), f"{r.iloc[0]:.2f}", f"{r.iloc[1]:.2f}"] for i, r in imp.iterrows()]
write("importance.tex", "Variable & BN $|$WoE$|$ & MLP $|$SHAP$|$", rows, "lcc")

# ---- Table: compute cost --------------------------------------------------------------------
c = pd.read_csv(RES / "phase2_cost.csv").set_index("model")
rows = [[n, f"{r['seconds per fold']:.3f}", str(r["parameters"]).replace("×", "$\\times$")] for n, r in c.iterrows()]
write("cost.tex", "Model & Seconds per fold & Size", rows, "lcc")
print("wrote tables to", OUT)
