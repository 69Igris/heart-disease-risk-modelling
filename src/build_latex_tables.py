"""Write the LaTeX tables used by report/main.tex directly from results/*.csv.

No number in the report tables is typed by hand: re-run the notebook, run this script,
then recompile the report.   Usage (from repo root):  python src/build_latex_tables.py
"""
from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def ci(r, k, d=3):
    return f"{r[k]:.{d}f} [{r[k + ' lo']:.{d}f}, {r[k + ' hi']:.{d}f}]"


def write(name, header, rows, colspec):
    body = " \\\\\n".join(" & ".join(str(c) for c in row) for row in rows)
    body = re.sub(r"(?<![A-Za-z0-9])-(?=\d)", "$-$", body)   # real minus signs for negative numbers
    tex = (f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body} \\\\\n"
           "\\bottomrule\n\\end{tabular}\n")
    (OUT / name).write_text(tex, encoding="utf-8")


# ---- Table: main results -------------------------------------------------------------
m = pd.read_csv(RES / "phase1_metrics.csv").set_index("model")
prior = m.loc["Prior only"]
# prior-only sens/spec are degenerate (it never reaches 0.5 as a positive call), so they are not shown
rows = [["Prior only", f"{prior['ROC AUC']:.3f}", f"{prior['PR AUC']:.3f}", "--", "--",
         f"{prior['Brier']:.3f}", f"{prior['ECE']:.3f}", "--"]]
for name in ["Naive Bayes", "Bayesian net", "LogReg (disc.)", "LogReg (cont.)", "Random forest"]:
    r = m.loc[name]
    rows.append([name.replace("(disc.)", "(binned)"),
                 ci(r, "ROC AUC"), f"{r['PR AUC']:.3f}", f"{r['Sensitivity']:.3f}",
                 f"{r['Specificity']:.3f}", f"{r['Brier']:.3f}", f"{r['ECE']:.3f}", f"{r['Cal. slope']:.2f}"])
write("main_results.tex", "Model & ROC AUC [95\\% CI] & AP & Sens. & Spec. & Brier & ECE & Slope", rows, "lccccccc")

# ---- Table: paired comparisons ---------------------------------------------------------
p = pd.read_csv(RES / "phase1_paired.csv")
rows = []
for (a, b), g in p.groupby(["A", "B"], sort=False):
    auc = g[g.metric == "ROC AUC"].iloc[0]; br = g[g.metric == "Brier"].iloc[0]; ap = g[g.metric == "PR AUC"].iloc[0]
    rows.append([f"{a} $-$ {b}".replace("(disc.)", "(binned)"),
                 f"{auc['mean diff (A-B)']:+.3f} [{auc['lo']:+.3f}, {auc['hi']:+.3f}]", f"{auc['corrected p']:.2f}",
                 f"{ap['mean diff (A-B)']:+.3f} [{ap['lo']:+.3f}, {ap['hi']:+.3f}]",
                 f"{br['mean diff (A-B)']:+.3f} [{br['lo']:+.3f}, {br['hi']:+.3f}]"])
write("paired.tex", "Comparison (A $-$ B) & $\\Delta$AUC [95\\% CI] & $p_{\\mathrm{corr}}$ & $\\Delta$AP [95\\% CI] & $\\Delta$Brier [95\\% CI]", rows, "lcccc")

# ---- Table: matched HR control ---------------------------------------------------------
mc = pd.read_csv(RES / "phase1_matched_hr.csv").set_index("model")
rows = []
for name, label in [("LogReg (matched HR)", "Continuous, HR ratio (matched)"), ("LogReg (disc.)", "Binned (all numerics)")]:
    r = mc.loc[name]; rows.append([label, ci(r, "ROC AUC"), f"{r['PR AUC']:.3f}", f"{r['Brier']:.3f}", f"{r['Cal. slope']:.2f}"])
r = m.loc["LogReg (cont.)"]; rows.insert(0, ["Continuous, raw max HR (baseline)", ci(r, "ROC AUC"), f"{r['PR AUC']:.3f}", f"{r['Brier']:.3f}", f"{r['Cal. slope']:.2f}"])
write("matched_hr.tex", "Logistic regression input & ROC AUC [95\\% CI] & AP & Brier & Slope", rows, "lcccc")

# ---- Table: without ca/thal ------------------------------------------------------------
b = pd.read_csv(RES / "phase1_scenarioB.csv").set_index("model")
rows = []
for name in ["Naive Bayes", "Bayesian net", "LogReg (cont.)", "Random forest"]:
    rb, ra = b.loc[name], m.loc[name]
    rows.append([name, f"{ra['ROC AUC']:.3f}", ci(rb, "ROC AUC"), f"{rb['ROC AUC'] - ra['ROC AUC']:+.3f}", f"{rb['Brier']:.3f}", f"{rb['Cal. slope']:.2f}"])
write("no_ca_thal.tex", "Model & AUC (13 feat.) & AUC without ca, thal [95\\% CI] & $\\Delta$AUC & Brier & Slope", rows, "lccccc")

# ---- Table: BDeu equivalent sample size ------------------------------------------------
s = pd.read_csv(RES / "phase1_sensitivity.csv").set_index("variant")
rows = [[v.replace("ess=", "ESS $=$ "), f"{r['ROC AUC']:.3f}", f"{r['PR AUC']:.3f}", f"{r['Brier']:.3f}", f"{r['ECE']:.3f}", f"{r['Cal. slope']:.2f}"]
        for v, r in s.iterrows()]
write("ess.tex", "Variant & ROC AUC & AP & Brier & ECE & Slope", rows, "lccccc")
print("wrote tables to", OUT)
