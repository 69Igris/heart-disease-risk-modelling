# Phase 2 IEEE report (LaTeX)
This report is separate from the Phase 1 report in `report/`. `main.tex` uses the standard `IEEEtran` class; `phase2_report.pdf` is the compiled copy to submit.

All result tables in `tables/` are generated from `results/phase2_*.csv`; never edit them by hand. Figures are read from `results/figures/` (`fig_p2_*.png`).

```bash
python src/build_phase2_tables.py           # from the repo root: refresh tables after any re-run of notebook 02
cd report/phase2
tectonic main.tex                           # or: pdflatex main && bibtex main && pdflatex main && pdflatex main
mv main.pdf phase2_report.pdf
```
No local LaTeX? Use Overleaf. Upload `main.tex`, `references.bib`, the `tables/` folder, and the `fig_p2_*.png` files from `results/figures/` into a folder called `figures`. Then change `\graphicspath{{../../results/figures/}}` in main.tex to `\graphicspath{{figures/}}`.

Before submitting: check every reference on Google Scholar. `references.bib` holds only the 10 sources the report cites.
