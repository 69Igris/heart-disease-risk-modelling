# IEEE report (LaTeX)
`main.tex` uses the standard `IEEEtran` class (`\documentclass[conference]{IEEEtran}`). `phase1_report.pdf` is the compiled copy to submit.

All result tables in `tables/` are generated from `results/*.csv`; never edit them by hand.

```bash
python src/build_latex_tables.py        # from the repo root: refresh tables after any re-run
cd report
pdflatex main && bibtex main && pdflatex main && pdflatex main
cp main.pdf phase1_report.pdf
```
No local LaTeX? Use Overleaf: upload `main.tex`, `references.bib`, the `tables/` folder and the PNGs from `results/figures/` into a folder called `figures`, and change `\graphicspath{{../results/figures/}}` in main.tex to `\graphicspath{{figures/}}`.

Before submitting: check every reference on Google Scholar, and check the affiliation line you want under the author names.
