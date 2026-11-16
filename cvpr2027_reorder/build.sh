#!/usr/bin/env bash
# Builds the CVPR 2027 main paper and supplementary material, and refreshes the
# static cross-reference label files (xref_*.tex) that let each document compile
# standalone on Overleaf. Run from anywhere: ./build.sh
set -e
cd "$(dirname "$0")"
rm -f *.aux *.bbl *.blg *.brf *.out
for pass in 1 2 3 4; do
  pdflatex -interaction=nonstopmode main_cvpr.tex  >/dev/null || true
  pdflatex -interaction=nonstopmode suppl_cvpr.tex >/dev/null || true
  if [ "$pass" -eq 1 ]; then bibtex main_cvpr >/dev/null; bibtex suppl_cvpr >/dev/null; fi
  python3 make_xref.py main_cvpr.aux  xref_main_cvpr.tex  >/dev/null
  python3 make_xref.py suppl_cvpr.aux xref_suppl_cvpr.tex >/dev/null
done
for d in main_cvpr suppl_cvpr; do
  echo "$d.pdf: $(pdfinfo $d.pdf | awk '/Pages/{print $2}') pages, $(grep -c '^!' $d.log) errors, $(grep -c 'undefined' $d.log) undefined"
done
echo "Commit xref_main_cvpr.tex and xref_suppl_cvpr.tex together with any change that renumbers sections, tables or figures."
