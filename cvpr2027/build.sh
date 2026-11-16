#!/usr/bin/env bash
# Builds the CVPR 2027 main paper and supplementary material.
# The two documents cross-reference each other (xr-hyper), so they are
# compiled alternately; run from this folder: ./build.sh
set -e
cd "$(dirname "$0")"
rm -f *.aux *.bbl *.blg *.brf *.out
for pass in 1 2 3 4; do
  pdflatex -interaction=nonstopmode main_cvpr.tex  >/dev/null || true
  pdflatex -interaction=nonstopmode suppl_cvpr.tex >/dev/null || true
  if [ "$pass" -eq 1 ]; then bibtex main_cvpr >/dev/null; bibtex suppl_cvpr >/dev/null; fi
done
for d in main_cvpr suppl_cvpr; do
  echo "$d.pdf: $(pdfinfo $d.pdf | awk '/Pages/{print $2}') pages, $(grep -c '^!' $d.log) errors, $(grep -c 'undefined' $d.log) undefined"
done
