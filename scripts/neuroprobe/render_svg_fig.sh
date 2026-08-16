#!/usr/bin/env bash
# SVG -> NeurIPS-safe PDF (+ PNG proof).  Usage: render_svg_fig.sh <in.svg> <outdir>
# rsvg-convert alone emits Type 3 subsets, which NeurIPS rejects, so the text is
# outlined by ghostscript (-dNoOutputFonts).  The result carries no fonts at all.
set -euo pipefail
SVG="$1"; OUT="${2:-.}"; STEM="$(basename "${SVG%.svg}")"
mkdir -p "$OUT"
rsvg-convert -f pdf -o "$OUT/$STEM.raw.pdf" "$SVG"
gs -q -o "$OUT/$STEM.pdf" -sDEVICE=pdfwrite -dNoOutputFonts \
   -dAutoRotatePages=/None -dPDFSETTINGS=/prepress "$OUT/$STEM.raw.pdf"
rm -f "$OUT/$STEM.raw.pdf"
rsvg-convert -f png -z 4 -o "$OUT/$STEM.png" "$SVG"
cp -f "$SVG" "$OUT/$STEM.svg"
echo "--- $OUT/$STEM.pdf"
pdfinfo "$OUT/$STEM.pdf" | grep -i "page size"
F=$(pdffonts "$OUT/$STEM.pdf" | tail -n +3)
if [ -z "$F" ]; then echo "fonts: none (all text outlined) -- Type3: False"; else echo "$F"; fi
