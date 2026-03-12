#!/usr/bin/env bash
set -euo pipefail

mkdir -p public/media/psi-0

if (($# > 0)); then
  sources=("$@")
else
  shopt -s nullglob
  sources=(Psi-0/*)
fi

for src in "${sources[@]}"; do
  base=${src##*/}
  stem=${base%.*}
  lower=$(printf "%s" "$stem" | tr "[:upper:]" "[:lower:]")
  out="public/media/psi-0/${lower}.mp4"

  if [[ "$stem" == "Psi_0_1080P" || "$stem" == "psi0_4k_60fps" || "$stem" == "Psi_0_compressed" ]]; then
    scale_filter="fps=24,scale='if(gt(iw,ih),1280,-2)':'if(gt(iw,ih),-2,720)':force_original_aspect_ratio=decrease"
    crf=30
    preset="veryfast"
  else
    scale_filter="fps=24,scale='if(gt(iw,ih),960,-2)':'if(gt(iw,ih),-2,960)':force_original_aspect_ratio=decrease"
    crf=29
    preset="veryfast"
  fi

  extra_args=()
  if [[ "$stem" == "Psi_0_compressed" ]]; then
    extra_args=(-t 45)
  fi

  ffmpeg -y \
    -i "$src" \
    "${extra_args[@]}" \
    -an \
    -c:v libx264 \
    -preset "$preset" \
    -crf "$crf" \
    -pix_fmt yuv420p \
    -movflags +faststart \
    -vf "$scale_filter" \
    "$out"
done

cp public/media/psi-0/psi_0_compressed.mp4 public/media/psi-0-teaser.mp4
