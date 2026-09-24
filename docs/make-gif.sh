#!/bin/sh
# Rebuild docs/img/pour.gif from one headless --shot run.
# A frame every 12 ticks at 20 fps plays back at the real 240 Hz rate.
set -eu
cd "$(dirname "$0")/.."
cargo build --release
frames=$(mktemp -d)
trap 'rm -rf "$frames"' EXIT

./target/release/swarf --shot "$frames/f.png" --every 12 --ticks 1260 \
    --scene lab --seed 7 --size 640x360 --zoom 1.25 --centre 2560,1068 \
    --drop water@2450,1100,55 --drop sand@2480,890,55 --drop water@2690,1010,40

ffmpeg -v error -y -framerate 20 -i "$frames/f-%04d.png" \
    -vf "tpad=stop_mode=clone:stop_duration=1.5,split[a][b];[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
    -loop 0 docs/img/pour.gif
ls -lh docs/img/pour.gif
