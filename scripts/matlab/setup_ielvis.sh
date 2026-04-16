#!/usr/bin/env bash
# Clone iELVis and print the MATLAB addpath lines needed to use it.
# Run once after MATLAB is installed.

set -euo pipefail

IELVIS_DIR="${IELVIS_DIR:-$HOME/matlab/iELVis}"
FREESURFER_HOME="${FREESURFER_HOME:-/Applications/freesurfer/8.2.0}"
SUBJECTS_DIR="${SUBJECTS_DIR:-/Users/bentang/Library/CloudStorage/Box-Box/ECoG_Recon}"

if [ ! -d "$IELVIS_DIR" ]; then
    echo "cloning iELVis -> $IELVIS_DIR"
    mkdir -p "$(dirname "$IELVIS_DIR")"
    git clone https://github.com/iELVis/iELVis.git "$IELVIS_DIR"
else
    echo "iELVis already present at $IELVIS_DIR"
fi

if [ ! -d "$SUBJECTS_DIR/cvs_avg35_inMNI152" ]; then
    echo "WARNING: $SUBJECTS_DIR/cvs_avg35_inMNI152 not found" >&2
    exit 1
fi

if [ ! -d "$FREESURFER_HOME/matlab" ]; then
    echo "WARNING: $FREESURFER_HOME/matlab not found — install FreeSurfer first" >&2
    exit 1
fi

cat <<EOF

---------------- paste into MATLAB (or add to startup.m) ----------------
setenv('FREESURFER_HOME', '$FREESURFER_HOME');
setenv('SUBJECTS_DIR',    '$SUBJECTS_DIR');
addpath(genpath('$IELVIS_DIR'));
addpath('$FREESURFER_HOME/matlab');
-------------------------------------------------------------------------

then run:
  cd('$(pwd)/scripts/matlab');
  plot_phase1_electrodes
EOF
