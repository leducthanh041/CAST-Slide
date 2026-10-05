#!/bin/bash

# Shared runtime defaults for a relocatable MergeSlide_TTA checkout.
MERGESLIDE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$MERGESLIDE_SCRIPT_DIR/.." && pwd)}"
export MERGESLIDE_DATA_ROOT="${MERGESLIDE_DATA_ROOT:-$(cd "$PROJECT_ROOT/.." && pwd)/dataset}"
export MERGESLIDE_CHECKPOINT_ROOT="${MERGESLIDE_CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints_ood}"
export MERGESLIDE_LOCAL_ROOT="${MERGESLIDE_LOCAL_ROOT:-$PROJECT_ROOT}"

if [ -z "${PYTHON_BIN:-}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
export PYTHON_BIN

