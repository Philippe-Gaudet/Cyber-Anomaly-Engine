#!/usr/bin/env bash
# Launch the Barcode dashboard using the project's .venv (Python 3.11).
# The system Streamlit on macOS uses Python 3.13, which can't import gtda.
set -e
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/streamlit" ]; then
  echo "Error: .venv/bin/streamlit not found." >&2
  echo "Run: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e ." >&2
  exit 1
fi

exec .venv/bin/streamlit run src/barcode/dashboard.py "$@"
