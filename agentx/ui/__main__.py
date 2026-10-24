"""Module execution entry point for python -m agentx.ui."""

import sys
from agentx.ui.app import run_ui

if __name__ == "__main__":
    sys.exit(run_ui())
