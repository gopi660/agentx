"""Standalone launcher script for the AgentX Terminal User Interface."""

import sys
from agentx.ui.app import run_ui

if __name__ == "__main__":
    sys.exit(run_ui())
