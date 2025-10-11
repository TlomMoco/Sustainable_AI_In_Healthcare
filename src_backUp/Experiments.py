from __future__ import annotations
import subprocess


# Minimal holder to show where to add experiment loops/config sweeps

CENTRALIZED_CMD = ["python", "-m", "src.Centralized"]
SERVER_CMD = ["python", "-m", "src.Server"]
CLIENT_CMD = lambda cid: ["python", "-m", "src.Client", "--cid", str(cid)]


if __name__ == "__main__":
    # Example: run centralized once
    subprocess.run(CENTRALIZED_CMD, check=True)
    # For federated, open multiple terminals or use process manager (tmux, etc.)