#!/usr/bin/env python3
"""One-time handover after the current controller drains its active fits."""
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
QUEUE_DIR = HERE.parent.parent
sys.path.insert(0, str(QUEUE_DIR))
from run_psd_candidate_recovery import active_prior_workers

old_pid = 28595
old_stat = Path(f'/proc/{old_pid}/stat')
old_identity = old_stat.read_text().rsplit(')', 1)[1].split()[19] if old_stat.exists() else None

def old_controller_alive():
    try:
        return old_stat.read_text().rsplit(')', 1)[1].split()[19] == old_identity
    except FileNotFoundError:
        return False

def message(text):
    print(datetime.now(timezone.utc).isoformat(), text, flush=True)

message('Recovery armed; waiting for old controller and all prior workers to finish.')
while old_controller_alive() or active_prior_workers():
    time.sleep(5)
message('Old controller and workers have exited; resuming checked queue with plot repair.')
# The queue reacquires its lock, validates inputs and skips completed results.
os.execv(sys.executable, [sys.executable, '-B', '-u',
                         str(QUEUE_DIR/'run_psd_candidate_recovery.py'),
                         '--run', '--max-running', '8'])
