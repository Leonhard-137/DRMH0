#!/bin/bash
cd /home/leonhard/sci001 || exit 125
/home/leonhard/miniforge3/envs/astro/bin/python -u /home/leonhard/sci001/src/run_mica.py /home/leonhard/sci001/test/mica_recovery/f9_m2_b/cases/response_beyond B run --stage mica_recovery > /home/leonhard/sci001/test/mica_recovery/f9_m2_b/monitoring/restarts/20260905T025350Z/response_beyond/launcher.log 2>&1
mica_exit=$?
printf '%s\n' "$mica_exit" > /home/leonhard/sci001/test/mica_recovery/f9_m2_b/monitoring/restarts/20260905T025350Z/response_beyond/exit_code.txt
exit "$mica_exit"
