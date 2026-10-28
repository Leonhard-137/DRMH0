#!/usr/bin/env zsh

set -u

project_root=/home/leonhard/sci001
python_bin=/home/leonhard/miniforge3/envs/mica2/bin/python
stage_name=mica_mix220
log_root=$project_root/output/mix220_post_9way_20260824

mkdir -p "$log_root"
cd "$project_root" || exit 1

if pgrep -af 'main.py .*ffa_comp0 --stage mica_mix220' | grep -v "pgrep -af" >/dev/null; then
  print -u2 'A mix220 ffa_comp0 process is already running; refusing to duplicate it.'
  exit 1
fi

remaining_sources=(
  Mrk279_b
  Mrk335
  Mrk509
  Mrk817
  NGC2617_a
  NGC4151_a
  NGC4593
  NGC5548
  NGC6814
)

all_sources=(
  F9_a
  MCG+08-11-011_b
  Mrk142
  Mrk279_b
  Mrk335
  Mrk509
  Mrk817
  NGC2617_a
  NGC4151_a
  NGC4593
  NGC5548
  NGC6814
)

run_parallel_action() {
  local action_name=$1
  shift
  local -a action_sources worker_pids
  local source_name worker_pid
  local worker_index=1

  action_sources=("$@")
  worker_pids=()

  for source_name in "${action_sources[@]}"; do
    printf '[%s] START %-12s %s\n' "$(date '+%F %T')" "$action_name" "$source_name"
    "$python_bin" main.py "$source_name" "$action_name" --stage "$stage_name" \
      >"$log_root/${action_name}_${source_name}.log" 2>&1 &
    worker_pids+=($!)
  done

  for worker_pid in "${worker_pids[@]}"; do
    source_name=${action_sources[$worker_index]}
    if wait "$worker_pid"; then
      printf '[%s] DONE  %-12s %s\n' "$(date '+%F %T')" "$action_name" "$source_name"
    else
      printf '[%s] FAILED %-12s %s — see %s\n' \
        "$(date '+%F %T')" "$action_name" "$source_name" \
        "$log_root/${action_name}_${source_name}.log"
    fi
    (( worker_index++ ))
  done
}

run_parallel_action ffa_comp0 "${remaining_sources[@]}"
run_parallel_action diagnostics "${all_sources[@]}"
run_parallel_action lagfit "${all_sources[@]}"
run_parallel_action distance "${all_sources[@]}"

if "$python_bin" main.py Mrk509 diagnostics --stage mica_gamma2_phys \
  >"$log_root/diagnostics_Mrk509_mica_gamma2_phys.log" 2>&1; then
  printf '[%s] DONE  diagnostics  Mrk509 mica_gamma2_phys\n' "$(date '+%F %T')"
else
  printf '[%s] FAILED diagnostics  Mrk509 mica_gamma2_phys\n' "$(date '+%F %T')"
fi

date '+%F %T' >"$log_root/FINISHED"
printf '[%s] ALL SCHEDULED NON-ICCF POST-PROCESSING FINISHED\n' "$(date '+%F %T')"
