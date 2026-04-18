#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

config_name=pi05_b1k-turning_on_radio_cfgrl_lr2.5e-6_step20k
exp_name=pi05_b1k-turning_on_radio-cfgrl_optimal-demos_suboptimal-rollbacks_step20k

CMD="python scripts/train.py ${config_name} --exp_name=${exp_name} --overwrite \
  --weight_loader.params_path=/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/params \
  --batch_size=256 \
  --checkpoint_base_dir=/mnt/project_rlinf_hs/mjwei/download_models/behavior-1k/cfgrl-comet/task00 \
  --assets_base_dir=/mnt/project_rlinf_hs/mjwei/download_models/sunshk/comet_submission/pi05-pt50-pretrain-50k/assets \
  --save_interval 5000 \
  --log_interval 10"

LOG_DIR="/mnt/project_rlinf/mjwei/repo/openpi-comet/logs/${exp_name}-$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${LOG_DIR}"

MEGA_LOG_FILE="${LOG_DIR}/run.log"
echo "Logging to ${MEGA_LOG_FILE}"
echo "${CMD}" > "${MEGA_LOG_FILE}"
${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"
