source /opt/venv/openpi/bin/activate

export PYTHONPATH=/mnt/public/quanlu/openpi-comet/src:$PYTHONPATH
export PYTHONPATH=/opt/venv/openpi/BEHAVIOR-1K/OmniGibson:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
python scripts/serve_b1k.py \
  --task_name=turning_on_radio \
  --control_mode=receeding_horizon \
  --max_len=32 \
  --port=8001 \
  policy:checkpoint \
  --policy.config="pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft" \
  --policy.dir=/mnt/public/chenjiawei/projects/models/openpi_comet/pi05-b1kpt12-cs32