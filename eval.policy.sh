source .venv/bin/activate

uv run scripts/serve_b1k.py \
  --task_name="freeze_pies" \
  --control_mode=receeding_horizon \
  --max_len=32 \
  policy:checkpoint \
  --policy.config=pi05_b1k-freeze_pies_lr2.5e-6_step20k_rft \
  --policy.dir="/home/dell/mjwei/download_models/sunshk/openpi_comet/pi05-b1kpt50-cs32"