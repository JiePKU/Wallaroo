export PYTHONPATH=/wallaroo
export WANDB_MODE=disabled



python3 /wallaroo/scripts/evaluate/ar_infer_i2i_mar_imgedit.py \
    config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage4_7B_omini.yaml \
    pretrained_path=/path/to/checkpoints/checkpoint.ckpt \
    save_path=/edit_bench_output/wallaroo_7B_stage4_omini \
    cfg=3.0 \
    resolution=512 \
    image_token_count=1024