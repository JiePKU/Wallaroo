export PYTHONPATH=/wallaroo
export WANDB_MODE=disabled



/workdir/conda_envs/wallaroo/bin/python3 /wallaroo/scripts/evaluate/ar_infer_t2i_geneval_mar.py \
    config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage4_7B_omini.yaml \
    pretrained_path=/path/to/checkpoints/checkpoint.ckpt \
    save_path=/evaluate_output/h100_stage4_7B_omini_geneval \
    cfg=3 \
    resolution=512 \
    image_token_count=1024 \
    hw_indicator=True \


sleep 5s



/workdir/conda_envs/wallaroo/bin/python3 /wallaroo/scripts/evaluate/ar_infer_t2i_dpg_mar.py \
    config=/wallaroo/examples/wallaroo/ar_wallaroo_7/h100_stage4_7B_omini.yaml \
    pretrained_path=/path/to/checkpoints/checkpoint.ckpt \
    save_path=/evaluate_output/h100_stage4_7B_omini_dpg \
    cfg=4 \
    resolution=512 \
    image_token_count=1024 \
    hw_indicator=True \