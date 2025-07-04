export HF_HOME=/mnt/data-raid/yangguangzhao/.cache/huggingface
export CUDA_VISIBLE_DEVICES=1

python slam_asr.py fit \
    --trainer.accelerator "auto" \
    --trainer.devices "auto" \
    --trainer.precision "bf16-true" \
    --trainer.strategy "auto" \
    --trainer.max_epochs 50 \
    --trainer.log_every_n_steps 20 \
    --trainer.val_check_interval=5000 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.monitor="val_loss" \
    --trainer.callbacks.save_top_k=10 \
    --trainer.callbacks.mode="min" \
    --trainer.callbacks.auto_insert_metric_name=True \
    --trainer.callbacks+=LearningRateMonitor \
    --trainer.callbacks+=DeviceStatsMonitor \
    --trainer.profiler=SimpleProfiler \
    --trainer.num_sanity_val_steps=1 \
    --model.batch_size=8 \
    --trainer.accumulate_grad_batches 8