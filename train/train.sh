python train/projector.py fit \
    --model.speech_encoder_path "openai/whisper-large-v3" \
    --model.checkpoint_path "lightning_logs/version_9/checkpoints/epoch=2-step=1875.ckpt" \
    --model.llm_path "pretrained_models/Qwen2.5-0.5B-Instruct" \
    --data.batch_size=2 \
    --data.num_workers=8 \
    --trainer.accelerator "auto" \
    --trainer.devices "auto" \
    --trainer.precision "bf16-true" \
    --trainer.strategy "auto" \
    --trainer.max_epochs 50 \
    --trainer.log_every_n_steps 10 \
    --trainer.val_check_interval=2500 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.monitor="val_cer" \
    --trainer.callbacks.save_top_k=5 \
    --trainer.callbacks.mode="min" \
    --trainer.callbacks.auto_insert_metric_name=True \
    --trainer.callbacks+=LearningRateMonitor \
    --trainer.callbacks+=DeviceStatsMonitor \
    --trainer.profiler=SimpleProfiler \
    --trainer.accumulate_grad_batches 4 \
    --trainer.num_sanity_val_steps 1 \
    #2> error.log 1> train.log 



