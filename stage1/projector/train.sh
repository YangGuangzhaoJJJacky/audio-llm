python projector.py fit \
    --model.speech_encoder_path "openai/whisper-large-v3" \
    --model.llm_path "mistralai/Mistral-7B-Instruct-v0.3" \
    --model.dataset_path "japanese-asr/ja_asr.jsut_basic5000" \
    --trainer.accelerator "auto" \
    --trainer.devices "auto" \
    --trainer.precision "bf16-true" \
    --trainer.strategy "auto" \
    --trainer.max_epochs 50 \
    --trainer.log_every_n_steps 100 \
    --trainer.val_check_interval=10000 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.monitor="val_wer" \
    --trainer.callbacks.save_top_k=5 \
    --trainer.callbacks.mode="min" \
    --trainer.callbacks.auto_insert_metric_name=True \
    --trainer.callbacks+=LearningRateMonitor \
    --trainer.callbacks+=DeviceStatsMonitor \
    --trainer.profiler=SimpleProfiler \
    --model.batch_size=1 \
    --trainer.accumulate_grad_batches 4 \
    | tee train.log



