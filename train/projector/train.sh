python train/projector.py fit \
    --model.speech_encoder_path "openai/whisper-large-v3" \
    --model.checkpoint_path ""  \
    --model.llm_path "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data.batch_size=16 \
    --data.num_workers=8 \
    --trainer.accelerator "auto" \
    --trainer.devices "auto" \
    --trainer.precision "bf16-true" \
    --trainer.strategy "auto" \
    --trainer.max_epochs 50 \
    --trainer.log_every_n_steps 10 \
    --trainer.val_check_interval=500 \
    --trainer.callbacks+=ModelCheckpoint \
    --trainer.callbacks.monitor="train_loss" \
    --trainer.callbacks.save_top_k=50 \
    --trainer.callbacks.mode="min" \
    --trainer.callbacks.auto_insert_metric_name=True \
    --trainer.callbacks+=LearningRateMonitor \
    --trainer.callbacks+=DeviceStatsMonitor \
    --trainer.profiler=SimpleProfiler \
    --trainer.accumulate_grad_batches 1 \
    --trainer.num_sanity_val_steps 1 \
    #2> error.log 1> train.log 



