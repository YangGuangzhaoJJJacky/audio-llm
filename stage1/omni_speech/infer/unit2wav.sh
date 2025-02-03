#!/bin/bash

ROOT=omni_speech/infer/unit2wav_samples

VOCODER_CKPT=vocoder/g_00500000
VOCODER_CFG=vocoder/config.json

python omni_speech/infer/convert_jsonl_to_txt.py $ROOT/answer.json $ROOT/answer.unit
python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
    --in-code-file $ROOT/answer.unit \
    --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
    --results-path $ROOT/answer_wav/ --dur-prediction