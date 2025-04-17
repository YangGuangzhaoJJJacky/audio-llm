# End2End

## Environment Setup

### 1. Conda Environment

To create and activate the Conda environment:

```bash
conda create -n projector python=3.12 -y
conda activate projector
```

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2. [Optional] Dependency Management with `uv`

If you prefer using [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

Make sure you have `requirements.txt` in the root directory.

---

## Running the Project

### 1. Download the LLM

Download **Qwen2.5-0.5B-Instruct** model manually and place it in the `pretrained_models` directory:

```
pretrained_models/Qwen2.5-0.5B-Instruct
```

> Make sure it includes `config.json`, `tokenizer.model`, and `pytorch_model.bin`.

---

### 2. Train the End-to-End Speech-to-Text Model

To start training, simply run:

```bash
bash train/train.sh
```

You can customize training configurations inside `train/train.sh` or directly pass arguments to `train/projector.py`.

