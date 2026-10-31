# ULM experiment code

This folder runs the ULM experiments for three models:

- Phi-3 Mini
- Qwen2.5-3B
- Llama-3.2-3B

The same experiments are also available as notebooks in `notebooks/`.

## What is needed

- Python 3
- An NVIDIA GPU with BF16 support, such as an A100 or H100
- Internet access to GitHub and Hugging Face
- Hugging Face access and `HF_TOKEN` for Llama

The CCT dataset is included at `data/cct_benchmark.zip`. The code finds it automatically.

## Install

```bash
python -m pip install -r requirements.txt
```

## Check the setup

```bash
python run.py --list-stages
python run.py --model all --dry-run
```

## Run one model

```bash
python run.py --model qwen
```

Use `phi3`, `qwen`, or `llama` as the model name.

## Run all models

```bash
python run.py --model all --workdir results
```

Each model gets its own output folder. Finished steps are cached, so an interrupted run can continue later.

The IFEval-derived stage is off by default. Add `--run-ifeval-derived` only if you need it.
