# ULM experiments

This repository reproduces the paper's three experiment groups—IFEval-Simple, FollowBench-derived, and CCT—for Phi-3 Mini, Qwen2.5-3B, and Llama-3.2-3B. The three matching model notebooks are in `notebooks/`.

## Setup

You need Python 3, internet access, and an NVIDIA GPU with BF16 support. Llama also needs an accepted Hugging Face license and `HF_TOKEN`.

```bash
python -m pip install -r requirements.txt
python run.py --model all --dry-run
```

## Reproduce the paper

Run all three models and create the paper configuration and compact result tables:

```bash
python run.py --model all --paper --workdir results
```

The summaries are written to `results/paper_results/table_2.csv`, `table_3.csv`, and `table_4.csv`. Finished steps are cached, so the same command resumes an interrupted run.

To rebuild these tables from completed runs without training again:

```bash
python paper_results.py results
```

To run one model, replace `all` with `phi3`, `qwen`, or `llama`.

## Datasets

CCT is included at `data/cct_benchmark.zip`. All 2,136 prompts are also available in a [browsable CSV](data/cct_benchmark_readable.csv), with a short [data guide](data/README.md).

The code automatically downloads the other public data when needed:

- [IFEval-Simple data](https://raw.githubusercontent.com/apple-aiml-research/ml-internal-llms-instruction-following/main/data/ifeval_simple.jsonl)
- [FollowBench data](https://github.com/YJiangcm/FollowBench/tree/master/data)
- [Official IFEval checker](https://github.com/google-research/google-research/tree/master/instruction_following_eval)

Manual downloads are not needed, but if you want to download and inspect them yourself, put IFEval-Simple at `<workdir>/IFEval-Simple/data/ifeval_simple.jsonl` and FollowBench files in `<workdir>/FollowBench-derived/<model-folder>/data/followbench_en/`. Pass a local IFEval checker with `--checker-dir PATH`.

## Useful checks

```bash
python run.py --list-stages
python run.py --model qwen --dry-run --paper
```
