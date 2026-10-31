# Experiment data

The CCT data is included in this directory. The other public datasets are downloaded automatically when a run needs them.

## CCT

`cct_benchmark.zip` is the canonical experiment input. Keep this file when running the code.

`cct_benchmark_readable.csv` contains the same 534 task-constraint instances and all 2,136 prompts in one file. It is intended for browsing and analysis. Multiline prompts are stored as quoted CSV fields.

The four conditions are:

- `O`: original requirement, canonical wording
- `R1`: original requirement, reformulated wording
- `C0`: changed requirement, canonical wording
- `C1`: changed requirement, reformulated wording

The CSV also includes each base task, target requirement, source dataset, task family, and frozen cross-validation fold. Rebuild it from the canonical ZIP with:

```bash
python tools/export_cct_readable.py
```

## IFEval-Simple

IFEval-Simple supplies the canonical and reformulated prompts used in the main formulation-transfer experiment.

- Source: [IFEval-Simple JSONL](https://raw.githubusercontent.com/apple-aiml-research/ml-internal-llms-instruction-following/main/data/ifeval_simple.jsonl)
- Download: automatic
- Local path after download: `<workdir>/IFEval-Simple/data/ifeval_simple.jsonl`

## FollowBench

FollowBench supplies the source instructions for the controlled FollowBench-derived experiment. The code downloads the six public constraint categories: content, situation, style, format, example, and mixed. The experiment evaluates the added deterministic constraints; it does not report FollowBench's native joint score.

- Source: [FollowBench data](https://github.com/YJiangcm/FollowBench/tree/master/data)
- Backup source: [FollowBench on Hugging Face](https://huggingface.co/datasets/YuxinJiang/FollowBench)
- Download: automatic
- Local path after download: `<workdir>/FollowBench-derived/<model-folder>/data/followbench_en/`

## Original IFEval

The original 541-example IFEval file is used only by the optional IFEval-derived study. Paper reproduction does not need this stage.

- Source: [Original IFEval JSONL](https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl)
- Download: automatic when `--run-ifeval-derived` is used
- Local filename after download: `original_ifeval_541.jsonl` inside the model's IFEval-derived data directory

## Official IFEval checker

The experiments use the official IFEval checker to assign deterministic compliance labels. The required checker files are downloaded and verified by SHA-256. They are code, not another dataset.

- Source: [Google Research IFEval checker](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- Download: automatic
- Optional local copy: pass its `instruction_following_eval` directory with `--checker-dir PATH`
