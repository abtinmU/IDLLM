#!/usr/bin/env python3
"""Run the reformulation-transfer and CCT experiments for one or more models.

Replaces running notebooks/{phi3,qwen,llama}.ipynb by hand. Each model writes to
its own folder; finished runs are detected and replayed without loading the model.

examples:
  python run.py --model qwen --cct-zip data/cct_benchmark.zip
  python run.py --model all --cct-zip /abs/path/cct_benchmark.zip --workdir results
  HF_TOKEN=... python run.py --model llama --until ifeval-simple
  python run.py --model phi3 --dry-run
  python run.py --list-stages
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ulm.models import ALIASES, MODEL_SPECS, get_spec  # noqa: E402
from ulm.pipeline import STAGE_NAMES, STAGES  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                     epilog=__doc__.split('\n\n', 2)[2],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', nargs='+', metavar='NAME',
                        help='phi3, qwen, llama, or all; several models run one after another')
    parser.add_argument('--cct-zip', type=Path, help='cct_benchmark.zip (sets ULM_CCT_ZIP)')
    parser.add_argument('--workdir', type=Path,
                        help='output folder (sets ULM_WORKDIR); default results/ulm_<model>_server_results. '
                             'With several models, each gets a ulm_<model>_server_results sub-folder')
    parser.add_argument('--model-rev', help='Hugging Face commit to pin on a fresh run (sets ULM_MODEL_REV)')
    parser.add_argument('--checker-dir', type=Path,
                        help='local official IFEval checker directory (sets ULM_CHECKER_SOURCE_DIR)')
    parser.add_argument('--run-ifeval-derived', action='store_true',
                        help='also run Part II, the IFEval-derived study, which the paper excludes')
    parser.add_argument('--until', choices=STAGE_NAMES, help='stop after this stage')
    parser.add_argument('--dry-run', action='store_true', help='print the resolved plan and exit')
    parser.add_argument('--list-stages', action='store_true', help='list stages and exit')
    args = parser.parse_args(argv)
    if not args.list_stages and not args.model:
        parser.error('--model is required')
    return parser, args


def resolve_models(parser, names):
    if any(n.lower() == 'all' for n in names):
        return list(MODEL_SPECS.values())
    specs = []
    for name in names:
        if name.lower() not in ALIASES:
            parser.error(f'unknown model {name!r}; choose from {", ".join(sorted(ALIASES))} or all')
        spec = get_spec(name)
        if spec not in specs:
            specs.append(spec)
    return specs


def workdir_for(args, spec, several):
    if args.workdir is None:
        return None
    root = args.workdir.expanduser().resolve()
    return root / f'ulm_{spec.tag}_server_results' if several else root


def apply_environment(args, spec, several):
    workdir = workdir_for(args, spec, several)
    if workdir is not None:
        os.environ['ULM_WORKDIR'] = str(workdir)
    if args.cct_zip is not None:
        os.environ['ULM_CCT_ZIP'] = str(args.cct_zip.expanduser().resolve())
    if args.model_rev:
        os.environ['ULM_MODEL_REV'] = args.model_rev
    if args.checker_dir is not None:
        os.environ['ULM_CHECKER_SOURCE_DIR'] = str(args.checker_dir.expanduser().resolve())


def print_plan(args, spec, several):
    workdir = workdir_for(args, spec, several)
    stages = list(STAGE_NAMES[:STAGE_NAMES.index(args.until) + 1] if args.until else STAGE_NAMES)
    if not args.run_ifeval_derived and 'ifeval-derived' in stages:
        stages[stages.index('ifeval-derived')] = 'ifeval-derived (skipped)'
    print(f'model        {spec.display_name} ({spec.model_id})')
    print(f'revision     {spec.pinned_revision or args.model_rev or "first resolved commit, then frozen"}')
    print(f'layers       {spec.layers} primary {spec.primary_layer} | width {spec.hidden_width}')
    print(f'workdir      {workdir or os.environ.get("ULM_WORKDIR") or f"results/ulm_{spec.tag}_server_results"}')
    print(f'cct zip      {args.cct_zip or os.environ.get("ULM_CCT_ZIP") or "data/cct_benchmark.zip (auto)"}')
    print(f'stages       {" -> ".join(stages)}')
    if spec.gated and not os.environ.get('HF_TOKEN'):
        print('note         gated checkpoint: accept the licence on Hugging Face and set HF_TOKEN')


def main(argv=None):
    parser, args = parse_args(argv)
    if args.list_stages:
        for name, filename, description in STAGES:
            print(f'{name:16} {filename:24} {description}')
        return 0

    specs = resolve_models(parser, args.model)
    several = len(specs) > 1
    if several and args.model_rev:
        parser.error('--model-rev applies to a single model')

    if args.dry_run:
        for spec in specs:
            print_plan(args, spec, several)
            print()
        return 0

    if several:
        # Run each model in a separate process.
        for spec in specs:
            print(f'\n######## {spec.display_name} ########', flush=True)
            code = subprocess.call(child_command(args, spec))
            if code != 0:
                print(f'{spec.display_name} failed with exit code {code}; stopping.', file=sys.stderr)
                return code
        return 0

    spec = specs[0]
    apply_environment(args, spec, several)
    if spec.gated and not os.environ.get('HF_TOKEN'):
        print('Note: gated checkpoint; set HF_TOKEN unless the Hugging Face cache is already authenticated.')
    from ulm import pipeline
    pipeline.run(spec, until=args.until, run_ifeval_derived=args.run_ifeval_derived)
    return 0


def child_command(args, spec):
    """Command line that runs a single model of a multi-model invocation."""
    command = [sys.executable, str(Path(__file__).resolve()), '--model', spec.tag]
    workdir = workdir_for(args, spec, several=True)
    if workdir is not None:
        command += ['--workdir', str(workdir)]
    if args.cct_zip is not None:
        command += ['--cct-zip', str(args.cct_zip.expanduser().resolve())]
    if args.checker_dir is not None:
        command += ['--checker-dir', str(args.checker_dir.expanduser().resolve())]
    if args.run_ifeval_derived:
        command.append('--run-ifeval-derived')
    if args.until:
        command += ['--until', args.until]
    return command


if __name__ == '__main__':
    sys.exit(main())
