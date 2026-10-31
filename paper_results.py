#!/usr/bin/env python3
"""Create the compact paper tables from one or more completed model runs."""
import argparse
import json
from pathlib import Path

import pandas as pd


SHORT_NAMES = {
    'microsoft/Phi-3-mini-128k-instruct': 'Phi-3',
    'Qwen/Qwen2.5-3B-Instruct': 'Qwen',
    'meta-llama/Llama-3.2-3B-Instruct': 'Llama',
}


def _run_roots(root):
    if (root / 'final_tables').is_dir():
        return [root]
    return sorted({path.parent.parent for path in root.rglob('final_tables/paper_table_main_AUROC.csv')})


def _model_name(run):
    manifest_path = run / 'RUN_MANIFEST.json'
    if manifest_path.is_file():
        model = json.loads(manifest_path.read_text(encoding='utf-8'))['checkpoint']
    else:
        model = json.loads((run / 'COMPLETED_RUN.json').read_text(encoding='utf-8'))['model']
    return SHORT_NAMES.get(model, model)


def _table_2(run, model):
    auc = pd.read_csv(run / 'final_tables' / 'paper_table_main_AUROC.csv')
    flips = pd.read_csv(run / 'final_tables' / 'paper_table_paired_flip_diagnostic.csv')
    rows = []
    for setting in ('IFEval-Simple', 'FollowBench-derived'):
        part = auc[auc.setting.eq(setting)]
        values = {(row.domain, row.method): 100 * row.AUROC_macro
                  for row in part.itertuples()}
        changed = flips[(flips.setting.eq(setting)) & (flips.method.eq('MT2'))]
        flip_acc = 100 * changed.flip_direction_correct.sum() / changed.changed_pairs.sum()
        rows.append({'model': model, 'dataset': setting,
                     'original_shared': values[('original', 'B0')],
                     'reform_shared': values[('held_out', 'B0')],
                     'reform_aug_shared': values[('held_out', 'B2')],
                     'reform_type_aware': values[('held_out', 'MT0')],
                     'reform_aug_type_aware': values[('held_out', 'MT2')],
                     'flip_accuracy': flip_acc})
    return rows


def _table_3(run, model):
    pairs = pd.read_csv(run / 'CCT_independent_refit' / 'paired_2x2_changes.csv')
    rows = []
    factors = {
        'Wording': ('wording_original_rule', 'wording_changed_rule'),
        'Requirement': ('requirement_original_wording', 'requirement_reworded'),
    }
    for factor, contrasts in factors.items():
        flip = sum(pairs[f'{name}_y_diff'].abs().mean() for name in contrasts) / len(contrasts)
        change = sum(pairs[f'{name}_MT2_score_diff'].abs().mean() for name in contrasts) / len(contrasts)
        rows.append({'model': model, 'factor': factor,
                     'behavior_flip_percent': 100 * flip,
                     'mean_abs_probability_change': change})
    return rows


def _table_4(run, model):
    matches = list(run.glob('FollowBench-derived/*/results/H2_bootstrap.csv'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one FollowBench H2 table under {run}')
    h2 = pd.read_csv(matches[0])
    h2 = h2[(h2.domain.eq('OOD')) & (h2.method.eq('MT2')) & h2.baseline.isin(['B0', 'B2'])]
    return [{'model': model, 'comparison': f'MT2 - {row.baseline}',
             'gain_percent': 100 * row.gain, 'ci_low_percent': 100 * row.ci_low,
             'ci_high_percent': 100 * row.ci_high} for row in h2.itertuples()]


def build_tables(root):
    runs = _run_roots(Path(root))
    if not runs:
        raise FileNotFoundError(f'No completed run found under {root}')
    output = Path(root) / 'paper_results' if len(runs) > 1 else runs[0] / 'paper_results'
    output.mkdir(parents=True, exist_ok=True)
    tables = {2: [], 3: [], 4: []}
    for run in runs:
        model = _model_name(run)
        tables[2].extend(_table_2(run, model))
        tables[3].extend(_table_3(run, model))
        tables[4].extend(_table_4(run, model))
    for number, rows in tables.items():
        frame = pd.DataFrame(rows)
        frame.to_csv(output / f'table_{number}.csv', index=False, float_format='%.6f')
        print(f'\nPaper Table {number}\n{frame.to_string(index=False, float_format=lambda x: f"{x:.2f}")}')
    print(f'\nSaved compact paper tables to {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='one model run or a parent results folder')
    build_tables(parser.parse_args().results.resolve())
