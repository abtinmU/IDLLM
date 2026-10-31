#!/usr/bin/env python3
"""Export the canonical CCT ZIP as one human-readable CSV."""
import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


CONDITIONS = {
    'O': ('original', 'canonical'),
    'R1': ('original', 'reformulated'),
    'C0': ('changed', 'canonical'),
    'C1': ('changed', 'reformulated'),
}

FIELDS = (
    'id', 'section', 'task_id', 'task_family_id', 'cct_fold', 'source_dataset',
    'constraint_type', 'condition', 'requirement', 'wording', 'target_requirement',
    'target_kwargs', 'base_task', 'prompt', 'prompt_sha256', 'generation_budget',
    'max_input_tokens',
)


def read_jsonl(archive, name):
    return [json.loads(line) for line in archive.read(name).decode('utf-8').splitlines() if line]


def requirement_text(constraint_type, kwargs):
    if constraint_type == 'startend:end_checker':
        return f'End with the exact phrase: {kwargs["end_phrase"]}'
    if constraint_type == 'detectable_content:number_placeholders':
        return f'Use at least {kwargs["num_placeholders"]} square-bracket placeholders'
    raise ValueError(f'Unknown constraint type: {constraint_type}')


def build_rows(zip_path):
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None, 'Corrupted CCT ZIP'
        declared = json.loads(archive.read('SHA256SUMS.json'))
        assert all(hashlib.sha256(archive.read(name)).hexdigest() == expected
                   for name, expected in declared.items()), 'CCT member hash mismatch'

        for section in 'ABCD':
            families = read_jsonl(archive, f'sections/{section}/section_{section}_families.jsonl')
            manifests = read_jsonl(archive, f'sections/{section}/section_{section}_C0_C1_prompts.jsonl')
            metadata = {int(item['task_id']): item for item in manifests}
            for family in families:
                task_id = int(family['task_id'])
                example = metadata[task_id]
                constraint_type = family.get('source_type', family.get('constraint_type'))
                for condition, (requirement, wording) in CONDITIONS.items():
                    target_kwargs = (family['original_kwargs'] if condition in ('O', 'R1')
                                     else family['counterfactual_kwargs'])
                    prompt = family[f'{condition}_prompt']
                    rows.append({
                        'id': f'CCT_{section}_{task_id:03d}_{condition}',
                        'section': section,
                        'task_id': task_id,
                        'task_family_id': family['task_family_id'],
                        'cct_fold': int(family['cct_fold']),
                        'source_dataset': ('IFEval-Simple' if section in 'AB'
                                           else 'IFEval (CCT source)'),
                        'constraint_type': constraint_type,
                        'condition': condition,
                        'requirement': requirement,
                        'wording': wording,
                        'target_requirement': requirement_text(constraint_type, target_kwargs),
                        'target_kwargs': json.dumps(target_kwargs, ensure_ascii=False, sort_keys=True),
                        'base_task': family['base_task'],
                        'prompt': prompt,
                        'prompt_sha256': hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                        'generation_budget': int(example['generation_budget']),
                        'max_input_tokens': int(example.get('max_input_tokens', 1024)),
                    })
    assert len(rows) == 2136
    assert len({row['id'] for row in rows}) == 2136
    assert len({row['task_family_id'] for row in rows}) == 282
    return rows


def export(zip_path, output_path):
    rows = build_rows(zip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} prompts to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, default=Path('data/cct_benchmark.zip'))
    parser.add_argument('--output', type=Path, default=Path('data/cct_benchmark_readable.csv'))
    args = parser.parse_args()
    export(args.zip.resolve(), args.output.resolve())
