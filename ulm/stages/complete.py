"""Write the verified complete-run marker used for fast replay.

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""






if not REPLAY_COMPLETE:
    # Write the completion marker after all required outputs pass.
    import hashlib,json,os
    from pathlib import Path
    _all_in_root=Path(ULM_WORKDIR)
    _completed_required=[
     'final_tables/paper_table_main_AUROC.csv',
     'final_tables/paper_table_main_comparisons.csv',
     'final_tables/all_out_of_fold_predictions.csv',
     'CCT_independent_refit/all_four_condition_oof_predictions.csv',
     'CCT_independent_refit/paired_2x2_changes.csv',
     'CCT_independent_refit/pairwise_flip_direction_clustered_CI.csv',
     'CCT_independent_refit/frozen_protocol.json',
     'CCT_independent_refit/probe_tuning_from_O_R1_only.csv',
     'RUN_MANIFEST.json'
    ]
    for _r in _completed_required:
        assert (_all_in_root/_r).is_file() and (_all_in_root/_r).stat().st_size>0, 'Cannot mark complete: '+_r
    _complete_files={}
    for _p in sorted(_all_in_root.rglob('*')):
        if not _p.is_file() or _p.name in ('COMPLETED_RUN.json','COMPLETED_RUN.json.tmp'):continue
        _rel=str(_p.relative_to(_all_in_root))
        _stat=_p.stat()
        _entry={'size':_stat.st_size}
        
        
        
        if (_rel.startswith('final_tables/') or _rel.startswith('CCT_independent_refit/')
            and not ('/response_cache/' in _rel or '/activation_cache/' in _rel or '/fold_probes/' in _rel)
            or _p.suffix=='.json' and '/cache/' not in _rel
            or _p.suffix=='.npz' and ('/probes/' in _rel or '/fold_probes/' in _rel)):
            _entry['sha256']=_fingerprint_file(_p)
        _complete_files[_rel]=_entry
    assert len(_complete_files)>40, 'Completed run is missing most of its saved intermediate files'
    _complete={'version':'verified-v2','model':_MODEL_LABEL,'notebook_sha256':_CODE_RELEASE_SHA,
     'cct_sha256':_fingerprint_file(_ULM_DATA),'files':_complete_files,
     'n_files':len(_complete_files),'completion_requires':_completed_required}
    _tmp=_RUN_MARKER.with_suffix('.json.tmp')
    _tmp.write_text(json.dumps(_complete,indent=2,sort_keys=True),encoding='utf-8')
    os.replace(_tmp,_RUN_MARKER)
    print('VERIFIED RUN COMPLETED:',len(_complete_files),'saved files; subsequent Run All uses fast replay.')
