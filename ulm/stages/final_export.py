"""Part IV. Unified final-paper results export.

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""








if not REPLAY_COMPLETE:
    # Merge validated outputs into the final tables.
    from pathlib import Path
    import pandas as pd, numpy as np, json
    from sklearn.metrics import roc_auc_score
    FINAL_ROOT = ULM_WORKDIR/'final_tables'
    FINAL_ROOT.mkdir(parents=True, exist_ok=True)
    METHOD_ORDER = ['B0','B2','MT0','MT2']

    
    simple_p = pd.read_csv(Path(SIMPLE_RESULTS)/'oof_original_and_R_predictions.csv')
    ifeval_p = pd.read_csv(Path(IFEVAL_RESULTS)/'predictions_all_forms.csv') if RUN_IFEVAL_DERIVED else None
    follow_p = pd.read_csv(Path(FOLLOWBENCH_RESULTS)/'oof_predictions.csv')

    def normalized(frame, setting):
        f=frame.copy()
        if setting=='IFEval-Simple':
            f=f.rename(columns={'task_id':'case'})
            f['family']=f['case']
        elif setting=='IFEval-derived':
            f=f.rename(columns={'group':'family','key':'case'})
        else:
            f=f.rename(columns={'group':'family','key':'case'})
        f['setting']=setting
        f['domain']=np.where(f['variant']=='original','original','held_out')
        assert f['variant'].isin(['original','R1','R2']).all()
        assert all(c in f.columns for c in ['setting','case','family','type','variant','label']+METHOD_ORDER)
        f['family']=f['family'].astype(str)
        f['case']=f['case'].astype(str)
        cols=['setting','case','family','type','variant','domain','label']+METHOD_ORDER
        for optional in ['fold','capped','output_tokens','type_only']:
            if optional in f.columns:cols.append(optional)
        return f[cols].copy()

    simple_p=simple_p[simple_p.variant.isin(['original','R1','R2'])].copy()
    if RUN_IFEVAL_DERIVED:ifeval_p=ifeval_p[ifeval_p.variant.isin(['original','R1','R2'])].copy()
    follow_p=follow_p[follow_p.variant.isin(['original','R1','R2'])].copy()
    
    ALL=pd.concat([normalized(simple_p,'IFEval-Simple')]+([normalized(ifeval_p,'IFEval-derived')] if RUN_IFEVAL_DERIVED else [])+
                  [normalized(follow_p,'FollowBench-derived')],ignore_index=True)
    assert ALL.setting.nunique()==(3 if RUN_IFEVAL_DERIVED else 2)
    assert not ALL.duplicated(['setting','case','type','variant']).any(), 'Duplicate example/type/formulation; inspect source grouping.'
    assert ALL.groupby(['setting','case','type']).variant.nunique().eq(3).all(), 'Incomplete original/R1/R2 task pairs.'
    ALL.to_csv(FINAL_ROOT/'all_out_of_fold_predictions.csv',index=False)
    print('Verified out-of-fold predictions',len(ALL),'across',ALL.setting.nunique(),'settings')

    def safe_auc(y,score):
        y=np.asarray(y,int);p=np.asarray(score,float)
        return float(roc_auc_score(y,p)) if np.unique(y).size==2 else np.nan

    summary=[]
    within=[]
    for (setting,domain),frame in ALL.groupby(['setting','domain']):
        for method in METHOD_ORDER:
            cells=[]
            for (typ,variant),g in frame.groupby(['type','variant']):
                auc=safe_auc(g.label,g[method]);cells.append(auc)
                within.append({'setting':setting,'domain':domain,'type':typ,'formulation':variant,
                              'method':method,'n':len(g),'positive':int(g.label.sum()),
                              'negative':int(len(g)-g.label.sum()),'AUROC':auc})
            values=np.array(cells,float)
            summary.append({'setting':setting,'domain':domain,'method':method,
                            'n':int(len(frame)),'n_families':int(frame.family.nunique()),
                            'AUROC_macro':float(np.nanmean(values)),
                            'AUROC_pooled_descriptive':safe_auc(frame.label,frame[method]),
                            'valid_cells':int(np.isfinite(values).sum()),'total_cells':len(values)})
    summary=pd.DataFrame(summary);within=pd.DataFrame(within)
    summary.to_csv(FINAL_ROOT/'paper_table_main_AUROC.csv',index=False)
    within.to_csv(FINAL_ROOT/'paper_table_within_type_AUROC.csv',index=False)

    
    wide=summary.pivot(index='setting',columns=['domain','method'],values='AUROC_macro')
    changes=[]
    for setting in wide.index:
        changes.append({'setting':setting,
          'B0_original':100*wide.loc[setting,('original','B0')],
          'B0_heldout':100*wide.loc[setting,('held_out','B0')],
          'B0_transfer_delta_pp':100*(wide.loc[setting,('held_out','B0')]-wide.loc[setting,('original','B0')]),
          'B2_heldout':100*wide.loc[setting,('held_out','B2')],
          'MT0_heldout':100*wide.loc[setting,('held_out','MT0')],
          'MT2_heldout':100*wide.loc[setting,('held_out','MT2')],
          'MT2_minus_B2_pp':100*(wide.loc[setting,('held_out','MT2')]-wide.loc[setting,('held_out','B2')]),
          'MT2_minus_B0_original_pp':100*(wide.loc[setting,('original','MT2')]-wide.loc[setting,('original','B0')])})
    main_table=pd.DataFrame(changes)
    main_table.to_csv(FINAL_ROOT/'paper_table_main_comparisons.csv',index=False)
    display(main_table.round(2))







if not REPLAY_COMPLETE:
    from scipy.stats import binomtest
    pairs=[]
    for setting,df in ALL.groupby('setting'):
        anchor=df[df.variant.eq('original')].set_index(['case','type'])
        for form in ['R1','R2']:
            reform=df[df.variant.eq(form)].set_index(['case','type'])
            assert anchor.index.is_unique and reform.index.is_unique
            assert set(anchor.index)==set(reform.index)
            joined=anchor[['label']+METHOD_ORDER].join(reform[['label']+METHOD_ORDER],lsuffix='_O',rsuffix='_R',validate='one_to_one')
            for m in METHOD_ORDER:
                y0=joined.label_O.to_numpy(int);y1=joined.label_R.to_numpy(int)
                delta=y1-y0; ds=joined[f'{m}_R'].to_numpy(float)-joined[f'{m}_O'].to_numpy(float)
                flips=delta!=0; n=int(flips.sum());correct=int((ds[flips]*delta[flips]>0).sum())
                ties=int((ds[flips]==0).sum());denom=n 
                pairs.append({'setting':setting,'formulation':form,'method':m,
                  'matched_pairs':len(joined),'changed_pairs':n,'unchanged_pairs':int((~flips).sum()),
                  'failure_to_success':int((delta==1).sum()),'success_to_failure':int((delta==-1).sum()),
                  'flip_direction_correct':correct,'score_ties_on_flips':ties,
                  'flip_direction_accuracy':float(correct/denom) if denom else np.nan,
                  'flip_fraction':float(n/len(joined))})
    PAIR=pd.DataFrame(pairs)
    
    
    BOOT_N = 2000
    flip_uncertainty=[]
    for setting,df in ALL.groupby('setting'):
        families=sorted(df.family.unique())
        ix={fam:i for i,fam in enumerate(families)}
        rng=np.random.default_rng(20261017+len(setting))
        weights=rng.multinomial(len(families),np.full(len(families),1/len(families)),size=BOOT_N)
        a=df[df.variant.eq('original')].set_index(['case','type'])
        for form in ['R1','R2']:
            b=df[df.variant.eq(form)].set_index(['case','type'])
            z=a[['family','label']+METHOD_ORDER].join(b[['label']+METHOD_ORDER],rsuffix='_R',validate='one_to_one')
            dy=z.label_R.to_numpy(int)-z.label.to_numpy(int)
            valid=dy!=0
            fg=np.array([ix[fam] for fam in z.family],int)
            counts=np.bincount(fg,weights=valid.astype(float),minlength=len(families))
            n_flips=int(valid.sum());up=int((dy==1).sum());down=int((dy==-1).sum())
            for method in METHOD_ORDER:
                diff=z[f'{method}_R'].to_numpy(float)-z[method].to_numpy(float)
                success=(diff*dy>0)&valid
                correct_by_family=np.bincount(fg,weights=success.astype(float),minlength=len(families))
                numer=weights@correct_by_family; denom=weights@counts
                samples=np.divide(numer,denom,out=np.full(BOOT_N,np.nan),where=denom>0)
                good=samples[np.isfinite(samples)]
                flip_uncertainty.append({'setting':setting,'formulation':form,'method':method,
                    'n_flips':n_flips,'flip_direction_accuracy':float(success.sum()/n_flips) if n_flips else np.nan,
                    'ci_low':float(np.quantile(good,0.025)) if len(good) else np.nan,
                    'ci_high':float(np.quantile(good,0.975)) if len(good) else np.nan,
                    'n_valid_bootstraps':int(len(good)),
                    
                    'majority_flip_direction_oracle':float(max(up,down)/n_flips) if n_flips else np.nan})
    PAIR_CI=pd.DataFrame(flip_uncertainty)
    PAIR_CI.to_csv(FINAL_ROOT/'paper_table_paired_flip_uncertainty.csv',index=False)
    PAIR.to_csv(FINAL_ROOT/'paper_table_paired_flip_diagnostic.csv',index=False)
    display(PAIR[PAIR.method.isin(['B0','MT2'])][['setting','formulation','method','changed_pairs',
       'flip_direction_accuracy','flip_fraction']].round(3))







if not REPLAY_COMPLETE:
    simple_source=Path(SIMPLE_RESULTS).parent/'all_1836_scored.jsonl'
    assert simple_source.is_file(), f'Missing frozen IFEval-Simple generation records: {simple_source}'
    ss=pd.read_json(simple_source,lines=True)
    orig=ss[ss.variant.eq('original')][['task_id','instruction_type','prompt','kwargs']]
    orig=orig.rename(columns={'prompt':'canonical_prompt','kwargs':'canonical_rule_kwargs'})
    rw=ss[ss.variant.isin(['R1','R2'])][['task_id','instruction_type','variant','prompt','kwargs']]
    review=rw.merge(orig,on=['task_id','instruction_type'],how='left',validate='many_to_one')
    assert len(review)==102*3*2 and review.canonical_prompt.notna().all()
    review['known_issue']=np.where((review.instruction_type=='detectable_content:number_placeholders')&
                                   (review.variant=='R1'),
                       'R1 uses nonempty placeholder wording, but the original verifier may count []', '')
    review['manual_semantic_validated']='NOT_REVIEWED'
    review.to_csv(FINAL_ROOT/'ifeval_simple_semantic_review_queue.csv',index=False)
    print('IFEval-Simple reformulations awaiting semantic review:',len(review))
    print('Known wording/verifier mismatch cases:',int(review.known_issue.ne('').sum()))







if not REPLAY_COMPLETE:
    import shutil, hashlib, zipfile, os
    required={
      'IFEval-Simple':{
        'dir':Path(SIMPLE_RESULTS),
        'files':['H2_paired_OOD_AUROC_bootstrap.csv','original_domain_retention.csv',
                 'flip_prediction_diagnostic.csv','char_tfidf_control.csv','per_type_formulation_metrics.csv']},
      'IFEval-derived':{
        'dir':Path(IFEVAL_RESULTS) if RUN_IFEVAL_DERIVED else None,
        'files':['H2_bootstrap_and_ID_retention.csv','OOD_controls.csv',
                 'behavioral_flips.csv','per_type_and_formulation.csv']},
      'FollowBench-derived':{
        'dir':Path(FOLLOWBENCH_RESULTS),
        'files':['H2_bootstrap.csv','text_controls_metrics.csv','OOD_class_support.csv',
                 'behavioral_flip_tracking.csv','per_type_formulation.csv']}}
    if not RUN_IFEVAL_DERIVED:required.pop('IFEval-derived')
    for name,info in required.items():
        missing=[fname for fname in info['files'] if not (info['dir']/fname).is_file()]
        assert not missing, f'{name}: expected source statistical outputs are missing: {missing}'

    audit={
     'model':CFG['model_id'],
     'frozen_revision':MODEL_REV,
     'skipped_studies':[] if RUN_IFEVAL_DERIVED else ['IFEval-derived'],
     'source_dirs':{'simple':SIMPLE_RESULTS,'ifeval':IFEVAL_RESULTS,'followbench':FOLLOWBENCH_RESULTS},
     'main_research_questions':['formulation_transfer','type_aware_vs_equally_augmented_shared',
                                'within_pair_predictive_reliability'],
     'protocol_cautions':['IFEval-derived and FollowBench-derived labels are for added verifiable rules',
     'Only fixed, official checker labels; manual semantic review must be completed for publication',
     'Looped-model extensions, B3 and extra layer searches are excluded; CCT is a separate group-refit validation section'],
     'n_matched_rows':int(len(ALL)),
     'sha256':{}
    }
    for f in sorted(FINAL_ROOT.glob('*.csv')):
        audit['sha256'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
    (FINAL_ROOT/'FINAL_PROTOCOL_AND_FILE_MANIFEST.json').write_text(json.dumps(audit,indent=2))
    archive=FINAL_ROOT/(SPEC.camel+'_Final_Paper_Tables_and_Audits.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(FINAL_ROOT.glob('*')):
            if f.is_file() and f != archive:z.write(f,arcname=f.name)
        for name,info in required.items():
            for fname in info['files']:
                z.write(info['dir']/fname,arcname=f'source_audits/{name}/{fname}')
        
        for reviewpath in ([Path(IFEVAL_RESULTS).parent/'data'/'rewrite_review_all.csv'] if RUN_IFEVAL_DERIVED else [])+[
                           Path(FOLLOWBENCH_RESULTS).parent/'data'/'formulation_semantic_review.csv']:
            if reviewpath.is_file():z.write(reviewpath,arcname=f'semantic_review/{reviewpath.parent.parent.name}/{reviewpath.name}')
    print('Final paper files:',FINAL_ROOT)
    print('Portable evidence archive:',archive)
    print('IMPORTANT: Perform and document human review of flagged reformulations before reporting them as validated.')
