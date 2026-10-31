"""Part V. Counterfactual Compliance Transfer (CCT): 2 x 2 wording/requirement diagnostic.

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""











if not REPLAY_COMPLETE:
    # Render the model-specific CCT helper.
    from ulm.cct import render_common
    CCT_MODULE_DIR=ULM_WORKDIR/'cct_modules'
    CCT_MODULE_DIR.mkdir(parents=True,exist_ok=True)
    (CCT_MODULE_DIR/'cct_common.py').write_text(render_common(SPEC,MODEL_REV))


if not REPLAY_COMPLETE:
    
    from ulm.cct import render_inference
    CCT_MODULE_DIR=ULM_WORKDIR/'cct_modules'
    CCT_MODULE_DIR.mkdir(parents=True,exist_ok=True)
    (CCT_MODULE_DIR/'cct_inference.py').write_text(render_inference(SPEC))


if not REPLAY_COMPLETE:
    import sys, os, json, hashlib, zipfile, itertools, gc
    from pathlib import Path
    from collections import defaultdict
    import pandas as pd, numpy as np, torch
    from tqdm.auto import tqdm
    from scipy.special import expit
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from ulm.compat import display
    if str(CCT_MODULE_DIR) not in sys.path:sys.path.insert(0,str(CCT_MODULE_DIR))
    for _mod in ('cct_common','cct_inference'):
        sys.modules.pop(_mod, None)
    from cct_common import load_cct,checker_module,hash_prompt,savejson,REVISION
    from cct_inference import CCTInference
    assert MODEL_REV==REVISION, 'CCT source model revision does not match main experiments'
    cct_new,cct_family,cct_protocol=load_cct(CCT_ZIP)
    CCT_ROOT=ULM_WORKDIR/'CCT_independent_refit'
    CCT_ROOT.mkdir(parents=True,exist_ok=True)
    CCT_CHECK=checker_module(CCT_ROOT)
    
    source_family_for_case={(r['section'],int(r['task_id'])):str(r['task_family_id']) for r in cct_new}
    assert len(set(source_family_for_case.values()))==282, 'Expected 282 independent CCT base-task families'
    assert all(sum(1 for sec,task in source_family_for_case if (sec,task)==key)==1 for key in cct_family)
    CCT_CASES=[]
    for (sec,task),f in sorted(cct_family.items()):
        family=source_family_for_case[(sec,task)]
        typ=('startend:end_checker' if sec in 'AC' else 'detectable_content:number_placeholders')
        new_example=next(r for r in cct_new if r['section']==sec and int(r['task_id'])==task)
        for cond in ('O','R1','C0','C1'):
            p=f[f'{cond}_prompt'];kw=f['original_kwargs'] if cond in ('O','R1') else f['counterfactual_kwargs']
            CCT_CASES.append({'id':f'{sec}_{task}_{cond}', 'section':sec,'task_id':int(task),
                'condition':cond,'task_family_id':family,'type':typ,'instruction_type':typ,
                'prompt':p,'prompt_sha256':hash_prompt(p),'target_kwargs':kw,
                'original_kwargs':f['original_kwargs'],'base_task':f['base_task'],
                'generation_budget':int(new_example['generation_budget']),
                'max_input_tokens':int(new_example.get('max_input_tokens',1024)),
                'fold':int(f['cct_fold'])})
    assert len(CCT_CASES)==2136 and len({x['id'] for x in CCT_CASES})==2136
    assert all(x['generation_budget']==(1024 if x['section'] in 'AB' else 1536) for x in CCT_CASES)
    cct_manifest={'source_zip_sha256':hashlib.sha256(CCT_ZIP.read_bytes()).hexdigest(),
     'checkpoint_revision':MODEL_REV,'independent_training':'O canonical, R1 augmentation from other families only',
     'heldout_requirement_conditions':['C0','C1'],'folds':5,
     'augmentation_training_source':'CCT O/R1 on training families, not archived source study probes',
     'method_names':['B0','B2','MT0','MT2'],'probe_layer':SPEC_PRIMARY_LAYER,'generation_policy':'BF16 SDPA greedy, source per-case caps'}
    from cct_common import savejson
    manifest_path=CCT_ROOT/'frozen_protocol.json'
    if manifest_path.exists():assert json.loads(manifest_path.read_text())==cct_manifest, 'CCT run provenance changed'
    else:savejson(manifest_path,cct_manifest)
    print('CCT verified:',len(CCT_CASES),'conditions,',len(set(source_family_for_case.values())),'source families')







if not REPLAY_COMPLETE:
    # Fit CCT probes on O/R1 training families.
    cct_runner=CCTInference(CCT_ROOT,CCT_CHECK)
    CCT_TYPES=('startend:end_checker','detectable_content:number_placeholders')
    CCT_TYPE_ID={t:i for i,t in enumerate(CCT_TYPES)}
    CCT_SEED=20261017
    CCT_FOLDS=5
    CCT_METHODS=('B0','B2','MT0','MT2')
    CCT_GROUPS=np.array(sorted(set(source_family_for_case.values())))
    cct_fold={item['task_family_id']:int(item['fold']) for item in CCT_CASES}
    assert len(cct_fold)==282
    assert set(cct_fold.values())==set(range(CCT_FOLDS))
    cct_split=[]
    for fold in range(CCT_FOLDS):
        test=np.flatnonzero([cct_fold[str(group)]==fold for group in CCT_GROUPS])
        train=np.flatnonzero([cct_fold[str(group)]!=fold for group in CCT_GROUPS])
        cct_split.append((train,test))
    cct_labels=[];cct_states=[]
    for item in tqdm([r for r in CCT_CASES if r['condition'] in ('O','R1')],desc='CCT source O/R1 (new responses + states)'):
        gen,_=cct_runner.get_response(item);st,_=cct_runner.get_state(item)
        cct_labels.append({k:item[k] for k in ('id','section','task_id','condition','task_family_id','type','base_task','prompt')}
             |{'label':int(gen['label']),'capped':int(gen['capped']),'tokens':int(gen['tokens']),
               'fold':cct_fold[item['task_family_id']]})
        cct_states.append(np.asarray(st[0],np.float32))
    CCT_TR=pd.DataFrame(cct_labels)
    CCT_X=np.stack(cct_states).astype(np.float32)
    assert len(CCT_TR)==1068 and len(CCT_X)==1068
    CCT_TR.to_csv(CCT_ROOT/'source_train_labels.csv',index=False)
    np.save(CCT_ROOT/('source_train_states_layer'+str(SPEC.primary_layer)+'.npy'),CCT_X.astype(np.float16))
    
    CCT_SOURCE_SHA=hashlib.sha256(CCT_TR[['id','label','fold']].to_csv(index=False).encode()+CCT_X.tobytes()).hexdigest()
    (CCT_ROOT/'source_training_fingerprint.sha256').write_text(CCT_SOURCE_SHA+'\n')
    CCT_TR['tid']=CCT_TR.type.map(CCT_TYPE_ID).astype(int)
    assert not CCT_TR.duplicated(['section','task_id','condition']).any()
    print('CCT training-source labels measured:',len(CCT_TR),'| success',round(float(CCT_TR.label.mean()),3))


if not REPLAY_COMPLETE:
    
    class CCTProbe:
        def __init__(self,mean,scale,w,delta,offset,b):
            self.mean=np.asarray(mean,np.float32);self.scale=np.asarray(scale,np.float32)
            self.w=np.asarray(w,np.float32);self.delta=np.asarray(delta,np.float32)
            self.offset=np.asarray(offset,np.float32);self.b=float(b)
        def predict(self,X,T):
            X=np.asarray(X,np.float32).reshape(-1,HIDDEN_WIDTH);T=np.asarray(T,int).reshape(-1)
            assert len(X)==len(T) and len(self.delta)==2
            z=(X-self.mean)/self.scale
            return expit(np.clip(np.sum(z*(self.w+self.delta[T]),axis=1)+self.b+self.offset[T],-35,35))
        def save(self,path):
            with open(path,'wb') as out:np.savez_compressed(out,mean=self.mean,scale=self.scale,w=self.w,
                delta=self.delta,offset=self.offset,b=np.array(self.b))
        @classmethod
        def load(cls,path):
            with np.load(path,allow_pickle=False) as x:return cls(x['mean'],x['scale'],x['w'],x['delta'],x['offset'],float(x['b']))

    def fit_cct(training_groups,method,C,penalty=16,max_iter=80):
        allowed=CCT_TR.task_family_id.isin(training_groups)
        chosen=allowed & (CCT_TR.condition.eq('O') if method in ('B0','MT0') else True)
        idx=np.flatnonzero(chosen.to_numpy())
        orig=np.flatnonzero((allowed&CCT_TR.condition.eq('O')).to_numpy())
        norm=StandardScaler().fit(CCT_X[orig]);scale=np.maximum(norm.scale_,1e-6).astype(np.float32)
        XX=((CCT_X[idx]-norm.mean_)/scale).astype(np.float32)
        yy=CCT_TR.iloc[idx].label.to_numpy(np.float32)
        typ=CCT_TR.iloc[idx].tid.to_numpy(np.int64)
        assert len(np.unique(yy))==2,('Missing classes for CCT training subset',method)
        groups=CCT_TR.iloc[idx].groupby(['task_family_id','type']).size().to_dict()
        weights=np.array([1/groups[(r.task_family_id,r.type)] for r in CCT_TR.iloc[idx].itertuples()],np.float32)
        weights/=weights.sum()
        device=torch.device('cuda:0')
        xt=torch.from_numpy(XX).to(device); yt=torch.from_numpy(yy).to(device)
        tt=torch.from_numpy(typ).to(device);sw=torch.from_numpy(weights).to(device)
        w=torch.nn.Parameter(torch.zeros(HIDDEN_WIDTH,device=device));b=torch.nn.Parameter(torch.zeros((),device=device))
        typed=method.startswith('MT')
        d=torch.nn.Parameter(torch.zeros((2,HIDDEN_WIDTH),device=device)) if typed else None
        o=torch.nn.Parameter(torch.zeros(2,device=device)) if typed else None
        opt=torch.optim.LBFGS([w,b]+([d,o] if typed else []),lr=1,max_iter=max_iter,
                               line_search_fn='strong_wolfe',tolerance_grad=2e-5,tolerance_change=1e-9)
        ridge=1/(2*float(C)*len(groups))
        def closure():
            opt.zero_grad()
            logits=(xt*(w+(d[tt] if typed else 0))).sum(1)+b
            if typed:logits=logits+o[tt]
            loss=(sw*torch.nn.functional.binary_cross_entropy_with_logits(logits,yt,reduction='none')).sum()+ridge*w.square().sum()
            if typed:loss=loss+ridge*float(penalty)*d.square().sum()+ridge*o.square().sum()
            assert torch.isfinite(loss).item()
            loss.backward();return loss
        opt.step(closure)
        output=CCTProbe(norm.mean_,scale,w.detach().cpu().numpy(),
          d.detach().cpu().numpy() if typed else np.zeros((2,HIDDEN_WIDTH),np.float32),
          o.detach().cpu().numpy() if typed else np.zeros(2,np.float32),float(b.detach().cpu()))
        del xt,yt,tt,sw,w,b,d,o,opt
        return output

    def cct_macro(probe,groups,conditions=('O','R1')):
        subset=CCT_TR[CCT_TR.task_family_id.isin(groups)&CCT_TR.condition.isin(conditions)]
        scores=[]
        for typ in CCT_TYPES:
            for cond in conditions:
                rows=subset[(subset.type==typ)&(subset.condition==cond)]
                if rows.label.nunique()<2:continue
                inds=rows.index.to_numpy(int)
                values=probe.predict(CCT_X[inds],np.full(len(inds),CCT_TYPE_ID[typ]))
                scores.append(roc_auc_score(rows.label,values))
        return float(np.mean(scores)) if scores else float('nan')

    CCT_PROBE_DIR=CCT_ROOT/'fold_probes';CCT_PROBE_DIR.mkdir(parents=True,exist_ok=True)
    CCT_FITTED={};CCT_TUNING=[]
    for fold,(tri,tei) in enumerate(cct_split):
        trainval=CCT_GROUPS[tri];held=CCT_GROUPS[tei]
        inner_tr,inner_val=train_test_split(trainval,test_size=.25,random_state=CCT_SEED+fold)
        base_orig=None
        for method in CCT_METHODS:
            dest=CCT_PROBE_DIR/f'{fold}_{method}.npz'
            selection=CCT_PROBE_DIR/f'{fold}_{method}.json'
            if dest.exists() and selection.exists():
                info=json.loads(selection.read_text())
                assert info['cct_protocol_sha256']==hashlib.sha256(json.dumps(cct_manifest,sort_keys=True).encode()).hexdigest()
                assert info['cct_source_label_state_sha256']==CCT_SOURCE_SHA, 'CCT refit weights do not match current generated source states/labels'
                probe=CCTProbe.load(dest);CCT_FITTED[(fold,method)]=probe;CCT_TUNING.append(info)
                if method=='B0':base_orig=info['original_val_AUROC']
                continue
            candidates=[]
            Cs=(0.03,0.1,0.3);penalties=(4.,16.)
            grid=[(c,16.) for c in Cs] if method in ('B0','B2') else list(itertools.product(Cs,penalties))
            for C,penalty in grid:
                p=fit_cct(inner_tr,method,C,penalty,max_iter=80)
                transfer=cct_macro(p,inner_val,('R1',))
                original=cct_macro(p,inner_val,('O',))
                if np.isfinite(transfer) and np.isfinite(original):
                    candidates.append((transfer,original,float(C),float(penalty)))
            assert candidates,(fold,method,'No class-balanced inner folds: CCT readout invalid')
            eligible=[e for e in candidates if base_orig is None or e[1]>=base_orig-.02]
            
            candidates_to_consider=eligible if eligible else candidates
            best=max(candidates_to_consider,key=lambda x:(x[0],x[1],-x[2]))
            if method=='B0':base_orig=best[1]
            p=fit_cct(trainval,method,best[2],best[3],max_iter=110)
            p.save(dest)
            setting={'fold':fold,'method':method,'C':best[2],'penalty':best[3],
              'original_val_AUROC':best[1],'rephrase_val_AUROC':best[0],
              'n_train_families':len(trainval),'n_test_families':len(held),
              'cct_protocol_sha256':hashlib.sha256(json.dumps(cct_manifest,sort_keys=True).encode()).hexdigest(),
              'cct_source_label_state_sha256':CCT_SOURCE_SHA}
            savejson(selection,setting)
            CCT_FITTED[(fold,method)]=p;CCT_TUNING.append(setting)
        print('Frozen CCT probe fold',fold,'| readouts',len(CCT_METHODS))
    assert len(CCT_FITTED)==20
    pd.DataFrame(CCT_TUNING).to_csv(CCT_ROOT/'probe_tuning_from_O_R1_only.csv',index=False)
    print('CCT readouts frozen before any counterfactual C0/C1 response generation')







if not REPLAY_COMPLETE:
    # Evaluate withheld C0/C1 conditions.
    for item in tqdm([r for r in CCT_CASES if r['condition'] in ('C0','C1')],desc='CCT withheld C0/C1 generation'):
        gen,_=cct_runner.get_response(item);_,_=cct_runner.get_state(item)
        assert int(gen['label'])==CCT_CHECK(item['instruction_type'],item['target_kwargs'],gen['response'])
    CCT_PROBE_PRED=[]
    for item in tqdm(CCT_CASES,desc='CCT all-condition out-of-fold scores'):
        gen,_=cct_runner.get_response(item);st,_=cct_runner.get_state(item)
        typ=CCT_TYPE_ID[item['instruction_type']];fold=cct_fold[item['task_family_id']]
        record={k:item[k] for k in ('id','section','task_id','condition','task_family_id','type','base_task','prompt')}
        record.update({'fold':fold,'label':int(gen['label']),'capped':int(gen['capped']),
          'tokens':int(gen['tokens']),'input_tokens':int(gen['input_tokens']),
          'original_grade_same_response':int(gen['original_grade_same_response'])})
        for method in CCT_METHODS:
            record[method]=float(CCT_FITTED[(fold,method)].predict(st[0],np.array([typ]))[0])
        CCT_PROBE_PRED.append(record)
    CCT_PRED=pd.DataFrame(CCT_PROBE_PRED)
    assert len(CCT_PRED)==2136
    assert CCT_PRED.groupby(['section','condition']).size().to_dict()=={
     (sec,cond):n for sec,n in {'A':102,'B':102,'C':161,'D':169}.items()
     for cond in ('O','R1','C0','C1')}
    assert not CCT_PRED.duplicated(['section','task_id','condition']).any()
    assert CCT_PRED.groupby('task_family_id').fold.nunique().max()==1
    CCT_PRED.to_csv(CCT_ROOT/'all_four_condition_oof_predictions.csv',index=False)
    
    task_scores=[]
    for fold in range(5):
        tr=CCT_PRED[(CCT_PRED.fold!=fold)&CCT_PRED.condition.eq('O')]
        te=CCT_PRED[CCT_PRED.fold==fold]
        assert tr.label.nunique()==2
        vector=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=25000)
        Xtr=vector.fit_transform(tr.base_task.astype(str))
        learner=LogisticRegression(max_iter=400,solver='liblinear',C=1.0)
        learner.fit(Xtr,tr.label)
        scores=learner.predict_proba(vector.transform(te.base_task.astype(str)))[:,1]
        task_scores.extend(zip(te.id.tolist(),scores.tolist()))
    prior=pd.DataFrame(task_scores,columns=['id','task_only'])
    CCT_PRED=CCT_PRED.merge(prior,on='id',validate='one_to_one')
    assert CCT_PRED.task_only.notna().all()
    CCT_PRED.to_csv(CCT_ROOT/'all_four_condition_oof_predictions.csv',index=False)
    
    print('CCT successful: all conditions have verified labels, frozen-fold model scores, and task-only controls.')







if not REPLAY_COMPLETE:
    CCT_STATS=[]
    for (sec,cond),g in CCT_PRED.groupby(['section','condition']):
        for m in (*CCT_METHODS,'task_only'):
            y=g.label.to_numpy(int)
            CCT_STATS.append({'section':sec,'condition':cond,'method':m,'n':len(g),
              'positive':int(y.sum()),'negative':int(len(y)-y.sum()),
              'AUROC':roc_auc_score(y,g[m]) if np.unique(y).size==2 else np.nan,
              'Brier':brier_score_loss(y,g[m])})
    CCT_STATS=pd.DataFrame(CCT_STATS)
    CCT_STATS.to_csv(CCT_ROOT/'four_condition_AUROC_by_section.csv',index=False)
    CCT_WIDE=CCT_PRED.pivot(index=['section','task_id','task_family_id','type','fold'],columns='condition',
       values=['label','B0','B2','MT0','MT2','task_only'])
    assert CCT_WIDE.notna().all().all()
    CCT_PAIRS=[]
    for (sec,task,family,typ,fold),r in CCT_WIDE.iterrows():
        entry={'section':sec,'task_id':task,'family':family,'type':typ,'fold':int(fold)}
        for a,b,name in [('O','R1','wording_original_rule'),('C0','C1','wording_changed_rule'),
                         ('O','C0','requirement_original_wording'),('R1','C1','requirement_reworded')]:
            dy=int(r[('label',b)]-r[('label',a)])
            entry[f'{name}_y_diff']=dy
            for method in (*CCT_METHODS,'task_only'):
                ds=float(r[(method,b)]-r[(method,a)])
                entry[f'{name}_{method}_score_diff']=ds
                entry[f'{name}_{method}_correct_on_flip']=int(ds*dy>0) if dy else np.nan
        entry['factorial_y_interaction']=int(r[('label','C1')]-r[('label','C0')]
                                             -r[('label','R1')]+r[('label','O')])
        CCT_PAIRS.append(entry)
    CCT_PAIRS=pd.DataFrame(CCT_PAIRS)
    CCT_PAIRS.to_csv(CCT_ROOT/'paired_2x2_changes.csv',index=False)
    
    rng=np.random.default_rng(20261017)
    FAMS=np.array(sorted(CCT_PAIRS.family.unique()));FIDX={x:i for i,x in enumerate(FAMS)}
    W=rng.multinomial(len(FAMS),np.full(len(FAMS),1/len(FAMS)),size=1000)
    CCT_FLIP=[]
    for section in ['A','B','C','D','all']:
        df=CCT_PAIRS if section=='all' else CCT_PAIRS[CCT_PAIRS.section==section]
        for name in ['wording_original_rule','wording_changed_rule','requirement_original_wording','requirement_reworded']:
            dy=df[f'{name}_y_diff'].to_numpy(int);changed=(dy!=0)
            family_index=np.array([FIDX[x] for x in df.family]);family_flips=np.bincount(family_index,weights=changed.astype(float),minlength=len(FAMS))
            up=int((dy==1).sum());down=int((dy==-1).sum())
            for method in (*CCT_METHODS,'task_only'):
                ds=df[f'{name}_{method}_score_diff'].to_numpy(float)
                right=(ds*dy>0)&changed
                family_right=np.bincount(family_index,weights=right.astype(float),minlength=len(FAMS))
                a=W@family_right;b=W@family_flips
                bootstrap=np.divide(a,b,out=np.full(len(W),np.nan),where=b>0)
                good=bootstrap[np.isfinite(bootstrap)]
                CCT_FLIP.append({'section':section,'contrast':name,'method':method,
                  'n_pairs':len(df),'n_flips':int(changed.sum()),'failure_to_success':up,'success_to_failure':down,
                  'flip_direction_acc':float(right.sum()/changed.sum()) if changed.sum() else np.nan,
                  'ci_low':float(np.quantile(good,.025)) if len(good) else np.nan,
                  'ci_high':float(np.quantile(good,.975)) if len(good) else np.nan,
                  'majority_direction_descriptive':max(up,down)/int(changed.sum()) if changed.sum() else np.nan})
    CCT_FLIP=pd.DataFrame(CCT_FLIP)
    CCT_FLIP.to_csv(CCT_ROOT/'pairwise_flip_direction_clustered_CI.csv',index=False)
    print('CCT C1 AUC:')
    display(CCT_STATS[(CCT_STATS.condition=='C1')][['section','method','n','positive','negative','AUROC']].round(3))
    print('CCT wording-change flip metrics (MT2 and task-only controls):')
    display(CCT_FLIP[(CCT_FLIP.section=='all')&(CCT_FLIP.method.isin(['MT2','task_only']))].round(3))







if not REPLAY_COMPLETE:
    import zipfile,hashlib,os,json
    manifest={
     'checkpoint':CFG['model_id'],'revision':MODEL_REV,
     'benchmark_zip':str(CCT_ZIP),'benchmark_sha256':hashlib.sha256(CCT_ZIP.read_bytes()).hexdigest(),
     'source_results':{'IFEval-Simple':str(SIMPLE_RESULTS),'IFEval-derived':str(IFEVAL_RESULTS) if RUN_IFEVAL_DERIVED else 'SKIPPED',
                       'FollowBench-derived':str(FOLLOWBENCH_RESULTS)},
     'cct_refit_protocol':'independent CCT five-fold O/R1-trained refit; NOT byte-identical archived source-study probes',
     'claims':['formulation_transfer','type-aware_over_equally_augmented_shared','paired_individual_change_prediction'],
     'model_weights_in_archive':False,
     'input_count':len(CCT_CASES),'source_families':len(FAMS)}
    (ULM_WORKDIR/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    package=ULM_WORKDIR/(SPEC.camel+'_Main_and_CCT_Final_Tables.zip')
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
        include=[ULM_WORKDIR/'RUN_MANIFEST.json']
        include+=list((ULM_WORKDIR/'final_tables').glob('*.csv'))
        include+=list((ULM_WORKDIR/'final_tables').glob('*.json'))
        include+=list(CCT_ROOT.glob('*.csv'))
        include+=list(CCT_ROOT.glob('frozen_protocol.json'))
        include+=list(CCT_ROOT.glob('probe_tuning_from_O_R1_only.csv'))
        for p in include:
            if p.is_file():z.write(p,arcname=str(p.relative_to(ULM_WORKDIR)))
    print('Completed.',package)
    print('Resume caches, full response texts, hidden states and probes stay under',ULM_WORKDIR)
    print('No previous Drive archives, manual downloads, Colab mount, or interactive upload was required.')
