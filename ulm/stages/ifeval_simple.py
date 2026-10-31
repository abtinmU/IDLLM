"""Part I. IFEval-Simple: source reproduction and matched reformulations.

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""











if not REPLAY_COMPLETE:
    # Download and verify IFEval-Simple data.
    DATA_URL = 'https://raw.githubusercontent.com/apple-aiml-research/ml-internal-llms-instruction-following/main/data/ifeval_simple.jsonl'
    DATA_FILE=DATA_DIR/'ifeval_simple.jsonl'
    CHECKER_FILES=('instructions.py','instructions_util.py','instructions_registry.py')
    CHECKER_DIR=DATA_DIR/'instruction_following_eval'
    CHECKER_DIR.mkdir(parents=True,exist_ok=True)
    (CHECKER_DIR/'__init__.py').touch(exist_ok=True)
    CHECKER_BASE='https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/'

    def fetch(url,p):
        p=Path(p)
        if p.is_file() and p.stat().st_size>0:return 'existing'
        local_checker=Path(os.environ.get('ULM_CHECKER_SOURCE_DIR',''))
        if url.startswith(CHECKER_BASE) and os.environ.get('ULM_CHECKER_SOURCE_DIR'):
            from hashlib import sha256
            pinned=local_checker/p.name
            expected={'instructions.py':'60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf',
              'instructions_util.py':'a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b',
              'instructions_registry.py':'ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc'}
            if pinned.is_file() and sha256(pinned.read_bytes()).hexdigest()==expected.get(p.name):
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(pinned.read_bytes())
                return 'bundled pinned official checker'

        r=requests.get(url,timeout=120);r.raise_for_status()
        p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.tmp')
        tmp.write_bytes(r.content)
        os.replace(tmp,p)
        return 'downloaded'
    
    
    
    import io
    SIMPLE_REPO='apple-aiml-research/ml-internal-llms-instruction-following'
    def _parse_simple_dataset(blob):
        try:
            obj=json.loads(blob.decode('utf8'))
            if isinstance(obj,dict):
                candidates=[obj[k] for k in ['examples','data','records','items'] if k in obj]
                obj=next((a for a in candidates if isinstance(a,list)),obj)
            if isinstance(obj,list):return obj
        except (ValueError,UnicodeDecodeError):pass
        try:return [json.loads(s) for s in blob.decode('utf8').splitlines() if s.strip()]
        except (ValueError,UnicodeDecodeError):return None

    def _valid_simple(rows):
        if not isinstance(rows,list) or len(rows)!=510:return False
        return all(isinstance(x,dict) and 'prompt' in x and 'instruction_id_list' in x
                   and 'kwargs' in x for x in rows)

    if DATA_FILE.exists():
        probe=_parse_simple_dataset(DATA_FILE.read_bytes())
        assert _valid_simple(probe),f'Cached IFEval-Simple is not the 510-record source: {DATA_FILE}'
    else:
        known=[DATA_URL,
          f'https://raw.githubusercontent.com/{SIMPLE_REPO}/main/data/ifeval_simple.json',
          f'https://raw.githubusercontent.com/{SIMPLE_REPO}/main/data/IFEval_simple.json',
          f'https://raw.githubusercontent.com/{SIMPLE_REPO}/main/data/IFEval-Simple.json',
          f'https://raw.githubusercontent.com/{SIMPLE_REPO}/main/data/IFEval_Simple.json']
        discovered=[]
        try:
            api=requests.get(f'https://api.github.com/repos/{SIMPLE_REPO}/git/trees/main?recursive=1',timeout=90)
            api.raise_for_status()
            entries=api.json()['tree']
            files=[x['path'] for x in entries if x.get('type')=='blob' and x['path'].startswith('data/')
                 and x['path'].lower().endswith(('.json','.jsonl'))]
            ranked=sorted(files,key=lambda x:(0 if 'simple' in x.lower() else 1, x))
            discovered=[f'https://raw.githubusercontent.com/{SIMPLE_REPO}/main/{path}' for path in ranked]
        except (requests.RequestException,KeyError,ValueError) as e:
            print('GitHub tree discovery unavailable; trying known official paths:',type(e).__name__)
        candidates=list(dict.fromkeys(known+discovered))
        found=None
        for url in candidates:
            try:
                reply=requests.get(url,timeout=120)
                if reply.status_code!=200:continue
                parsed=_parse_simple_dataset(reply.content)
                if _valid_simple(parsed):
                    found=(url,parsed)
                    break
            except requests.RequestException:continue
        assert found is not None, ('Unable to discover the official 510-record IFEval-Simple dataset. '
             'Check public GitHub network access; no alternate private file is assumed.')
        DATA_URL,probe=found
        
        norm=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in probe)
        tmp=DATA_FILE.with_suffix('.jsonl.tmp');tmp.write_text(norm,encoding='utf8');os.replace(tmp,DATA_FILE)
        (DATA_DIR/'simple_public_source_url.txt').write_text(DATA_URL+'\n',encoding='utf8')
        print('Official dataset discovered:',DATA_URL)
    print('IFEval-Simple records validated:',len(probe),'| local copy:',DATA_FILE)
    for name in CHECKER_FILES:print('Verifier',name,fetch(CHECKER_BASE+name,CHECKER_DIR/name))
    raw=[json.loads(s) for s in DATA_FILE.read_text(encoding='utf-8').splitlines() if s.strip()]
    assert len(raw)==510, f'Expected the original 510-row IFEval-Simple source: got {len(raw)}'
    assert all(raw[i*102+j]['instruction_id_list']==[TYPES[i]] for i in range(5) for j in range(102))
    base_tasks=[raw[k]['prompt'].split('\nDo not include the following keywords:')[0].strip() for k in range(102)]
    assert len(set(base_tasks))==102
    assert all(raw[t*102+k]['prompt'].startswith(base_tasks[k]) for t in range(5) for k in range(102))
    
    if str(DATA_DIR) not in sys.path:sys.path.insert(0,str(DATA_DIR))
    from instruction_following_eval import instructions_registry
    assert set(TYPES).issubset(instructions_registry.INSTRUCTION_DICT)
    SOURCE={'dataset_sha256':shafile(DATA_FILE),
            'checker_sha256':{f:shafile(CHECKER_DIR/f) for f in CHECKER_FILES},
            'original_repo_url':DATA_URL,'model_id':MODEL_ID,'revision':MODEL_REV}
    TYPE_TO_ID={t:i for i,t in enumerate(TYPES)}
    N_TYPES=len(TYPES)

    def kwargs_one(x):
        if isinstance(x,dict):x=[x]
        assert isinstance(x,list) and len(x)==1 and isinstance(x[0],dict),x
        return x[0]

    def strict_score(record,response):
        if not isinstance(response,str) or not response.strip():return 0
        iid=record['instruction_type']
        checker=instructions_registry.INSTRUCTION_DICT[iid](iid)
        checker.build_description(**kwargs_one(record['kwargs']))
        return int(bool(checker.check_following(response)))

    def score(typ,kw,response):
        return strict_score({'instruction_type':typ,'kwargs':kw},response)
    print('Verified source: 510 rows / 102 groups / 5 official types; SHA computed at runtime')


if not REPLAY_COMPLETE:
    from collections import Counter
    def phrase(instruction_id, kw, variant):
        assert variant in ('P1','P2','P3')
        q=lambda x:json.dumps(str(x),ensure_ascii=False)
        if instruction_id=='keywords:forbidden_words':
            assert set(kw)=={'forbidden_words'} and isinstance(kw['forbidden_words'],list)
            words=', '.join(q(z) for z in kw['forbidden_words'])
            templates=[f'Never include any of these keywords in the answer: {words}.',
                       f'Your response must not contain any keyword from this list: {words}.',
                       f'Constraint on your answer: avoid every one of these keywords: {words}.']
        elif instruction_id=='keywords:existence':
            assert set(kw)=={'keywords'} and isinstance(kw['keywords'],list)
            words=', '.join(q(z) for z in kw['keywords'])
            templates=[f'Include all the following keywords in your answer: {words}.',
                       f'Your response must contain each keyword in this list: {words}.',
                       f'Required keywords (include every one): {words}.']
        elif instruction_id=='keywords:frequency':
            assert set(kw)=={'relation','keyword','frequency'}
            rel=kw['relation']; n=int(kw['frequency']); w=q(kw['keyword'])
            assert rel in ('at least','less than') and n>=0
            if rel=='at least':
                templates=[f'Use the word {w} at least {n} times in your answer.',
                           f'Your response must contain {w} no fewer than {n} times.',
                           f'Keyword-count requirement: {w} must occur {n} times or more.']
            else:
                templates=[f'Use the word {w} fewer than {n} times in your answer.',
                           f'Your response must contain {w} less than {n} times.',
                           f'Keyword-count requirement: {w} must occur strictly fewer than {n} times.']
        elif instruction_id=='detectable_content:number_placeholders':
            assert set(kw)=={'num_placeholders'}
            n=int(kw['num_placeholders']); assert n>=0
            templates=[f'Include at least {n} placeholders enclosed in square brackets, such as [name].',
                       f'Your answer must contain no fewer than {n} square-bracket placeholders (for example, [name]).',
                       f'Placeholder requirement: include {n} or more placeholders of the form [name].']
        elif instruction_id=='startend:end_checker':
            assert set(kw)=={'end_phrase'}; ending=q(kw['end_phrase'])
            templates=[f'End your answer with the exact phrase {ending}.',
                       f'The final words of your response must be exactly {ending}.',
                       f'Required ending (nothing should appear after it): {ending}.']
        else: raise ValueError(instruction_id)
        return dict(zip(('P1','P2','P3'),templates))[variant]

    def new_rule(typ,kw,v):
        a=kwargs_one(kw);q=lambda z:json.dumps(str(z),ensure_ascii=False)
        assert v in R_TEMPLATES
        if typ=='keywords:forbidden_words':
            ws=', '.join(q(x) for x in a['forbidden_words'])
            return (f'Forbidden vocabulary for the final answer: {ws}. Do not use any of these terms.' if v=='R1' else
                    f'Ensure that none of the following expressions appears in your reply: {ws}.')
        if typ=='keywords:existence':
            ws=', '.join(q(x) for x in a['keywords'])
            return (f'Mandatory terms to include in the output: {ws}. Include every term.' if v=='R1' else
                    f'Your complete answer must include all these exact words: {ws}.')
        if typ=='keywords:frequency':
            n=int(a['frequency']);w=q(a['keyword']);rel=a['relation']
            assert rel in ('at least','less than')
            if rel=='at least':
                return (f'Occurrence requirement: include the word {w} a minimum of {n} times.' if v=='R1' else
                        f'The word {w} must be present {n} times or more in the complete reply.')
            return (f'Occurrence restriction: use the word {w} fewer than {n} times.' if v=='R1' else
                    f'The total number of uses of the word {w} must be strictly below {n}.')
        if typ=='detectable_content:number_placeholders':
            n=int(a['num_placeholders'])
            return (f'Square-bracket placeholders are required: add at least {n} nonempty entries such as [name].' if v=='R1' else
                    f'Include {n} or more placeholders written within square brackets, for example [name].')
        if typ=='startend:end_checker':
            end=q(a['end_phrase'])
            return (f'Mandatory last phrase: {end}. The response must finish with that exact phrase.' if v=='R1' else
                    f'The exact final words of your answer must be {end}, without any subsequent text.')
        raise ValueError(typ)

    def make_prompt(task,rule,first=False):
        
        
        return (str(rule)+'\n'+str(task)) if first else (str(task)+'\n'+str(rule))


if not REPLAY_COMPLETE:
    
    rows=[]
    for task in range(102):
        for typ in PRIMARY_TYPES:
            tid=TYPE_TO_ID[typ]
            x=raw[tid*102+task]
            kw=x['kwargs']
            for variant in ('original','P1','P2','P3','R1','R2'):
                if variant=='original':prompt=x['prompt']
                elif variant in ('P1','P2','P3'):
                    prompt=base_tasks[task]+'\n'+phrase(typ,kwargs_one(kw),variant)
                else:prompt=make_prompt(base_tasks[task],new_rule(typ,kw,variant))
                rows.append({'id':f'{task:03d}_{tid}_{variant}','task_id':task,'type_id':tid,
                            'instruction_type':typ,'variant':variant,'prompt':prompt,
                            'kwargs':kw,'task_text':base_tasks[task]})
    assert len(rows)==102*3*6 and len({r['id'] for r in rows})==len(rows)
    assert len({r['prompt'] for r in rows})==len(rows),'Source prompts duplicated unexpectedly'
    records=pd.DataFrame(rows)
    TASKS=np.arange(102)
    outer=list(KFold(n_splits=FOLDS,shuffle=True,random_state=SEED).split(TASKS))
    fold_task={int(TASKS[j]):i for i,(_,te) in enumerate(outer) for j in te}
    assert len(fold_task)==102
    PROTOCOL={
     'study':'ifeval_simple_corrected_phase3_cross_model_v1', 'model':MODEL,
     'model_id':MODEL_ID,'revision':MODEL_REV,'dataset':SOURCE,
     'seed':SEED,'folds':FOLDS,'fold_assignment':{str(k):int(v) for k,v in fold_task.items()},
     'layers':list(LAYERS),'primary_layer':PRIMARY_LAYER,
     'types':list(PRIMARY_TYPES),'type_universe':TYPES,'training_variants':['original','P1','P2','P3'],
     'test_variants':['R1','R2'],'Cs':list(Cs),'penalties':list(DELTA_PENALTIES),
     'max_new_tokens':GENERATION_BUDGET,'max_input_tokens':MAX_INPUT_TOKENS,
     'decoding':'greedy','attention':'sdpa','dtype':'bf16',
     'transformers_version':transformers.__version__,'torch_version':torch.__version__,
     'gpu_name':GPU_NAME, 'chat_template':'single_user_add_generation_prompt',
     'prompt_manifest_sha256':sha(canonical([{'id':r['id'],'prompt':r['prompt'],'kwargs':r['kwargs']} for r in rows]).encode())
    }
    RUN_HASH=sha(canonical(PROTOCOL).encode())[:20]
    RUN=MODEL_ROOT/'runs'/RUN_HASH
    RUN.mkdir(parents=True,exist_ok=True)
    freeze_json(RUN/'protocol.json',PROTOCOL)
    atomic_json(RUN/'source_provenance.json',SOURCE)
    records.to_json(RUN/'frozen_prompts.jsonl',orient='records',lines=True,force_ascii=False)
    GEN_DIR=RUN/'generations';ST_DIR=RUN/'prompt_states';P_DIR=RUN/'probes';RESULT_DIR=RUN/'results'
    for d in (GEN_DIR,ST_DIR,P_DIR,RESULT_DIR):d.mkdir(parents=True,exist_ok=True)
    print('Run folder:',RUN)
    print('Prompt count:',len(records),'| expected original',len(records[records.variant.eq('original')]))







if not REPLAY_COMPLETE:
    
    model=None;tokenizer=None;INPUT_DEVICE=None;HIDDEN_SIZE=int(CFG['expected_hidden']);STOP_IDS=None
    from transformers import AutoModelForCausalLM,AutoTokenizer

    # Load the model only when cached data is unavailable.
    def ensure_model():
        global model,tokenizer,INPUT_DEVICE,HIDDEN_SIZE
        if model is not None:return
        assert transformers.__version__=='4.53.3'
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(
            MODEL_ID,revision=MODEL_REV,torch_dtype=DTYPE,device_map='auto',
            trust_remote_code=False,attn_implementation='sdpa')
        model.eval()
        global STOP_IDS
        STOP_IDS=prepare_tokenizer(SPEC,tokenizer,model)
        assert model.config.hidden_size==CFG['expected_hidden'], (model.config.hidden_size,CFG['expected_hidden'])
        assert len(model.model.layers)==CFG['expected_layers'],len(model.model.layers)
        assert len(LAYERS)==3 and min(LAYERS)>=1 and max(LAYERS)<=model.config.num_hidden_layers
        assert model.__class__.__module__.startswith('transformers.models.'+CFG['family']),model.__class__.__module__
        assert getattr(model.config,'_commit_hash',None) in (None,MODEL_REV), 'Loaded model revision does not match pin'
        INPUT_DEVICE=model.get_input_embeddings().weight.device
        assert INPUT_DEVICE.type=='cuda'
        HIDDEN_SIZE=int(model.config.hidden_size)
        print('Loaded',MODEL_ID,'@',MODEL_REV,'| hidden width',HIDDEN_SIZE)

    def encode_prompt(p):
        assert tokenizer is not None
        enc=tokenizer.apply_chat_template([{'role':'user','content':p}],
            add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
        n=int(enc['input_ids'].shape[-1])
        assert 0<n<=MAX_INPUT_TOKENS, f'Input length {n} exceeds MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}; never silently truncate'
        return {k:v.to(INPUT_DEVICE) for k,v in enc.items()},n

    @torch.inference_mode()
    def generate_response(p):
        enc,n=encode_prompt(p)
        tokens=model.generate(**enc,max_new_tokens=GENERATION_BUDGET,do_sample=False,
                              **generation_stop_kwargs(SPEC,tokenizer,STOP_IDS),use_cache=True)
        new=tokens[0,n:]
        k=int(new.numel())
        ends=end_token_ids(SPEC,tokenizer,STOP_IDS)
        cap=bool(k>=GENERATION_BUDGET and (k==0 or int(new[-1]) not in ends))
        response=tokenizer.decode(new,skip_special_tokens=True).strip()
        del tokens,enc,new
        return {'response':response,'input_tokens':n,'output_tokens':k,'hit_token_cap':cap}

    @torch.inference_mode()
    def extract_state(p):
        enc,_=encode_prompt(p)
        out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
        v=np.stack([out.hidden_states[i][0,-1].float().cpu().numpy() for i in LAYERS]).astype(np.float16)
        del out,enc
        assert v.shape==(len(LAYERS),HIDDEN_SIZE) and np.isfinite(v).all()
        return v

    
    def cache_key(r):return sha((str(r['id'])+'|'+sha(r['prompt'].encode())).encode())[:32]
    def expected_args(r):
        return {'id':str(r['id']),'prompt_sha256':sha(r['prompt'].encode()),
                'rule_sha256':sha(canonical(kwargs_one(r['kwargs'])).encode()),
                'run_sha256':sha(canonical(PROTOCOL).encode()),'model_id':MODEL_ID,
                'revision':MODEL_REV,'max_new_tokens':GENERATION_BUDGET,'layers':list(LAYERS)}

    def get_generation(r):
        p=GEN_DIR/(cache_key(r)+'.json');args=expected_args(r)
        if p.is_file():
            item=json.loads(p.read_text(encoding='utf-8'))
            assert item['args']==args,f'Generation fingerprint changed: {p}'
            assert int(item['label'])==strict_score(r,item['response']),'Cached verifier label mismatch'
            return item,True
        ensure_model()
        generated=generate_response(r['prompt'])
        item={'args':args,**generated,'label':strict_score(r,generated['response'])}
        atomic_json(p,item)
        return item,False

    def get_state(r):
        key=cache_key(r);p=ST_DIR/(key+'.npz');meta=ST_DIR/(key+'.json');args=expected_args(r)
        if p.is_file() and meta.is_file():
            assert json.loads(meta.read_text(encoding='utf-8'))==args,'State fingerprint mismatch'
            with np.load(p,allow_pickle=False) as z:a=z['states'].astype(np.float32)
            assert a.shape==(len(LAYERS),HIDDEN_SIZE) and np.isfinite(a).all()
            return a,True
        ensure_model()
        a=extract_state(r['prompt'])
        atomic_npz(p,states=a)
        atomic_json(meta,args)
        return a.astype(np.float32),False

    
    
    assert PRIMARY_LAYER in LAYERS


if not REPLAY_COMPLETE and SPEC.smoke_test:
    
    
    ensure_model()
    _smoke='Reply with one word: ready.'
    _smoke_enc,_smoke_prompt_tokens=encode_prompt(_smoke)
    with torch.inference_mode():
        _smoke_generated=model.generate(**_smoke_enc,do_sample=False,max_new_tokens=4,
                              **generation_stop_kwargs(SPEC,tokenizer,STOP_IDS),use_cache=True)
    _smoke_suffix=_smoke_generated[0,_smoke_prompt_tokens:]
    assert 1<=len(_smoke_suffix)<=4, 'Failed to generate the smoke-test response'
    _smoke_states=extract_state(_smoke)
    assert _smoke_states.shape==(3,HIDDEN_WIDTH) and np.isfinite(_smoke_states).all()
    assert model.config.model_type==SPEC.expected_model_type and model.config.num_hidden_layers==SPEC.num_layers
    print(SPEC.short_name,'model smoke test passed:',tokenizer.decode(_smoke_suffix,skip_special_tokens=True),
          '| state shape',_smoke_states.shape, '| stop IDs',STOP_IDS)
    del _smoke_enc,_smoke_generated,_smoke_suffix,_smoke_states


if not REPLAY_COMPLETE:
    
    
    rows_scored=[];states=[];reused_gen=reused_states=0
    for k,r in enumerate(tqdm(rows,desc=MODEL+' Simple prompts')):
        gen,hit=get_generation(r)
        z,hit_state=get_state(r)
        reused_gen+=int(hit);reused_states+=int(hit_state)
        rows_scored.append({**r,'label':int(gen['label']),
                 'response':gen['response'],'input_tokens':int(gen['input_tokens']),
                 'output_tokens':int(gen['output_tokens']),'hit_token_cap':bool(gen['hit_token_cap']),
                 'fold':fold_task[int(r['task_id'])]})
        states.append(z)
        if (k+1)%50==0:
            print('Processed',k+1,'/',len(rows),'| cached responses',reused_gen,'| cached states',reused_states)
        if (k+1)%125==0:
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    scored=pd.DataFrame(rows_scored)
    state_array=np.stack(states).astype(np.float16)
    assert len(scored)==102*len(PRIMARY_TYPES)*6 and scored.id.is_unique
    assert state_array.shape==(len(scored),len(LAYERS),HIDDEN_SIZE)
    scored.to_json(RUN/'all_1836_scored.jsonl',orient='records',lines=True,force_ascii=False)
    np.save(RUN/'states_ordered.npy',state_array,allow_pickle=False)
    assert all(int(r.label)==strict_score(r._asdict(),r.response) for r in scored.itertuples(index=False)), 'Official checker replay mismatch'
    print('All outputs verified:',len(scored),'| generations reused',reused_gen,'| states reused',reused_states)
    print('Token cap rate:',round(float(scored.hit_token_cap.mean()),4))
    display(scored.groupby(['instruction_type','variant']).agg(n=('label','size'),success=('label','mean'),cap=('hit_token_cap','mean')).round(4))
    
    if model is not None:
        del model,tokenizer
        model=None;tokenizer=None;INPUT_DEVICE=None
        gc.collect();torch.cuda.empty_cache()







if not REPLAY_COMPLETE:
    
    # Build the probe training set without held-out variants.
    train_df=scored[scored.variant.isin(['original','P1','P2','P3'])].copy().reset_index(drop=True)
    train_idx=scored.index[scored.variant.isin(['original','P1','P2','P3'])].to_numpy()
    X_old=state_array[train_idx,LAYERS.index(PRIMARY_LAYER)].astype(np.float32)
    assert len(train_df)==102*len(PRIMARY_TYPES)*4 and X_old.shape==(len(train_df),HIDDEN_SIZE)
    assert sorted(train_df.variant.unique().tolist())==['P1','P2','P3','original']
    print('Model-specific labels and hidden states ready for original Phase 3 training:',X_old.shape)


if not REPLAY_COMPLETE:
    class Probe:
        def __init__(self,mean,scale,w,delta,offset,b,method,setting):
            self.mean=np.asarray(mean,np.float32);self.scale=np.asarray(scale,np.float32)
            self.w=np.asarray(w,np.float32);self.delta=np.asarray(delta,np.float32)
            self.offset=np.asarray(offset,np.float32);self.b=float(b)
            self.method=method;self.setting=setting
        def predict(self,X,t):
            X=np.asarray(X,np.float32)
            if X.ndim==1:X=X[None,:]
            t=np.asarray(t,dtype=int).reshape(-1)
            assert len(X)==len(t) and X.shape[1]==len(self.w)
            v=(X-self.mean)/self.scale
            logits=np.sum(v*(self.w[None,:]+self.delta[t]),axis=1)+self.b+self.offset[t]
            return expit(np.clip(logits,-35,35))
        def save(self,path):
            atomic_npz(path,mean=self.mean,scale=self.scale,w=self.w,delta=self.delta,
                       offset=self.offset,b=np.array(self.b),setting=np.array(canonical(self.setting)))
        @staticmethod
        def load(path,method):
            with np.load(path,allow_pickle=False) as z:
                return Probe(z['mean'],z['scale'],z['w'],z['delta'],z['offset'],
                             float(z['b']),method,json.loads(str(z['setting'])))

    def fit_probe(task_ids,method,C,penalty=16.,max_iter=80):
        assert method in METHODS
        subset=train_df.task_id.isin(task_ids) & (train_df.variant.eq('original') if method in ('B0','MT0') else True)
        idx=np.flatnonzero(subset.to_numpy())
        orig_idx=np.flatnonzero((train_df.task_id.isin(task_ids)&train_df.variant.eq('original')).to_numpy())
        scaler=StandardScaler().fit(X_old[orig_idx])
        scale=np.maximum(scaler.scale_,1e-6).astype(np.float32)
        X=((X_old[idx]-scaler.mean_)/scale).astype(np.float32)
        y=train_df.iloc[idx].label.to_numpy(dtype=np.float32)
        t=train_df.iloc[idx].type_id.to_numpy(dtype=np.int64)
        assert np.unique(y).size==2
        groups=train_df.iloc[idx].groupby(['task_id','type_id']).size().to_dict()
        sample_weights=np.array([1/groups[(int(a),int(b))] for a,b in train_df.iloc[idx][['task_id','type_id']].to_numpy()],dtype=np.float32)
        sample_weights/=sample_weights.sum()
        
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xt=torch.from_numpy(X).to(device);yt=torch.from_numpy(y).to(device)
        tt=torch.from_numpy(t).to(device);sw=torch.from_numpy(sample_weights).to(device)
        ww=torch.nn.Parameter(torch.zeros(HIDDEN_SIZE,device=device,dtype=torch.float32))
        bb=torch.nn.Parameter(torch.zeros((),device=device,dtype=torch.float32))
        mt=method.startswith('MT')
        dd=torch.nn.Parameter(torch.zeros((N_TYPES,HIDDEN_SIZE),device=device,dtype=torch.float32)) if mt else None
        oo=torch.nn.Parameter(torch.zeros(N_TYPES,device=device,dtype=torch.float32)) if mt else None
        params=[ww,bb]+([dd,oo] if mt else [])
        opt=torch.optim.LBFGS(params,lr=1.,max_iter=max_iter,line_search_fn='strong_wolfe',
                              tolerance_grad=2e-5,tolerance_change=1e-9)
        unit_count=int(len(groups));ridge=1.0/(2*float(C)*unit_count)
        def closure():
            opt.zero_grad()
            logit=(xt* (ww[None,:]+dd[tt] if mt else ww[None,:])).sum(dim=1)+bb
            if mt:logit=logit+oo[tt]
            bce=torch.nn.functional.binary_cross_entropy_with_logits(logit,yt,reduction='none')
            loss=(sw*bce).sum()+ridge*ww.square().sum()
            if mt:loss=loss+ridge*float(penalty)*dd.square().sum()+ridge*oo.square().sum()
            assert torch.isfinite(loss).item(),'Nonfinite probe loss'
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            out=Probe(scaler.mean_,scale,ww.cpu().numpy(),
               dd.cpu().numpy() if mt else np.zeros((N_TYPES,HIDDEN_SIZE),dtype=np.float32),
               oo.cpu().numpy() if mt else np.zeros(N_TYPES,dtype=np.float32),
               bb.cpu().item(),method,{'C':float(C),'penalty':float(penalty)})
        del xt,yt,tt,sw,ww,bb,dd,oo,opt
        return out

    def macro_old(probe,task_ids,variants):
        scores=[]
        for typ in PRIMARY_TYPES:
            tid=TYPE_TO_ID[typ]
            for variant in variants:
                ix=np.flatnonzero((train_df.task_id.isin(task_ids)&train_df.instruction_type.eq(typ)&train_df.variant.eq(variant)).to_numpy())
                if len(ix)<1:continue
                y=train_df.iloc[ix].label.to_numpy(dtype=int)
                if len(np.unique(y))<2:continue
                pred=probe.predict(X_old[ix],np.full(len(ix),tid))
                scores.append(auc(y,pred))
        return float(np.mean(scores)) if scores else float('nan')


if not REPLAY_COMPLETE:
    
    
    PROBE_DIR=P_DIR
    PROBE_DIR.mkdir(exist_ok=True)

    def selected_rows(task_ids,variants):
        return np.flatnonzero((train_df.task_id.isin(task_ids)&train_df.variant.isin(variants)).to_numpy())

    
    train_content=sha(canonical([{'id':r['id'],'label':int(r['label']),
          'prompt_sha256':sha(str(r['prompt']).encode())} for r in train_df.to_dict('records')]).encode())
    fit_def={'protocol':RUN_HASH,'training_sha256':train_content,
             'methods':list(METHODS),'source_method':'Phase3_torch_LBFGS_weighted_BCE',
             'hidden_width':HIDDEN_SIZE,'type_count':N_TYPES,
             'fit_iters_train':80,'fit_iters_refit':110}
    freeze_json(PROBE_DIR/'fit_manifest.json',fit_def)
    FIT_FP=sha(canonical(fit_def).encode())
    fitted={};tuning=[]
    for fold,(trval_idx,test_idx) in enumerate(outer):
        trval=TASKS[trval_idx];test=TASKS[test_idx]
        inner_train,inner_val=train_test_split(trval,test_size=0.25,random_state=SEED+fold)
        assert set(inner_train).isdisjoint(inner_val) and set(test).isdisjoint(trval)
        base_orig=None
        for method in METHODS:
            mpath=PROBE_DIR/f'fold{fold}_{method}.npz'
            meta=PROBE_DIR/f'fold{fold}_{method}.json'
            if mpath.is_file() and meta.is_file():
                setting=json.loads(meta.read_text(encoding='utf-8'))
                assert setting['fit_fp']==FIT_FP, f'Fitted readout fingerprint mismatch: {mpath}'
                fitted[(fold,method)]=Probe.load(mpath,method)
                tuning.append(setting)
                if method=='B0':base_orig=setting['val_original_auc']
                print(f'Reused fold={fold} method={method}')
                continue
            choices=[(C,16.) for C in Cs] if not method.startswith('MT') else list(itertools.product(Cs,DELTA_PENALTIES))
            candidates=[]
            for C,penalty in choices:
                p=fit_probe(inner_train,method,C,penalty)
                valid=macro_old(p,inner_val,('P1','P2','P3'))
                org=macro_old(p,inner_val,('original',))
                if np.isfinite(valid) and np.isfinite(org):candidates.append((valid,org,C,penalty))
            if not candidates:
                raise RuntimeError(f'Model {MODEL}, fold {fold}, {method}: no binary-class validation cells. '                               'Report class support; do not silently tune on R1/R2.')
            feasible=[x for x in candidates if base_orig is None or x[1]>=base_orig-ORIG_MARGIN]
            best=max(feasible if feasible else candidates,key=lambda x:(x[0],x[1],-x[2],-x[3]))
            final=fit_probe(trval,method,best[2],best[3],max_iter=110)
            setting={'fold':fold,'method':method,'fit_fp':FIT_FP,
                     'trainval_tasks':list(map(int,trval)),'test_tasks':list(map(int,test)),
                     'C':float(best[2]),'delta_penalty':float(best[3]),
                     'val_shift_auc':float(best[0]),'val_original_auc':float(best[1]),
                     'val_feasible':bool(feasible)}
            final.save(mpath);atomic_json(meta,setting)
            fitted[(fold,method)]=final;tuning.append(setting)
            if method=='B0':base_orig=best[1]
            print(f'FITTED fold={fold} method={method}')
    assert len(fitted)==FOLDS*len(METHODS)
    tune_df=pd.DataFrame(tuning)
    tune_df.to_csv(RESULT_DIR/'probe_tuning.csv',index=False)
    print('All',len(fitted),'fold-specific probes are fitted and frozen from this model.')







if not REPLAY_COMPLETE:
    
    all_eval=[]
    for j,r in enumerate(scored.itertuples(index=False)):
        if r.variant not in ('original','R1','R2'):continue
        fold=fold_task[int(r.task_id)]
        p={'id':r.id,'task_id':int(r.task_id),'fold':fold,'type':r.instruction_type,
           'variant':r.variant,'label':int(r.label),'capped':bool(r.hit_token_cap),
           'input_tokens':int(r.input_tokens),'output_tokens':int(r.output_tokens)}
        for m in METHODS:
            p[m]=float(fitted[(fold,m)].predict(
                 state_array[j,LAYERS.index(PRIMARY_LAYER)].astype(np.float32),[int(r.type_id)])[0])
        
        old_tr=train_df[(train_df.task_id.isin(TASKS[outer[fold][0]]))&
                        train_df.instruction_type.eq(r.instruction_type)&train_df.variant.eq('original')]
        p['type_only']=float((int(old_tr.label.sum())+1)/(len(old_tr)+2))
        all_eval.append(p)
    preds=pd.DataFrame(all_eval)
    assert len(preds)==102*len(PRIMARY_TYPES)*3
    assert preds.groupby(['task_id','type']).size().eq(3).all()
    assert all(x not in set(TASKS[outer[int(p.fold)][0]]) for p in preds.itertuples(index=False) for x in [p.task_id])
    preds.to_csv(RESULT_DIR/'oof_original_and_R_predictions.csv',index=False)

    def macro(t,m,variants=('R1','R2')):
        scores=[]
        for typ in PRIMARY_TYPES:
            for v in variants:
                g=t[(t.type==typ)&(t.variant==v)]
                if len(g) and g.label.nunique()==2:scores.append(safe_auc(g.label,g[m]))
        return float(np.mean(scores)) if scores else np.nan

    metrics=[]
    for (typ,v),g in preds.groupby(['type','variant']):
        for m in (*METHODS,'type_only'):
            metrics.append({'type':typ,'variant':v,'method':m,'n':len(g),
                'positives':int(g.label.sum()),'negatives':int(len(g)-g.label.sum()),
                'auroc':safe_auc(g.label,g[m]),
                'brier':float(brier_score_loss(g.label,g[m])),
                'logloss':float(log_loss(g.label,g[m],labels=[0,1]))})
    metrics_df=pd.DataFrame(metrics)
    metrics_df.to_csv(RESULT_DIR/'per_type_formulation_metrics.csv',index=False)
    print('Original-domain and held-out AUROC:')
    display(metrics_df.groupby(['variant','method']).auroc.mean().unstack('method').round(4))


if not REPLAY_COMPLETE:
    
    
    R_pred=preds[preds.variant.isin(['R1','R2'])].copy()
    oof=preds[preds.variant.eq('original')].copy()
    old_lookup={(int(r.task_id),r.type):r for r in oof.itertuples(index=False)}
    flip_rows=[]
    for r in R_pred.itertuples(index=False):
        old=old_lookup[(int(r.task_id),r.type)]
        change=int(r.label)-int(old.label)
        row={'task_id':int(r.task_id),'type':r.type,'variant':r.variant,
             'old_label':int(old.label),'new_label':int(r.label),'change':change,
             'old_capped':bool(old.capped),'new_capped':bool(r.capped)}
        for m in METHODS:
            delta=float(getattr(r,m)-getattr(old,m))
            row[m+'_delta']=delta
            row[m+'_correct']=float(delta*change>0) if change else np.nan
        flip_rows.append(row)
    flip_df=pd.DataFrame(flip_rows)
    flip_df.to_csv(RESULT_DIR/'paired_behavioral_flips.csv',index=False)

    H1=[]
    for typ in PRIMARY_TYPES:
        for v in R_TEMPLATES:
            g=flip_df[(flip_df.type==typ)&flip_df.variant.eq(v)]
            H1.append({'type':typ,'variant':v,'n':len(g),'flips':int(g.change.ne(0).sum()),
                       'flip_rate':float(g.change.ne(0).mean()),
                       'up':int(g.change.eq(1).sum()),'down':int(g.change.eq(-1).sum()),
                       'mean_success_change':float(g.change.mean()),
                       'mcnemar_exact_p':float(binomtest(int(g.change.eq(1).sum()),int(g.change.ne(0).sum()),0.5).pvalue)
                            if int(g.change.ne(0).sum()) else 1.0})
    H1=pd.DataFrame(H1)
    H1.to_csv(RESULT_DIR/'H1_pairwise_behavior.csv',index=False)

    def family_bootstrap(table, variants, seed_offset):
        """Fast source-family bootstrap with exact weighted tie-aware AUROC.

        Resampling 102 task families with replacement is equivalent to giving
        each original family's observations their multinomial sample count.
        All methods use identical draws (paired effects). Single-class draw
        cells are skipped, matching the source Phase 3 macro definition.
        """
        rng=np.random.default_rng(SEED+seed_offset)
        tasks=np.array(sorted(table.task_id.unique()))
        pos={int(k):i for i,k in enumerate(tasks)}
        draws=rng.multinomial(len(tasks),np.full(len(tasks),1./len(tasks)),size=N_BOOT)
        methods=(*METHODS,'type_only')
        auc_boot={m:np.full(N_BOOT,np.nan) for m in methods}
        brier_boot={m:np.full(N_BOOT,np.nan) for m in methods}
        groups=[]
        for typ in PRIMARY_TYPES:
            for v in variants:
                g=table[(table.type==typ)&table.variant.eq(v)]
                if g.empty:continue
                ix=np.array([pos[int(t)] for t in g.task_id],dtype=np.int32)
                y=g.label.to_numpy(dtype=int)
                scores={m:g[m].to_numpy(dtype=float) for m in methods}
                ties={m:np.unique(scores[m],return_inverse=True)[1] for m in methods}
                groups.append((ix,y,scores,ties))
        for m in methods:
            for k,draw in enumerate(draws):
                aa=[];bb=[]
                for ix,y,scores,ties in groups:
                    w=draw[ix].astype(float)
                    if w.sum()==0:continue
                    p=scores[m]
                    bb.append(np.dot(w,(y-p)**2)/w.sum())
                    posw=w*y;negw=w*(1-y)
                    if posw.sum()==0 or negw.sum()==0:continue
                    group_ids=ties[m]
                    ngrp=int(group_ids.max())+1
                    pg=np.bincount(group_ids,weights=posw,minlength=ngrp)
                    ng=np.bincount(group_ids,weights=negw,minlength=ngrp)
                    win=np.dot(pg,np.cumsum(ng)-ng+0.5*ng)
                    aa.append(float(win/(posw.sum()*negw.sum())))
                auc_boot[m][k]=float(np.mean(aa)) if aa else np.nan
                brier_boot[m][k]=float(np.mean(bb)) if bb else np.nan
        return auc_boot,brier_boot

    def interval(s,observed,one_sided=False):
        s=np.asarray(s,dtype=float);s=s[np.isfinite(s)]
        if len(s)==0:return {'delta':float(observed),'ci_low':np.nan,'ci_high':np.nan,'valid':0}
        ans={'delta':float(observed),'ci_low':float(np.quantile(s,.025)),
             'ci_high':float(np.quantile(s,.975)),'valid':int(len(s))}
        if one_sided:ans['one_sided_95_lower']=float(np.quantile(s,.05))
        return ans

    
    boot_R,boot_R_brier=family_bootstrap(R_pred,R_TEMPLATES,201)
    boot_O,boot_O_brier=family_bootstrap(oof,('original',),301)
    h2=[];id_ret=[];briers=[]
    for m in ('B2','MT0','MT2'):
        h2.append({'method':m,**interval(boot_R[m]-boot_R['B0'],macro(R_pred,m)-macro(R_pred,'B0'))})
        id_ret.append({'method':m,**interval(boot_O[m]-boot_O['B0'],
                           macro(oof,m,('original',))-macro(oof,'B0',('original',)),True)})
        briers.append({'method':m,**interval(boot_R_brier['B0']-boot_R_brier[m],
             float(np.mean([(g.B0.sub(g.label).pow(2).mean()-g[m].sub(g.label).pow(2).mean())
                            for _,g in R_pred.groupby(['type','variant'])])))})
    h2=pd.DataFrame(h2);ret=pd.DataFrame(id_ret);brier=pd.DataFrame(briers)
    ret['noninferior_at_2pt_onesided_95']=ret.one_sided_95_lower > -ORIG_MARGIN
    h2.to_csv(RESULT_DIR/'H2_paired_OOD_AUROC_bootstrap.csv',index=False)
    ret.to_csv(RESULT_DIR/'original_domain_retention.csv',index=False)
    brier.to_csv(RESULT_DIR/'H2_Brier_gain_bootstrap.csv',index=False)
    flip_acc=[]
    for typ in PRIMARY_TYPES:
        for v in R_TEMPLATES:
            g=flip_df[(flip_df.type==typ)&flip_df.variant.eq(v)&flip_df.change.ne(0)]
            for m in METHODS:
                flip_acc.append({'type':typ,'variant':v,'method':m,'n_flips':len(g),
                                 'correct_direction':float(g[m+'_correct'].mean()) if len(g) else np.nan})
    pd.DataFrame(flip_acc).to_csv(RESULT_DIR/'flip_prediction_diagnostic.csv',index=False)
    print('H1 wording flips:');display(H1.round(3))
    print('H2 held-out differences:');display(h2.round(4))
    print('Original-domain retention:');display(ret.round(4))


if not REPLAY_COMPLETE:
    
    if RUN_TEXT_CONTROL:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        texts_old=train_df.prompt.astype(str).tolist()
        rt=scored[scored.variant.isin(['R1','R2'])].reset_index(drop=True)
        text_scores=np.zeros(len(rt),dtype=float)
        for fold,(tr_idx,te_idx) in enumerate(outer):
            train_tasks=TASKS[tr_idx]
            ix=np.flatnonzero((train_df.task_id.isin(train_tasks)&train_df.variant.eq('original')).to_numpy())
            ev=np.flatnonzero(rt.task_id.map(fold_task).eq(fold).to_numpy())
            vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=6000,
                                sublinear_tf=True,lowercase=True)
            A=vec.fit_transform([texts_old[j] for j in ix])
            yt=train_df.iloc[ix].label.to_numpy(int)
            if np.unique(yt).size<2:raise RuntimeError('Single-class text-control training fold')
            clf=LogisticRegression(C=1.,max_iter=500,solver='liblinear',random_state=SEED)
            clf.fit(A,yt)
            text_scores[ev]=clf.predict_proba(vec.transform(rt.iloc[ev].prompt.astype(str)))[:,1]
        assert set(rt.id)==set(R_pred.id)
        txt=pd.DataFrame({'id':rt.id,'text_tfidf':text_scores})
        R_with_text=R_pred.merge(txt,on='id',validate='one_to_one')
        R_with_text.to_csv(RESULT_DIR/'heldout_predictions_with_text_control.csv',index=False)
        text_metrics=[]
        for (typ,v),g in R_with_text.groupby(['type','variant']):
            text_metrics.append({'type':typ,'variant':v,'char_tfidf_auc':safe_auc(g.label,g.text_tfidf),
                                 'MT2_auc':safe_auc(g.label,g.MT2)})
        pd.DataFrame(text_metrics).to_csv(RESULT_DIR/'char_tfidf_control.csv',index=False)
        print('Text vs hidden state macro AUROC:',macro(R_with_text,'text_tfidf'),macro(R_with_text,'MT2'))







if not REPLAY_COMPLETE:
    
    # Save the reproducibility summary.
    summary={
     'model':MODEL,'model_id':MODEL_ID,'revision_resolved_at_first_run':MODEL_REV,
     'task_families':102,'constraint_types':list(PRIMARY_TYPES),
     'generated_count':len(scored),'capped_count':int(scored.hit_token_cap.sum()),
     'ood_macro_auroc':{m:macro(R_pred,m) for m in METHODS},
     'o_macro_auroc':{m:macro(oof,m,('original',)) for m in METHODS},
     'mt2_minus_b0_ood_auc':float(macro(R_pred,'MT2')-macro(R_pred,'B0')),
     'wording_flip_rate':float(flip_df.change.ne(0).mean()),
     'source_sha256':SOURCE, 'protocol_sha256':sha(canonical(PROTOCOL).encode()),
     'semantic_note':'R1 placeholders says nonempty but official checker counts empty [] as placeholders; original Phase 3 exact text preserved.',
     'hardware_note':'Cache fingerprint contains GPU type to avoid mixing generation environments.',
     'readout_note':'All 20 model-specific probes fitted on generated original/P1-P3 labels from this run, task-disjoint.'}
    atomic_json(RESULT_DIR/'FINAL_SUMMARY.json',summary)
    files_for_export=[RUN/'protocol.json',RUN/'source_provenance.json',RUN/'frozen_prompts.jsonl',
                      RUN/'all_1836_scored.jsonl',RUN/'states_ordered.npy']
    files_for_export+=list(sorted(RESULT_DIR.rglob('*')))
    files_for_export+=list(sorted(P_DIR.glob('*')))
    files_for_export=[x for x in files_for_export if x.is_file()]
    manifest={str(p.relative_to(RUN)):shafile(p) for p in files_for_export}
    atomic_json(RUN/'EXPORT_SHA256_MANIFEST.json',manifest)
    result_zip=RUN/f'IFEvalSimple_{MODEL}_Full_Results.zip'
    with zipfile.ZipFile(result_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files_for_export:z.write(p,arcname=p.relative_to(RUN).as_posix())
        z.write(RUN/'EXPORT_SHA256_MANIFEST.json','EXPORT_SHA256_MANIFEST.json')
    with zipfile.ZipFile(result_zip) as z:
        assert z.testzip() is None
        for name,ref_hash in manifest.items():assert sha(z.read(name))==ref_hash,name
    print('ALL DONE. Results archive:',result_zip)
    print('Files:',len(manifest),'ZIP bytes:',result_zip.stat().st_size)
    print('H2 OOD macro AUROC:',summary['ood_macro_auroc'])
    print('MT2 minus B0:',summary['mt2_minus_b0_ood_auc'])
    print('Flip rate:',summary['wording_flip_rate'])
    print('Original root never modified:',PHI_LEGACY_ROOT/'runs')


if not REPLAY_COMPLETE:
    
    import gc, torch
    if 'model' in globals() and model is not None:
        del model
        model = None
    if 'tokenizer' in globals() and tokenizer is not None:
        del tokenizer
        tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    SIMPLE_RESULTS = str(RESULT_DIR)
    print('IFEval-Simple results:', SIMPLE_RESULTS)
    SIMPLE_CONTEXT={"fitted":fitted.copy(),"fold_task":fold_task.copy(),"types":tuple(PRIMARY_TYPES),"probe_dir":str(PROBE_DIR)}
