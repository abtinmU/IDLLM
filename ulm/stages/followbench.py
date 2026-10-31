"""Part II. FollowBench-derived controlled instruction setting.

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""






if not REPLAY_COMPLETE:
    import os, sys, re, json, hashlib, itertools, math, random, shutil, importlib
    from pathlib import Path
    from collections import Counter, defaultdict
    import numpy as np, pandas as pd, requests, torch, nltk
    from tqdm.auto import tqdm
    from ulm.compat import display
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    from scipy.stats import binomtest

    FOLDER='ULM_'+SPEC.camel+'_FollowBench_Controlled_v1'
    ROOT=ULM_WORKDIR/'FollowBench-derived'/FOLDER
    DATA=ROOT/'data'; CACHE=ROOT/'cache'; RESULTS=ROOT/'results'; MODELS=ROOT/'probes'
    for d in (DATA,CACHE,RESULTS,MODELS):d.mkdir(parents=True,exist_ok=True)

    SEED=20261018
    MODEL_ID=CFG['model_id']
    assert MODEL_REV == json.loads(REV_RECORD.read_text())['revision']
    LAYERS=SPEC_LAYERS; PRIMARY_LAYER=SPEC_PRIMARY_LAYER; FOLDS=5
    GEN_BUDGET=1536; MAX_INPUT_TOKENS=4096
    N_PILOT_GROUPS=30; MIN_STUDY_GROUPS=85; MAX_STUDY_GROUPS=160
    PILOT_MIN_CLASS=5; PILOT_MAX_CAP=.12; PRIMARY_N_TYPES=3
    MIN_TEST_CLASS_PER_FORM=10
    BOOTSTRAPS=2000; Cs=(.03,.1,.3); DELTA_PENALTIES=(4.,16.)
    ID_MARGIN=.02; METHODS=('B0','B2','MT0','MT2')
    RUN_FULL_541=False
    RUN_TEXT_CONTROL=not PAPER_ONLY
    RUN_GEOMETRY=not PAPER_ONLY
    RUN_OTHER_LAYERS=False

    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    def sha(b):return hashlib.sha256(b).hexdigest()
    def shafile(p):
        h=hashlib.sha256()
        with open(p,'rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        return h.hexdigest()
    def canon(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'))
    def save_json(p,x):
        p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.tmp')
        tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf8');os.replace(tmp,p)
    def save_npz(p,**arrays):
        p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_name(p.name+'.tmp')
        with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
        os.replace(tmp,p)
    def freeze(p,x):
        if Path(p).exists():
            assert json.loads(Path(p).read_text())==json.loads(json.dumps(x,sort_keys=True,default=str)), f'Configuration mismatch: {p}. Choose a NEW FOLDER.'
        else:save_json(p,x)
    def auc(y,p):
        y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
        return float(roc_auc_score(y,p)) if len(y)>1 and np.unique(y).size==2 else np.nan
    def holm(ps):
        arr=np.asarray(ps,float);order=np.argsort(arr);out=np.ones(len(arr));v=0.
        for j,i in enumerate(order):v=max(v,min(1.,(len(arr)-j)*arr[i]));out[i]=v
        return out

    def fetch(url,path):
        path=Path(path)
        if path.is_file() and path.stat().st_size:return
        response=requests.get(url,timeout=120);response.raise_for_status()
        assert len(response.content)>50, f'Unexpected short download: {url}'
        tmp=path.with_name(path.name+'.part');tmp.write_bytes(response.content);os.replace(tmp,path)
    print('Workspace:',ROOT)





if not REPLAY_COMPLETE:
    # Load the public FollowBench data.
    FB_FILES=('content','situation','style','format','example','mixed')
    FB_BASE='https://raw.githubusercontent.com/YJiangcm/FollowBench/master/data/'
    FB_BACKUP='https://huggingface.co/datasets/YuxinJiang/FollowBench/resolve/main/data_en/'
    FB_DIR=DATA/'followbench_en';FB_DIR.mkdir(exist_ok=True)
    fb_source_files={}
    for cat in FB_FILES:
        fn=f'{cat}_constraints.json';p=FB_DIR/fn
        if not p.exists():
            try:fetch(FB_BASE+fn,p)
            except requests.RequestException:fetch(FB_BACKUP+fn,p)
        fb_source_files[cat]=p
        assert isinstance(json.loads(p.read_text(encoding='utf8')),list),f'Unexpected {fn} format'

    raw_entries=[]
    for cat,p in fb_source_files.items():
        dat=json.loads(p.read_text(encoding='utf8'))
        for item in dat:
            assert isinstance(item,dict),f'Invalid row in {p}'
            assert {'example_id','level','instruction'}.issubset(item),f'Missing FollowBench columns in {p}'
            ins=item['instruction']
            if not isinstance(ins,str):continue
            raw_entries.append({'category':cat,'example_id':str(item['example_id']),
                'level':int(item['level']),'instruction':ins,'source':str(item.get('source','')),
                'target':item.get('target','')})
    assert len(raw_entries)>300, 'Unexpected FollowBench row count; inspect downloads.'
    assert not pd.DataFrame(raw_entries).duplicated(['category','example_id','level']).any(), 'Duplicate source evolution-level key'

    
    VERIFIER_DIR=DATA/'instruction_following_eval';VERIFIER_DIR.mkdir(exist_ok=True)
    (VERIFIER_DIR/'__init__.py').touch()
    VER_FILES=('instructions.py','instructions_util.py','instructions_registry.py','evaluation_lib.py')
    V_URL='https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/'
    for fn in VER_FILES:fetch(V_URL+fn,VERIFIER_DIR/fn)
    verifier_sha={f:shafile(VERIFIER_DIR/f) for f in VER_FILES}
    for fn,prefix in {'instructions.py':'60e086f5342a','instructions_util.py':'a73797261eee',
                      'instructions_registry.py':'ec92d72c264f'}.items():
        assert verifier_sha[fn].startswith(prefix),f'Verifier changed: {fn}; audit before proceeding'
    if str(DATA) not in sys.path:sys.path.insert(0,str(DATA))
    from instruction_following_eval import instructions_registry
    for pkg in ('punkt','punkt_tab'):
        assert nltk.download(pkg,quiet=True),f'NLTK unavailable: {pkg}'

    def score_one(typ,kw,response,prompt):
        if not isinstance(response,str) or not response.strip():return 0
        obj=instructions_registry.INSTRUCTION_DICT[typ](typ)
        obj.build_description(**kw)
        args=obj.get_instruction_args()
        if args and 'prompt' in args:obj.build_description(prompt=prompt)
        return int(bool(obj.check_following(response)))
    assert score_one('punctuation:no_comma',{},'Hello world','x')==1
    assert score_one('punctuation:no_comma',{},'Hello, world','x')==0

    
    base_families=defaultdict(list)
    for row in raw_entries:
        if row['instruction'].strip():base_families[(row['category'],row['example_id'])].append(row)
    
    
    low_priority={2:0,1:1,3:2,4:3,5:4}
    high_priority={4:0,5:1,3:2,2:3,1:4}
    raw=[]
    for (cat,eid),rows in sorted(base_families.items()):
        valid=[r for r in rows if 45<=len(r['instruction'])<=4800 and r['level']>=1]
        if not valid:continue
        low=min(valid,key=lambda r:(low_priority.get(r['level'],99),len(r['instruction'])))
        high=min(valid,key=lambda r:(high_priority.get(r['level'],99),len(r['instruction'])))
        chosen=[low]+([high] if high['level']!=low['level'] else [])
        for choice in chosen:
            raw.append({'key':len(raw),'prompt':choice['instruction'],'category':cat,
                        'example_id':eid,'level':choice['level'],'source':choice['source'],
                        'family':f'{cat}:{eid}'})
    original_by_key={r['key']:r for r in raw}
    assert len(raw)>0
    pd.DataFrame(raw).to_csv(DATA/'native_family_representatives.csv',index=False)
    summary=pd.DataFrame(raw).groupby(['category','level']).size().rename('n').reset_index()
    display(summary)
    print('Native rows:',len(raw_entries),'| evolution families:',len(base_families),
          '| selected source-level prompts:',len(raw))
    print('Native constraints are NOT automatically graded in this study.')


if not REPLAY_COMPLETE:
    
    
    import json, re
    import numpy as np
    from langdetect import DetectorFactory
    DetectorFactory.seed = SEED

    FORMS = ('original', 'P1', 'P2', 'R1', 'R2')
    CANDIDATE_TYPES = (
        'keywords:forbidden_words',
        'detectable_content:number_placeholders',
        'startend:end_checker',
        'change_case:english_lowercase',
        'change_case:english_capital',
        'punctuation:no_comma',
    )
    TYPE_TO_KW = {
        'detectable_content:number_placeholders': {'num_placeholders': 3},
        'startend:end_checker': {'end_phrase': 'This concludes my response.'},
        'change_case:english_lowercase': {},
        'change_case:english_capital': {},
        'punctuation:no_comma': {},
    }
    FORBID_POOL = (
        'however', 'important', 'example', 'different',
        'consider', 'information', 'specific', 'provide',
    )

    def kw_for(base_key, typ):
        if typ != 'keywords:forbidden_words':
            return dict(TYPE_TO_KW[typ])
        rng = np.random.default_rng(SEED + int(base_key) * 17)
        return {'forbidden_words': sorted(map(str, rng.choice(FORBID_POOL, 2, replace=False)))}

    def wording(typ, kw, form):
        assert form in FORMS
        if typ == 'keywords:forbidden_words':
            words = ', '.join(json.dumps(w, ensure_ascii=False) for w in kw['forbidden_words'])
            opts = (
                f'Do not use any of these words in your response: {words}.',
                f'Never include these words in your answer: {words}.',
                f'Avoid every word listed here in your reply: {words}.',
                f'Excluded vocabulary: none of {words} may appear in your response.',
                f'Ensure your answer contains zero occurrences of these words: {words}.',
            )
        elif typ == 'detectable_content:number_placeholders':
            n = kw['num_placeholders']
            opts = (
                f'Include at least {n} square-bracket placeholders such as [name] in your answer.',
                f'Your response must contain no fewer than {n} [name]-style placeholders.',
                f'Add {n} or more placeholders in square brackets, for example [name].',
                f'Placeholder requirement: use at least {n} items formatted like [name].',
                f'Provide a minimum of {n} square-bracketed placeholders (e.g., [name]) in your reply.',
            )
        elif typ == 'startend:end_checker':
            end = json.dumps(kw['end_phrase'], ensure_ascii=False)
            opts = (
                f'End your response with the exact phrase {end}.',
                f'The final words of your answer must be {end}.',
                f'Required ending: finish with {end}, and write nothing afterward.',
                f'Your reply must conclude with this exact phrase: {end}.',
                f'Make {end} the last phrase in your response.',
            )
        elif typ == 'change_case:english_lowercase':
            
            opts = (
                'Write your entire response in English using only lowercase letters.',
                'Answer in English, and do not use any uppercase letters.',
                'Use English throughout your reply; every letter must be lowercase.',
                'All text in your answer must be English and written in lowercase.',
                'Your response must be in English with no capital letters anywhere.',
            )
        elif typ == 'change_case:english_capital':
            
            opts = (
                'Write your entire response in English using only uppercase letters.',
                'Answer in English, and do not use any lowercase letters.',
                'Use English throughout your reply; every letter must be uppercase.',
                'All text in your answer must be English and written in capital letters.',
                'Your response must be in English with no small letters anywhere.',
            )
        elif typ == 'punctuation:no_comma':
            opts = (
                'Do not use commas anywhere in your response.',
                'Your answer must contain no comma punctuation marks.',
                'Avoid commas throughout your reply.',
                'Use zero commas in your entire response.',
                'Never put a comma in your answer.',
            )
        else:
            raise ValueError(f'Unknown instruction type: {typ}')
        assert len(opts) == len(FORMS) and len(set(opts)) == len(FORMS)
        return opts[FORMS.index(form)]

    def compatible(base, typ, kw):
        
        
        text = base['prompt'].lower()
        if not 45 <= len(text) <= 4800:
            return False
        if typ.startswith('change_case:') and re.search(
            r'(?i)json|xml|csv|case-sensitive|capitaliz|all-caps|all caps|'
            r'uppercase|lowercase|case of', text
        ):
            return False
        if typ == 'punctuation:no_comma' and re.search(
            r'(?i)csv|comma.separated|commas? between|separated by commas?', text
        ):
            return False
        if typ == 'startend:end_checker' and re.search(
            r'(?i)end (?:your |the )?(?:response|answer|reply) with|'
            r'finish (?:your |the )?(?:response|answer|reply) with|'
            r'last line must|only (?:json|xml)', text
        ):
            return False
        if typ == 'detectable_content:number_placeholders' and re.search(
            r'(?i)no square brackets|do not use brackets|only (?:one|a single) word|'
            r'single word answer|only (?:json|xml)', text
        ):
            return False
        if typ == 'keywords:forbidden_words' and any(
            re.search(r'(?i)\b' + re.escape(w) + r'\b', text)
            for w in kw['forbidden_words']
        ):
            return False
        return True

    def overlay(base_key, typ, form):
        base_key = int(base_key)
        base = original_by_key[base_key]
        kw = kw_for(base_key, typ)
        assert compatible(base, typ, kw), (base_key, typ)
        prompt = (
            base['prompt'].rstrip() +
            '\n\nAdditional requirement for your answer:\n' +
            wording(typ, kw, form)
        )
        return {
            'key': base_key, 'type': typ, 'variant': form,
            'prompt': prompt, 'target_kwargs': kw,
            'category': base['category'],
            'native_level': base['level'],
            'family': base['family'],
        }

    
    
    positive_text = {
        'keywords:forbidden_words': 'The children walked through the garden to read a book.',
        'detectable_content:number_placeholders': 'Use [name], [location], and [date] in the document.',
        'startend:end_checker': 'The explanation is complete. This concludes my response.',
        'change_case:english_lowercase': (
            'this is an english sentence. every word uses lowercase letters. '
            'the children are reading a book in a quiet garden.'
        ),
        'change_case:english_capital': (
            'THIS IS AN ENGLISH SENTENCE. EVERY LETTER IS CAPITALIZED. '
            'THE CHILDREN ARE READING A BOOK IN A QUIET GARDEN.'
        ),
        'punctuation:no_comma': 'This is a complete English sentence without commas.',
    }

    
    
    def negative_text(typ, kw):
        if typ == 'keywords:forbidden_words':
            return 'The word ' + kw['forbidden_words'][0] + ' appears here.'
        if typ == 'detectable_content:number_placeholders':
            return 'The example contains no square bracket placeholders.'
        if typ == 'startend:end_checker':
            return 'The response does not end with the required closing phrase.'
        if typ == 'change_case:english_lowercase':
            return 'THIS RESPONSE IS WRITTEN IN CAPITAL ENGLISH LETTERS.'
        if typ == 'change_case:english_capital':
            return 'this response is written in lowercase english letters.'
        if typ == 'punctuation:no_comma':
            return 'This answer has one comma, which must fail.'
        raise ValueError(typ)

    smoke = []
    for typ in CANDIDATE_TYPES:
        k = next(
            (int(r['key']) for r in raw if compatible(r, typ, kw_for(r['key'], typ))),
            None,
        )
        assert k is not None, f'No compatible FollowBench source prompt for {typ}'
        kw = kw_for(k, typ)
        pos = positive_text[typ]
        neg = negative_text(typ, kw)
        records = [overlay(k, typ, v) for v in FORMS]
        assert len({r['prompt'] for r in records}) == len(FORMS)
        assert all(r['target_kwargs'] == kw for r in records)
        assert all(r['prompt'].startswith(original_by_key[k]['prompt'].rstrip()) for r in records)
        for r in records:
            p = int(score_one(typ, kw, pos, r['prompt']))
            n = int(score_one(typ, kw, neg, r['prompt']))
            assert p == 1 and n == 0, (
                f'Verifier mismatch: type={typ}, form={r["variant"]}, '
                f'positive={p}, negative={n}; inspect checker semantics.'
            )
            smoke.append((typ, r['variant'], p, n))

    assert len(smoke) == len(CANDIDATE_TYPES) * len(FORMS)
    print(f'PASS: {len(CANDIDATE_TYPES)} types x {len(FORMS)} formulations; '
          f'{2 * len(smoke)} positive/negative official-checker tests.')
    print('FollowBench source prompts unchanged; native FollowBench constraints are not graded.')


if not REPLAY_COMPLETE:
    
    
    texts=[r['prompt'] for r in raw]
    TF=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=20000,min_df=1).fit_transform(texts)
    sim=cosine_similarity(TF).astype(np.float32)
    parent=list(range(len(raw)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=root(i),root(j)
        if a!=b:parent[max(a,b)]=min(a,b)
    for i in range(len(raw)):
        for j in range(i+1,len(raw)):
            if sim[i,j]>=.91:union(i,j)
    
    first={}
    for i,r in enumerate(raw):
        if r['family'] in first:union(i,first[r['family']])
        else:first[r['family']]=i
    clusters=defaultdict(list)
    for i,r in enumerate(raw):clusters[root(i)].append(int(r['key']))
    GROUP={k:min(ks) for ks in clusters.values() for k in ks}
    rep_by_group={min(ks):min(ks) for ks in clusters.values()}
    keys_by_group={min(ks):sorted(ks,key=lambda k:original_by_key[k]['level']) for ks in clusters.values()}
    coverage=[]
    for g,k in sorted(rep_by_group.items()):
        base=original_by_key[k]
        rec={'group':g,'key':k,'category':base['category'],'native_level':base['level'],
             'source_level_count':len(keys_by_group[g])}
        for t in CANDIDATE_TYPES:
            rec[t]=int(any(compatible(original_by_key[kk],t,kw_for(kk,t)) for kk in keys_by_group[g]))
        coverage.append(rec)
    coverage=pd.DataFrame(coverage)
    coverage.to_csv(DATA/'preflight_type_coverage.csv',index=False)

    
    
    usable=coverage[coverage[list(CANDIDATE_TYPES)].sum(axis=1)>=3]
    assert len(usable)>=N_PILOT_GROUPS+MIN_STUDY_GROUPS,(
        f'Only {len(usable)} independent compatible families; need at least '
        f'{N_PILOT_GROUPS+MIN_STUDY_GROUPS}. Do not run GPU generation.')
    rng=np.random.default_rng(SEED+8)
    chosen=rng.permutation(usable.group.to_numpy(int))
    PILOT_GROUPS=sorted(map(int,chosen[:N_PILOT_GROUPS]))
    STUDY_GROUPS=sorted(map(int,chosen[N_PILOT_GROUPS:N_PILOT_GROUPS+MAX_STUDY_GROUPS]))
    assert len(STUDY_GROUPS)>=MIN_STUDY_GROUPS
    assert not (set(PILOT_GROUPS)&set(STUDY_GROUPS))
    sub=coverage[coverage.group.isin(STUDY_GROUPS)]
    counts={t:int(sub[t].sum()) for t in CANDIDATE_TYPES}
    assert sum(n>=MIN_STUDY_GROUPS-15 for n in counts.values())>=3,(
        f'Fewer than three constraint families have broad source coverage: {counts}')
    freeze(DATA/'task_groups.json',{'near_duplicate_threshold':.91,
      'pilot_groups':PILOT_GROUPS,'study_groups':STUDY_GROUPS,
      'group_map':{str(k):int(v) for k,v in GROUP.items()}})
    proto={'study':'followbench_derived_controlled_v1','source_sha256':{c:shafile(p) for c,p in fb_source_files.items()},
     'model':MODEL_ID,'revision':MODEL_REV,'verifier_sha':verifier_sha,
     'budget':GEN_BUDGET,'max_input_tokens':MAX_INPUT_TOKENS,'layers':list(LAYERS),
     'folds':FOLDS,'seed':SEED,'pilot_groups':PILOT_GROUPS,'study_groups':STUDY_GROUPS,
     'C':list(Cs),'delta_penalties':list(DELTA_PENALTIES),'margin':ID_MARGIN,
     'methods':list(METHODS),'formulations':['original','P1','P2','R1','R2'],
     'label':'added deterministic target constraint ONLY; NOT official FollowBench native constraint satisfaction'}
    freeze(ROOT/'protocol.json',proto)
    print('Preflight PASS. FollowBench evolution/near-duplicate groups:',len(clusters))
    print('Pilot:',len(PILOT_GROUPS),'| study:',len(STUDY_GROUPS),'| compatible task families:',counts)
    print('Study source-level prompts:',sum(len(keys_by_group[g]) for g in STUDY_GROUPS))
    
    assert len(set(GROUP.values()))==len(clusters)





if not REPLAY_COMPLETE:
    # Reuse cached responses and hidden states when valid.
    GEN_CACHE=CACHE/'generations'; STATE_CACHE=CACHE/'states'
    GEN_CACHE.mkdir(exist_ok=True);STATE_CACHE.mkdir(exist_ok=True)
    MODEL_CONTEXT={'id':MODEL_ID,'revision':MODEL_REV,'transformers':'4.53.3',
      'chat_template':'single_user_add_generation_prompt','sampling':'greedy','max_input':MAX_INPUT_TOKENS,
      'attention':'native_sdpa','weight_dtype':('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')}
    model=None;tokenizer=None;INPUT_DEVICE=None

    def ensure_model():
        global model,tokenizer,INPUT_DEVICE
        if model is not None:return
        assert torch.cuda.is_available(), 'Select a GPU runtime for generation.'
        import transformers
        assert transformers.__version__=='4.53.3',f'Expected Transformers 4.53.3, found {transformers.__version__}'
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False)
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,
             torch_dtype=dtype,device_map='auto',trust_remote_code=False,attn_implementation='sdpa')
        model.eval()
        global STOP_IDS
        STOP_IDS=prepare_tokenizer(SPEC,tokenizer,model);INPUT_DEVICE=model.get_input_embeddings().weight.device
        assert INPUT_DEVICE.type=='cuda' and model.config.hidden_size==HIDDEN_WIDTH and len(model.model.layers)==SPEC.num_layers
        assert model.__class__.__module__.startswith('transformers.models.'+SPEC.family+'.')
        batch,n=enc_prompt('Say hello.')
        with torch.inference_mode():
            output=model.generate(**batch,max_new_tokens=3,do_sample=False,use_cache=True,**generation_stop_kwargs(SPEC,tokenizer,STOP_IDS))
            assert output.shape[-1]>n
        del output,batch
        print(SPEC.short_name,'native SDPA smoke test passed.')

    def enc_prompt(prompt):
        encoded=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
           tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt')
        n=int(encoded['input_ids'].shape[-1])
        if n>MAX_INPUT_TOKENS:raise ValueError(f'Input {n} tokens exceeds MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}')
        return {k:v.to(INPUT_DEVICE) for k,v in encoded.items()},n

    def gen_args(r):
        return {'prompt_sha':sha(r['prompt'].encode()),'budget':GEN_BUDGET,'context':MODEL_CONTEXT,
                'verifier':verifier_sha,'type':r.get('type'),'kwargs':r.get('target_kwargs'),
                'category':r.get('category'),'native_level':r.get('native_level')}
    def state_args(r):
        return {'prompt_sha':sha(r['prompt'].encode()),'context':MODEL_CONTEXT,
                'layers':list(LAYERS),'readout':'final_prompt_token'}

    def cache_gen(r):
        args=gen_args(r);p=GEN_CACHE/(sha(canon(args).encode())+'.json')
        if p.exists():
            item=json.loads(p.read_text());assert item['fingerprint']==args
            flags=score_record(r,item['response'])
            assert flags==item['scoring'], 'Scored cache does not match official verifier.'
            return item,True
        return None,False

    def score_record(r,response):
        target=score_one(r['type'],r['target_kwargs'],response,r['prompt'])
        return {'target':target}

    def gen_run(r):
        ensure_model();enc,n=enc_prompt(r['prompt'])
        with torch.inference_mode():
            out=model.generate(**enc,max_new_tokens=GEN_BUDGET,do_sample=False,
                  use_cache=True,**generation_stop_kwargs(SPEC,tokenizer,STOP_IDS))
        seq=out[0,n:];nout=int(seq.numel())
        ends=end_token_ids(SPEC,tokenizer,STOP_IDS)
        cap=bool(nout>=GEN_BUDGET and (not nout or int(seq[-1]) not in ends))
        response=tokenizer.decode(seq,skip_special_tokens=True).strip()
        del seq,enc,out
        return {'response':response,'output_tokens':nout,'input_tokens':n,'capped':cap,
                'scoring':score_record(r,response),'fingerprint':gen_args(r)}

    def generation(r):
        item,hit=cache_gen(r)
        if item is not None:return item,hit
        item=gen_run(r)
        save_json(GEN_CACHE/(sha(canon(item['fingerprint']).encode())+'.json'),item)
        return item,False

    def state(r):
        args=state_args(r);stem=sha(canon(args).encode());p=STATE_CACHE/(stem+'.npz');meta=STATE_CACHE/(stem+'.json')
        if p.exists() and meta.exists():
            assert json.loads(meta.read_text())==args
            with np.load(p,allow_pickle=False) as f:z=np.asarray(f['states'],np.float32)
            assert z.shape==(len(LAYERS),HIDDEN_WIDTH) and np.isfinite(z).all()
            return z,True
        ensure_model();enc,n=enc_prompt(r['prompt'])
        with torch.inference_mode():
            out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
            z=np.stack([out.hidden_states[j][0,-1].float().cpu().numpy() for j in LAYERS]).astype(np.float32)
            del out,enc
        assert z.shape==(len(LAYERS),HIDDEN_WIDTH) and np.isfinite(z).all()
        save_npz(p,states=z.astype(np.float16));save_json(meta,args)
        return z,False

    def result(r,with_state=False):
        item,hit=generation(r)
        return (item,state(r)[0],hit) if with_state else (item,None,hit)
    print('Cache ready. Old generations are never reused under a different token budget.')


if not REPLAY_COMPLETE:
    pilot_rows=[]
    for group,key in tqdm([(g,k) for g in PILOT_GROUPS for k in keys_by_group[g]],desc='Pilot (original/P1 only)'):
        for typ in CANDIDATE_TYPES:
            kw=kw_for(key,typ)
            if not compatible(original_by_key[key],typ,kw):continue
            for form in ('original','P1'):
                r=overlay(key,typ,form);it,_,_=result(r)
                pilot_rows.append({'group':group,'key':key,'type':typ,'variant':form,
                    'target':it['scoring']['target'],'native_level':r['native_level'],'category':r['category'],
                    'capped':int(it['capped'])})
    pilot=pd.DataFrame(pilot_rows)
    assert len(pilot)>0
    pilot.to_csv(RESULTS/'pilot.csv',index=False)
    summary=[]
    for typ,frame in pilot.groupby('type'):
        n=len(frame);success=int(frame.target.sum());failure=n-success;cap=float(frame.capped.mean())
        pos_groups=frame[frame.target.eq(1)].group.nunique()
        neg_groups=frame[frame.target.eq(0)].group.nunique()
        summary.append({'type':typ,'n':n,'success':success,'failure':failure,'cap_rate':cap,
           'groups':frame.group.nunique(),'positive_groups':pos_groups,'negative_groups':neg_groups,
           'eligible':bool(min(success,failure)>=PILOT_MIN_CLASS and
             min(pos_groups,neg_groups)>=4 and cap<=PILOT_MAX_CAP)})
    screen=pd.DataFrame(summary).sort_values('type');screen.to_csv(RESULTS/'pilot_screen.csv',index=False)
    display(screen.assign(cap_pct=100*screen.cap_rate).drop(columns=['cap_rate']).round(2))
    eligible=screen[screen.eligible].copy()
    eligible=eligible[eligible.type.map(counts)>=MIN_STUDY_GROUPS-15].copy()
    if len(eligible)<PRIMARY_N_TYPES:
        raise RuntimeError('Fewer than three balanced, low-truncation types. This is a DATA limitation. '
         'Inspect pilot_screen.csv and test new predeclared constraint categories in a NEW study version.')
    eligible['balance']=np.minimum(eligible.success,eligible.failure)/eligible.n
    eligible=eligible.sort_values(['balance','groups','type'],ascending=[False,False,True])
    PRIMARY_TYPES=eligible.type.tolist()[:PRIMARY_N_TYPES]
    assert len(PRIMARY_TYPES)>=2
    TYPE_TO_ID={t:i for i,t in enumerate(PRIMARY_TYPES)};N_TYPES=len(PRIMARY_TYPES)
    freeze(ROOT/'selected_types.json',{'types':PRIMARY_TYPES,'pilot_sha':shafile(RESULTS/'pilot.csv'),
        'selection':'largest minimum class proportion among pilot-eligible types; no R1/R2 outcome used'})
    print('Selected primary types:',PRIMARY_TYPES)
    print('Study groups remain fully held out:',len(STUDY_GROUPS))


if not REPLAY_COMPLETE:
    train_rows=[];train_z=[]
    for group,key in tqdm([(g,k) for g in STUDY_GROUPS for k in keys_by_group[g]],desc='Training formulation caches'):
        for typ in PRIMARY_TYPES:
            kw=kw_for(key,typ)
            if not compatible(original_by_key[key],typ,kw):continue
            for form in ('original','P1','P2'):
                r=overlay(key,typ,form)
                item,z,_=result(r,with_state=True)
                train_rows.append({'group':group,'key':key,'type':typ,'variant':form,
                  'prompt':r['prompt'],'label':item['scoring']['target'],
                  'category':r['category'],'native_level':r['native_level'],'capped':int(item['capped']),
                  'tokens':int(item['output_tokens'])})
                train_z.append(z)
    TR=pd.DataFrame(train_rows);Ztrain=np.stack(train_z).astype(np.float32)
    assert len(TR)==len(Ztrain) and not TR.duplicated(['key','type','variant']).any()
    assert set(TR.variant)=={'original','P1','P2'}
    TR.to_json(RESULTS/'training_1536.jsonl',orient='records',lines=True,force_ascii=False)
    np.save(RESULTS/'training_states.npy',Ztrain.astype(np.float16))
    cap=TR.groupby(['type','variant']).agg(n=('label','size'),success=('label','sum'),capped=('capped','mean'))
    cap['fail']=cap.n-cap.success;cap.to_csv(RESULTS/'training_label_support.csv');display(cap.assign(capped_pct=100*cap.capped).drop(columns=['capped']).round(2))
    if (cap.capped>.10).any():print('WARNING: high cap rate in some training cells. Include it in limitations; consider a NEW higher-budget protocol.')
    GROUPS=np.array(sorted(TR.group.unique()),dtype=int)
    assert len(GROUPS)>=FOLDS*10, 'Too few independent groups for five-fold evaluation.'
    print('Training examples:',len(TR),'| independent groups:',len(GROUPS))





if not REPLAY_COMPLETE:
    from scipy.special import expit
    # Fit the four readout methods on grouped folds.
    METHODS=('B0','B2','MT0','MT2')
    N_TYPES=len(PRIMARY_TYPES)
    TR=TR.reset_index(drop=True).copy()
    X=Ztrain[:,LAYERS.index(PRIMARY_LAYER)].astype(np.float32)
    assert len(TR)==len(X)
    GROUPS=np.array(sorted(set(TR.group)))
    outer=list(KFold(n_splits=FOLDS,shuffle=True,random_state=SEED).split(GROUPS))
    fold_for_group={int(GROUPS[i]):fold for fold,(_,te) in enumerate(outer) for i in te}
    assert len(fold_for_group)==len(GROUPS)
    for tr,te in outer:assert set(GROUPS[tr]).isdisjoint(GROUPS[te])

    class Probe:
        def __init__(self,mean,scale,w,delta,offset,b):
            self.mean=np.asarray(mean,np.float32);self.scale=np.asarray(scale,np.float32)
            self.w=np.asarray(w,np.float32);self.delta=np.asarray(delta,np.float32)
            self.offset=np.asarray(offset,np.float32);self.b=float(b)
        def predict(self,x,t):
            x=np.asarray(x,np.float32)
            if x.ndim==1:x=x[None,:]
            t=np.asarray(t,int).reshape(-1)
            assert len(x)==len(t) and x.shape[1]==len(self.w)
            z=(x-self.mean)/self.scale
            logits=np.sum(z*(self.w[None,:]+self.delta[t]),axis=1)+self.b+self.offset[t]
            return expit(np.clip(logits,-35,35))
        def save(self,path):
            tmp=path.with_suffix('.tmp')
            with open(tmp,'wb') as f:np.savez_compressed(f,mean=self.mean,scale=self.scale,
               w=self.w,delta=self.delta,offset=self.offset,b=np.array(self.b))
            os.replace(tmp,path)
        @classmethod
        def load(cls,path):
            with np.load(path,allow_pickle=False) as q:
                return cls(q['mean'],q['scale'],q['w'],q['delta'],q['offset'],float(q['b']))

    def fit_probe(train_groups,method,C,penalty=16.,layer=PRIMARY_LAYER,max_iter=90):
        tr=TR.group.isin(train_groups) & (TR.variant.eq('original') if method in ('B0','MT0') else True)
        idx=np.flatnonzero(tr.to_numpy());ixorig=np.flatnonzero((TR.group.isin(train_groups)&TR.variant.eq('original')).to_numpy())
        assert len(idx)>0 and len(ixorig)>0
        sc=StandardScaler().fit(X[ixorig] if layer==PRIMARY_LAYER else Ztrain[ixorig,LAYERS.index(layer)].astype(np.float32))
        XX=X[idx] if layer==PRIMARY_LAYER else Ztrain[idx,LAYERS.index(layer)].astype(np.float32)
        z=((XX-sc.mean_)/np.maximum(sc.scale_,1e-6)).astype(np.float32)
        y=TR.iloc[idx].label.to_numpy(np.float32)
        tt=TR.iloc[idx].type.map(TYPE_TO_ID).to_numpy(np.int64)
        if len(np.unique(y))<2:
            prior=np.clip(float(y.mean()),1e-5,1-1e-5)
            return Probe(sc.mean_,np.maximum(sc.scale_,1e-6),np.zeros(X.shape[1]),
                np.zeros((N_TYPES,X.shape[1])),np.zeros(N_TYPES),np.log(prior/(1-prior)))
        
        g=TR.iloc[idx].groupby(['group','type']).size().to_dict()
        wt=np.array([1/g[(int(gg),str(tt))] for gg,tt in TR.iloc[idx][['group','type']].itertuples(index=False,name=None)],np.float32);wt/=wt.sum()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xx=torch.as_tensor(z,device=device); yy=torch.as_tensor(y,device=device)
        ids=torch.as_tensor(tt,device=device);wts=torch.as_tensor(wt,device=device)
        w=torch.nn.Parameter(torch.zeros(X.shape[1],device=device))
        b=torch.nn.Parameter(torch.zeros((),device=device))
        mt=method.startswith('MT')
        delta=torch.nn.Parameter(torch.zeros((N_TYPES,HIDDEN_WIDTH),device=device)) if mt else None
        offsets=torch.nn.Parameter(torch.zeros(N_TYPES,device=device)) if mt else None
        params=[w,b]+([delta,offsets] if mt else [])
        opt=torch.optim.LBFGS(params,lr=1.,max_iter=max_iter,line_search_fn='strong_wolfe')
        ridge=1.0/(2*float(C)*int(len(g)))
        def closure():
            opt.zero_grad()
            v=torch.sum(xx*(w[None,:]+delta[ids] if mt else w[None,:]),dim=1)+b
            if mt:v=v+offsets[ids]
            loss=(wts*torch.nn.functional.binary_cross_entropy_with_logits(v,yy,reduction='none')).sum()+ridge*w.square().sum()
            if mt:loss=loss+ridge*float(penalty)*delta.square().sum()+ridge*offsets.square().sum()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite logistic objective')
            loss.backward();return loss
        opt.step(closure)
        pp=Probe(sc.mean_,np.maximum(sc.scale_,1e-6),w.detach().cpu().numpy(),
            delta.detach().cpu().numpy() if mt else np.zeros((N_TYPES,X.shape[1])),
            offsets.detach().cpu().numpy() if mt else np.zeros(N_TYPES),b.detach().cpu().item())
        del xx,yy,ids,wts,opt
        return pp

    def macro_validation(probe,groups,variants,layer=PRIMARY_LAYER):
        mask=(TR.group.isin(groups)&TR.variant.isin(variants)).to_numpy();idx=np.flatnonzero(mask)
        if not len(idx):return np.nan
        XX=X if layer==PRIMARY_LAYER else Ztrain[:,LAYERS.index(layer)].astype(np.float32)
        ss=[]
        for typ in PRIMARY_TYPES:
            for v in variants:
                sub=idx[(TR.iloc[idx].type.eq(typ)&TR.iloc[idx].variant.eq(v)).to_numpy()]
                if not len(sub):continue
                yy=TR.iloc[sub].label.to_numpy(int)
                if len(np.unique(yy))<2:continue
                pp=probe.predict(XX[sub],np.full(len(sub),TYPE_TO_ID[typ]))
                ss.append(auc(yy,pp))
        return float(np.nanmean(ss)) if ss else np.nan

    FROZEN_FP=sha(canon({'dataset':proto['source_sha256'],'protocol':proto,
     'selected_types':PRIMARY_TYPES,'selected_keys':sorted(set(TR.key)),
     'label_file_sha':shafile(RESULTS/'training_1536.jsonl'),
     'layers':LAYERS,'methods':METHODS,'software':'torch_LBFGS_v3'}).encode())
    freeze(MODELS/'probe_model_fingerprint.json',{'sha256':FROZEN_FP})
    fitted={};training_log=[]
    for fold,(trix,teix) in enumerate(outer):
        trainval=GROUPS[trix];test=GROUPS[teix]
        inner_tr,inner_val=train_test_split(trainval,test_size=.25,random_state=SEED+fold)
        base_id=None
        for method in METHODS:
            path=MODELS/f'{fold}_{method}.npz';meta=MODELS/f'{fold}_{method}.json'
            if path.exists() and meta.exists():
                info=json.loads(meta.read_text());assert info['fp']==FROZEN_FP
                pp=Probe.load(path);fitted[(fold,method)]=pp;training_log.append(info)
                if method=='B0':base_id=info['id_auc']
                continue
            choices=[(c,16.) for c in Cs] if not method.startswith('MT') else list(itertools.product(Cs,DELTA_PENALTIES))
            scored=[]
            for c,p in choices:
                tmp=fit_probe(inner_tr,method,c,p)
                shift=macro_validation(tmp,inner_val,['P1','P2'])
                original=macro_validation(tmp,inner_val,['original'])
                if np.isfinite(original) and np.isfinite(shift):scored.append((shift,original,c,p))
            if not scored:
                print('Warning: validation has too few within-type class pairs; using prior-fixed regularization for',fold,method)
                scored=[(0.5,0.5,Cs[1],16.)]
            feasible=[z for z in scored if base_id is None or z[1]>=base_id-ID_MARGIN]
            best=max(feasible or scored,key=lambda z:(z[0],z[1],-z[2],-z[3]))
            pp=fit_probe(trainval,method,best[2],best[3],max_iter=110)
            pp.save(path)
            info={'fold':fold,'method':method,'fp':FROZEN_FP,'C':best[2],
                  'delta_penalty':best[3],'validation_shift_auc':best[0],'id_auc':best[1],
                  'constraint_feasible':bool(feasible),'train_groups':list(map(int,trainval)),
                  'heldout_groups':list(map(int,test))}
            save_json(meta,info);fitted[(fold,method)]=pp;training_log.append(info)
            if method=='B0':base_id=best[1]
    pd.DataFrame(training_log).drop(columns=['train_groups','heldout_groups']).to_csv(RESULTS/'probe_tuning.csv',index=False)
    assert len(fitted)==FOLDS*len(METHODS)
    print('All fitted probes:',len(fitted),' | frozen before final R1/R2 scoring')


if not REPLAY_COMPLETE:
    
    test_rows=[];test_z=[]
    for group,key in tqdm([(int(g),k) for g in GROUPS for k in keys_by_group[int(g)]],desc='Final unseen R1/R2'):
        for typ in PRIMARY_TYPES:
            if not compatible(original_by_key[key],typ,kw_for(key,typ)):continue
            for form in ('R1','R2'):
                r=overlay(key,typ,form)
                item,z,_=result(r,with_state=True)
                test_rows.append({'group':int(group),'key':key,'type':typ,'variant':form,
                     'prompt':r['prompt'],'label':item['scoring']['target'],
                     'category':r['category'],'native_level':r['native_level'],'capped':int(item['capped']),
                     'tokens':int(item['output_tokens'])})
                test_z.append(z)
    TE=pd.DataFrame(test_rows);Ztest=np.stack(test_z).astype(np.float32)
    assert len(TE)==len(Ztest) and not TE.duplicated(['key','type','variant']).any()
    TE.to_json(RESULTS/'unseen_R_scored.jsonl',orient='records',lines=True,force_ascii=False)
    np.save(RESULTS/'unseen_R_states.npy',Ztest.astype(np.float16))
    print('Held-out test predictions:',len(TE),'| original independent groups:',TE.group.nunique())





if not REPLAY_COMPLETE:
    
    PRED=[]
    for domain,frame,Z in [('ID',TR,Ztrain),('OOD',TE,Ztest)]:
        for ix,row in enumerate(frame.itertuples(index=False)):
            if domain=='ID' and row.variant!='original':continue
            fold=fold_for_group[int(row.group)]
            layer=Z[ix,LAYERS.index(PRIMARY_LAYER)]
            rec={'domain':domain,'group':int(row.group),'key':int(row.key),'type':row.type,
                 'variant':row.variant,'label':int(row.label),'category':row.category,'native_level':int(row.native_level),
                 'capped':int(row.capped),'tokens':int(row.tokens),'fold':fold,'prompt':row.prompt}
            for method in METHODS:
                rec[method]=float(fitted[(fold,method)].predict(layer,[TYPE_TO_ID[row.type]])[0])
            PRED.append(rec)
    P=pd.DataFrame(PRED);P.to_csv(RESULTS/'oof_predictions.csv',index=False)
    ID=P[P.domain.eq('ID')].copy();OOD=P[P.domain.eq('OOD')].copy()

    def macro_auc(frame,method):
        vals=[]
        target='label'
        for (_, _),s in frame.groupby(['type','variant']):
            v=auc(s[target],s[method])
            if np.isfinite(v):vals.append(v)
        return float(np.mean(vals)) if vals else np.nan
    summary=[]
    for domain,f in [('ID',ID),('OOD',OOD)]:
        for method in METHODS:
            summary.append({'domain':domain,'method':method,'macro_auc':macro_auc(f,method),
              'Brier':float(brier_score_loss(f.label,f[method])),
              'logloss':float(log_loss(f.label,np.clip(f[method],1e-7,1-1e-7),labels=[0,1]))})
    pd.DataFrame(summary).to_csv(RESULTS/'primary_metrics.csv',index=False)
    display(pd.DataFrame(summary).assign(AUROC=lambda d:100*d.macro_auc,Brier_x100=lambda d:100*d.Brier)[['domain','method','AUROC','Brier_x100']].round(2))

    cells=OOD.groupby(['type','variant']).agg(n=('label','size'),success=('label','sum'),capped=('capped','mean'))
    cells['failure']=cells.n-cells.success
    cells['auc_defined']=((cells.success>0)&(cells.failure>0))
    
    positive_groups=OOD[OOD.label.eq(1)].groupby(['type','variant']).group.nunique()
    negative_groups=OOD[OOD.label.eq(0)].groupby(['type','variant']).group.nunique()
    cells['positive_groups']=positive_groups.reindex(cells.index,fill_value=0)
    cells['negative_groups']=negative_groups.reindex(cells.index,fill_value=0)
    cells['adequate_class_support']=((cells.success>=MIN_TEST_CLASS_PER_FORM)&
     (cells.failure>=MIN_TEST_CLASS_PER_FORM)&
     (cells.positive_groups>=8)&(cells.negative_groups>=8))
    cells['acceptable_cap_rate']=cells.capped.le(PILOT_MAX_CAP+1e-12)
    cells.to_csv(RESULTS/'OOD_class_support.csv');display(cells.assign(cap_pct=100*cells.capped).drop(columns=['capped']).round(2))
    PRIMARY_EVALUABLE=bool(cells.adequate_class_support.all() and cells.acceptable_cap_rate.all())
    if not PRIMARY_EVALUABLE:
        print('CAUTION: held-out cells have insufficient independent binary class support or excess truncation. Do not claim confirmatory AUROC.')

    
    
    h1=[]
    for typ in PRIMARY_TYPES:
        o=ID[ID.type.eq(typ)][['key','group','label','capped']]
        for form in ('R1','R2'):
            q=OOD[(OOD.type.eq(typ))&(OOD.variant.eq(form))][['key','group','label','capped']]
            g=o.merge(q,on=['key','group'],suffixes=('_original','_new'),validate='one_to_one')
            if not len(g):continue
            a=g.label_original.to_numpy(int);b=g.label_new.to_numpy(int)
            up=int(((a==0)&(b==1)).sum());down=int(((a==1)&(b==0)).sum())
            means=(g.label_new-g.label_original).groupby(g.group).mean().to_numpy(float)
            observed=abs(float(means.mean()))
            rs=np.random.default_rng(SEED+len(h1)*127)
            nperm=20000
            signs=rs.choice([-1.,1.],size=(nperm,len(means)))
            null_abs=np.abs((signs*means[None,:]).mean(axis=1))
            p_group=float((1+(null_abs>=observed-1e-12).sum())/(nperm+1))
            h1.append({'type':typ,'form':form,'n':len(g),'independent_groups':g.group.nunique(),
              'success_original':float(a.mean()),'success_new':float(b.mean()),
              'delta':float((b-a).mean()),'failure_to_success':up,'success_to_failure':down,
              'group_randomization_p':p_group,'original_capped':float(g.capped_original.mean()),
              'new_capped':float(g.capped_new.mean()),'flip_rate':float((a!=b).mean())})
    h1=pd.DataFrame(h1)
    h1['holm_p']=holm(h1.group_randomization_p)
    h1.to_csv(RESULTS/'H1_paired.csv',index=False)
    h1_display=h1.copy()
    for col in ['success_original','success_new','delta','original_capped','new_capped','flip_rate']:
        h1_display[col+'_pct']=100*h1_display.pop(col)
    display(h1_display.round(2))

    
    def boot_delta(frame,method,baseline='B0',nboot=BOOTSTRAPS,seed=SEED):
        rg=np.random.default_rng(seed);groups=np.array(sorted(frame.group.unique()),int)
        loc={g:i for i,g in enumerate(groups)}
        series=[]
        for _,t in frame.groupby(['type','variant']):
            y=t.label.to_numpy(int);a=t[method].to_numpy(float);b=t[baseline].to_numpy(float)
            g=np.array([loc[int(k)] for k in t.group],int)
            series.append((y,a,b,g))
        values=[]
        for _ in range(nboot):
            cnt=rg.multinomial(len(groups),np.ones(len(groups))/len(groups))
            dif=[]
            for y,a,b,g in series:
                w=cnt[g]
                if w[y==0].sum()==0 or w[y==1].sum()==0:continue
                dif.append(roc_auc_score(y,a,sample_weight=w)-roc_auc_score(y,b,sample_weight=w))
            if dif:values.append(float(np.mean(dif)))
        values=np.asarray(values)
        return {'gain':macro_auc(frame,method)-macro_auc(frame,baseline),
                'ci_low':float(np.quantile(values,.025)) if len(values) else np.nan,
                'ci_high':float(np.quantile(values,.975)) if len(values) else np.nan,
                'one_sided_lower_95':float(np.quantile(values,.05)) if len(values) else np.nan,
                'valid_bootstraps':int(len(values))}
    contrasts=[('B2','B0'),('MT0','B0'),('MT2','B0'),('MT2','B2'),('MT2','MT0')]
    rows=[]
    for domain,f in [('OOD',OOD),('ID',ID)]:
        for j,(a,b) in enumerate(contrasts):
            rows.append({'domain':domain,'method':a,'baseline':b,
                        **boot_delta(f,a,b,seed=SEED+100*j+(domain=='ID'))})
    H2=pd.DataFrame(rows);H2['significant_95']=H2.ci_low.gt(0)
    H2['ID_noninferior_margin_002']=np.where(H2.domain.eq('ID'),H2.one_sided_lower_95>-ID_MARGIN,np.nan)
    H2.to_csv(RESULTS/'H2_bootstrap.csv',index=False)
    H2_display=H2.copy()
    for col in ['gain','ci_low','ci_high','one_sided_lower_95']:
        H2_display[col+'_points']=100*H2_display.pop(col)
    display(H2_display.round(2))

    sub=[]
    for domain,frame in [('ID',ID),('OOD',OOD)]:
        for (typ,form),g in frame.groupby(['type','variant']):
            for m in METHODS:
                sub.append({'domain':domain,'type':typ,'variant':form,'method':m,'n':len(g),
                  'positive':int(g.label.sum()),'negative':int(len(g)-g.label.sum()),
                  'auc':auc(g.label,g[m]),'Brier':float(brier_score_loss(g.label,g[m]))})
    pd.DataFrame(sub).to_csv(RESULTS/'per_type_formulation.csv',index=False)

    fold_rows=[]
    for (typ,form,fold),g in OOD.groupby(['type','variant','fold']):
        for m in METHODS:
            fold_rows.append({'type':typ,'form':form,'fold':int(fold),'method':m,'n':len(g),
                      'success':int(g.label.sum()),'auc':auc(g.label,g[m])})
    pd.DataFrame(fold_rows).to_csv(RESULTS/'within_fold_AUROC.csv',index=False)

    paired=[]
    for row in OOD.itertuples(index=False):
        old=ID[(ID.key==row.key)&(ID.type==row.type)]
        assert len(old)==1
        old=old.iloc[0];flip=int(row.label)-int(old.label)
        rr={'group':int(row.group),'key':int(row.key),'type':row.type,'form':row.variant,
            'behavior_change':flip,'changed':int(flip!=0)}
        for m in METHODS:
            delta=float(getattr(row,m)-old[m]);rr[m+'_score_shift']=delta
            rr[m+'_flip_correct']=int(delta*flip>0) if flip else np.nan
            rr[m+'_stable_delta']=abs(delta) if not flip else np.nan
        paired.append(rr)
    pd.DataFrame(paired).to_csv(RESULTS/'behavioral_flip_tracking.csv',index=False)
    print('Main H1/H2 analysis complete.')


if not REPLAY_COMPLETE:
    control=OOD.copy()
    control['type_prior']=np.nan
    for i,row in control.iterrows():
        fold=int(row.fold);train_groups=set(GROUPS[outer[fold][0]])
        subset=TR[(TR.group.isin(train_groups))&(TR.type.eq(row.type))&(TR.variant.eq('original'))]
        control.at[i,'type_prior']=float((subset.label.sum()+1)/(len(subset)+2))
    if RUN_TEXT_CONTROL:
        control['char_ngram']=np.nan
        for fold,(itr,ite) in enumerate(outer):
            train_groups=GROUPS[itr];test_groups=GROUPS[ite]
            train=TR[(TR.group.isin(train_groups))&(TR.variant.eq('original'))]
            sub=control[control.group.isin(test_groups)]
            if len(sub)==0:continue
            vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=10000,sublinear_tf=True)
            a=vec.fit_transform(train.prompt);b=vec.transform(sub.prompt)
            if train.label.nunique()<2:probs=np.full(len(sub),float(train.label.mean()))
            else:
                clf=LogisticRegression(C=1.,solver='liblinear',max_iter=500)
                clf.fit(a,train.label);probs=clf.predict_proba(b)[:,1]
            control.loc[sub.index,'char_ngram']=probs
    assert control.type_prior.notna().all()
    if RUN_TEXT_CONTROL:assert control.char_ngram.notna().all()
    control.to_csv(RESULTS/'text_controls_predictions.csv',index=False)
    ms=list(METHODS)+['type_prior']+(['char_ngram'] if RUN_TEXT_CONTROL else [])
    ctrl=[{'model':m,'macro_auc':macro_auc(control,m),'Brier':float(brier_score_loss(control.label,control[m]))} for m in ms]
    pd.DataFrame(ctrl).to_csv(RESULTS/'text_controls_metrics.csv',index=False);display(pd.DataFrame(ctrl).assign(AUROC=lambda d:100*d.macro_auc,Brier_x100=lambda d:100*d.Brier)[['model','AUROC','Brier_x100']].round(2))

    if RUN_GEOMETRY:
        geom=[]
        for fold,(trix,teix) in enumerate(outer):
            gtr=GROUPS[trix]
            for typ in PRIMARY_TYPES:
                ix_type=TR.type.eq(typ)
                for form in ('P1','P2'):
                    
                    
                    vectors=[]
                    for v in ('original',form):
                        z=TR[ix_type&TR.variant.eq(v)&TR.group.isin(gtr)]
                        ix=z.index.to_numpy()
                        if z.label.nunique()<2:vectors=[];break
                        sc=StandardScaler().fit(Ztrain[ix,LAYERS.index(PRIMARY_LAYER)])
                        xx=sc.transform(Ztrain[ix,LAYERS.index(PRIMARY_LAYER)])
                        clf=LogisticRegression(C=.1,solver='liblinear',max_iter=1000)
                        clf.fit(xx,z.label.to_numpy(int))
                        
                        vdir=clf.coef_[0]/np.maximum(sc.scale_,1e-6)
                        vectors.append(vdir)
                    if len(vectors)==2:
                        a,b=vectors;cos=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
                        geom.append({'fold':fold,'type':typ,'form':form,'direction_cosine':cos})
        pd.DataFrame(geom).to_csv(RESULTS/'direction_geometry.csv',index=False)
        print('Direction comparisons:',len(geom))


if not REPLAY_COMPLETE:
    
    source_cells=[]
    for (category,level,form),frame in OOD.groupby(['category','native_level','variant']):
        source_cells.append({'category':category,'native_level':int(level),'form':form,
            'n':len(frame),'success_pct':100*float(frame.label.mean()),
            'cap_pct':100*float(frame.capped.mean()),
            'B0_auc_pct':100*auc(frame.label,frame.B0),
            'MT2_auc_pct':100*auc(frame.label,frame.MT2)})
    pd.DataFrame(source_cells).to_csv(RESULTS/'followbench_source_category_level.csv',index=False)

    
    report=pd.read_csv(RESULTS/'primary_metrics.csv')
    report['AUROC_0_100']=100*report.macro_auc
    report['Brier_x100']=100*report.Brier
    display(report[['domain','method','AUROC_0_100','Brier_x100']].round(2))

    contrasts=pd.read_csv(RESULTS/'H2_bootstrap.csv')
    for k in ['gain','ci_low','ci_high','one_sided_lower_95']:
        contrasts[k+'_points']=100*contrasts[k]
    display(contrasts[contrasts.domain.eq('OOD')][
        ['method','baseline','gain_points','ci_low_points','ci_high_points']].round(2))

    manifest={'protocol':proto,'primary_label':'compliance with added deterministic IFEval-verifiable output constraint',
     'native_followbench_scoring':'NOT RUN. Native constraints often require LLM judging.',
     'n_native_rows':len(raw_entries),'n_native_families':len(base_families),
     'selected_types':PRIMARY_TYPES,'n_study_groups':len(GROUPS),
     'n_ood':len(OOD),'evaluable':PRIMARY_EVALUABLE,
     'study_warning':'Controlled FollowBench-derived extension, not native FollowBench benchmark HSR/SSR.',
     'bootstrap_unit':'source evolution family/near-duplicate cluster',
     'H1_p_values':'20k paired cluster sign-flip permutations, Holm adjusted',
     'primary_comparison':'MT2 vs B0 on heldout R1/R2 macro within-type AUROC'}
    save_json(ROOT/'RESULT_MANIFEST.json',manifest)
    archive=shutil.make_archive(str(ROOT/'FollowBench_Controlled_Results'),'zip',ROOT,'results')
    print('Results archive:',archive,'| All persistent caches:',ROOT)


if not REPLAY_COMPLETE:
    
    import gc, torch
    if 'model' in globals() and model is not None:
        del model
        model = None
    if 'tokenizer' in globals() and tokenizer is not None:
        del tokenizer
        tokenizer = None
    gc.collect(); torch.cuda.empty_cache()
    FOLLOWBENCH_RESULTS = str(RESULTS)
    print('FollowBench-derived results:', FOLLOWBENCH_RESULTS)
