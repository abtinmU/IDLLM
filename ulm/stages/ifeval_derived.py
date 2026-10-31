"""Part II. IFEval-derived controlled instruction setting (off unless --run-ifeval-derived).

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""








IFEVAL_RESULTS = None
# This optional stage runs only when requested.
if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    import os, sys, re, io, json, hashlib, gc, itertools, warnings, shutil, zipfile
    from pathlib import Path
    from collections import Counter, defaultdict
    from datetime import datetime, timezone
    import numpy as np, pandas as pd, torch, requests
    from tqdm.auto import tqdm
    from ulm.compat import display
    from scipy.special import expit
    from scipy.stats import binomtest
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, balanced_accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics.pairwise import cosine_similarity
    import matplotlib.pyplot as plt

    ROOT = ULM_WORKDIR/'IFEval-derived'
    OLD = ROOT/'unused_original_phase1'
    OLD_PHASE3 = ROOT/'unused_old_phase3'  
    OUT = ROOT/(SPEC.tag+'_ifeval_derived_v1')
    DATA = OUT/'data'; CACHE = OUT/'cache'; RESULTS = OUT/'results'; MODELS = OUT/'models'
    for p in (DATA,CACHE,RESULTS,MODELS):p.mkdir(parents=True,exist_ok=True)
    
    SEED = 20261017
    GEN_BUDGET = 1024
    MAX_INPUT_TOKENS = 8192
    LAYERS = SPEC_LAYERS
    LAYER = SPEC_PRIMARY_LAYER
    FOLDS = 5
    Cs = (.03,.1,.3)
    DELTA_PENALTIES=(4.,16.)
    MARGIN_ID=.02
    BOOTSTRAPS=1500
    RUN_FULL_541_BENCHMARK=False  
    RUN_TEXT_CONTROL=True
    RUN_LAYER_SENSITIVITY=False  
    RUN_GPU = True                
    
    MIN_PER_TYPE = 12
    MIN_PILOT_PER_LABEL = 3
    PILOT_FRACTION = .20
    MAX_TRAIN_CAP_RATE = .12
    np.random.seed(SEED);torch.manual_seed(SEED)

    def sha(data):return hashlib.sha256(data).hexdigest()
    def shafile(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        return h.hexdigest()
    def cj(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
    def write_json(path,x):
        path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
        temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')
        os.replace(temp,path)
    def freeze(path,x):
        if path.exists():
            assert json.loads(path.read_text())==json.loads(json.dumps(x,sort_keys=True,default=str)), f'Frozen configuration changed: {path}. Use a new output directory.'
        else:write_json(path,x)
    def safe_auc(y,s):
        y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
        return float(roc_auc_score(y,s)) if len(y)>1 and len(np.unique(y))==2 else np.nan
    def exact_mc(a,b):
        a=np.asarray(a,int);b=np.asarray(b,int)
        u=int(((a==0)&(b==1)).sum());d=int(((a==1)&(b==0)).sum())
        return (float(binomtest(u,u+d).pvalue) if u+d else 1.,u,d)
    def holm(p):
        x=np.asarray(p,float);order=np.argsort(x);y=np.ones(len(x));m=0.
        for j,i in enumerate(order):m=max(m,min(1.,(len(x)-j)*x[i]));y[i]=m
        return y

    MODEL_ID=CFG['model_id']
    assert MODEL_REV == json.loads(REV_RECORD.read_text())['revision']
    assert MODEL_REV and len(MODEL_REV)>=12
    CHECKER=ROOT/'data'/'instruction_following_eval'
    VER_FILES=('instructions.py','instructions_util.py','instructions_registry.py')
    CHECKER.mkdir(parents=True,exist_ok=True)
    (CHECKER/'__init__.py').touch(exist_ok=True)
    CHECKER_URL='https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/'
    for fn in VER_FILES:
        local=CHECKER/fn
        if not local.is_file() and os.environ.get('ULM_CHECKER_SOURCE_DIR'):
            pinned=Path(os.environ['ULM_CHECKER_SOURCE_DIR'])/fn
            if pinned.is_file():local.write_bytes(pinned.read_bytes())
        if not local.is_file():
            reply=requests.get(CHECKER_URL+fn,timeout=120);reply.raise_for_status()
            temp=local.with_suffix('.py.part');temp.write_bytes(reply.content);os.replace(temp,local)
    assert all((CHECKER/f).is_file() for f in VER_FILES), 'Public checker download incomplete'
    
    assert {f:shafile(CHECKER/f) for f in VER_FILES} == {f:shafile(DATA_DIR/'instruction_following_eval'/f) for f in VER_FILES}, (
        'Inconsistent IFEval checker files across source settings')
    VER_HASH={f:shafile(CHECKER/f) for f in VER_FILES}
    if str(ROOT/'data') not in sys.path:sys.path.insert(0,str(ROOT/'data'))
    from instruction_following_eval import instructions_registry
    import nltk
    for pkg in ('punkt','punkt_tab'):
        assert nltk.download(pkg,quiet=True),f'NLTK {pkg} unavailable.'

    PROTOCOL={'dataset':'official_google_ifeval_541','model':MODEL_ID,'model_revision':MODEL_REV,
     'generation_budget':GEN_BUDGET,'input_budget':MAX_INPUT_TOKENS,
     'layers':list(LAYERS),'primary_layer':LAYER,'folds':FOLDS,'seed':SEED,
     'C_grid':Cs,'delta_penalties':DELTA_PENALTIES,'id_margin':MARGIN_ID,
     'strict_checker_files':VER_HASH,'selection':'conservative_exact_span_rewrites_single_constraint_v1',
     'min_type':MIN_PER_TYPE,'min_pilot_label':MIN_PILOT_PER_LABEL,
     'pilot_fraction':PILOT_FRACTION,'max_cap':MAX_TRAIN_CAP_RATE,
     'templates':['original','P1','P2','R1','R2']}
    freeze(OUT/'frozen_protocol.json',PROTOCOL)
    print('Dataset/model:',MODEL_ID,MODEL_REV[:16]);print('New output location:',OUT)





if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    ORIGINAL_URL='https://raw.githubusercontent.com/google-research/google-research/master/instruction_following_eval/data/input_data.jsonl'
    src=DATA/'original_ifeval_541.jsonl'
    if not src.exists():
        req=requests.get(ORIGINAL_URL,timeout=90)
        req.raise_for_status()
        parsed=[json.loads(z) for z in req.text.splitlines() if z.strip()]
        assert len(parsed)==541 and len({int(q['key']) for q in parsed})==541
        src.write_text(req.text,encoding='utf-8')
    raw=[json.loads(z) for z in src.read_text(encoding='utf-8').splitlines() if z.strip()]
    assert len(raw)==541 and len({int(x['key']) for x in raw})==541
    for r in raw:
        assert isinstance(r['prompt'],str) and r['prompt'].strip()
        assert len(r['instruction_id_list'])==len(r['kwargs'])>0
        assert all(k in instructions_registry.INSTRUCTION_DICT for k in r['instruction_id_list'])
        assert all(x is None or isinstance(x,dict) for x in r['kwargs'])
        r['kwargs']=[{} if x is None else x for x in r['kwargs']]
    SOURCE_SHA=shafile(src)
    freeze(DATA/'dataset_fingerprint.json',{'source':ORIGINAL_URL,'sha256':SOURCE_SHA,'n':541})
    base=pd.DataFrame([{'key':int(x['key']),'prompt':x['prompt'],'type':x['instruction_id_list'][0],
        'n_constraints':len(x['instruction_id_list']),'instruction_ids':x['instruction_id_list'],
        'kwargs':x['kwargs'],'prompt_sha':sha(x['prompt'].encode('utf-8'))} for x in raw])
    base.to_json(DATA/'original_541_metadata.jsonl',orient='records',lines=True,force_ascii=False)
    print('Total:',len(base),'| Single constraint:',int((base.n_constraints==1).sum()),
          '| Multi-constraint:',int((base.n_constraints>1).sum()),
          '| Unique instruction IDs:',len({s for r in raw for s in r['instruction_id_list']}))
    print(base[base.n_constraints==1].type.value_counts().head(20).to_string())

    
    def official_instruction(inst,kw,response,prompt,loose=False):
        
        obj=instructions_registry.INSTRUCTION_DICT[inst](inst)
        obj.build_description(**(kw or {}))
        args=obj.get_instruction_args()
        if args and 'prompt' in args:obj.build_description(prompt=prompt)
        if not isinstance(response,str) or not response.strip():return 0
        if not loose:return int(bool(obj.check_following(response)))
        lines=response.split('\n')
        candidates=[response, '\n'.join(lines[1:]).strip(),
          '\n'.join(lines[:-1]).strip(), '\n'.join(lines[1:-1]).strip()]
        candidates += [x.replace('*','') for x in list(candidates)]
        return int(any(x.strip() and obj.check_following(x) for x in candidates))
    def strict_labels(insts,kws,response,prompt):
        flags=[official_instruction(inst,k,response,prompt) for inst,k in zip(insts,kws)]
        return flags,int(all(flags))
    assert strict_labels(['punctuation:no_comma'],[{}],'Hello world','test')[1]==1
    assert strict_labels(['punctuation:no_comma'],[{}],'Hello, world','test')[1]==0


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    REWRITE_VERSION='exact_span_paraphrases_v2_expanded_verified_spans'
    
    VARIANTS=('P1','P2','R1','R2')
    def patch_for(row):
        text=row['prompt'];typ=row['instruction_id_list'][0];kw=row['kwargs'][0]
        def one(pattern,words,flags=re.I):
            matches=list(re.finditer(pattern,text,flags))
            if len(matches)!=1:return None, 'no_unique_verified_span'
            m=matches[0]
            if not all(x and isinstance(x,str) for x in words):return None,'invalid_replacements'
            if m.start()==0 or text[:m.start()].rstrip().endswith(('.', '!', '?')):
                words=[w[0].upper()+w[1:] for w in words]
            return (m.start(),m.end(),dict(zip(VARIANTS,words))),''
        if typ=='punctuation:no_comma':
            more=one(r"\b(?:don't use any commas in your entire reply|don't use commas in your (?:entire )?(?:reply|answer|response)|your (?:entire )?(?:reply|answer|response) (?:should|must) (?:not |never )?(?:contain|include) any commas|there should be no commas in your (?:reply|answer|response)|do not include any commas(?: in the (?:reply|answer|response))?)\b",
                ['do not use any commas in your entire response',
                 'write the entire answer without any commas',
                 'your answer must include zero commas',
                 'avoid comma punctuation in the complete response'])
            if more[0] is not None:return more
        if typ=='punctuation:no_comma':
            return one(r'\b(?:you are not allowed to use any commas in your response|do not use any commas(?: in your response)?|do not use commas in your response|do not contain commas in your response|without using any commas|no commas are allowed in your response)\b',
                ['do not include commas in your answer','make your entire reply comma-free',
                 'avoid commas throughout the response','use zero commas in your reply'])
        if typ=='change_case:english_lowercase':
            more=one(r'\b(?:all in lower(?:-| )?case(?: letters)?|all letters (?:must|should) be lower(?:-| )?case|use (?:only|exclusively) lower(?:-| )?case letters|write (?:the )?(?:whole |entire )?(?:response|reply|answer) in lower(?:-| )?case)\b',
                ['entirely in lowercase letters', 'using no uppercase letters',
                 'with lowercase letters only', 'in lower-case letters throughout'])
            if more[0] is not None:return more
        if typ=='change_case:english_lowercase':
            return one(r'\b(?:in all lowercase letters|all lower case|all lowercase letters|in all lower case|all lower-case letters)\b',
                ['using lowercase letters exclusively','with lowercase letters only',
                 'without uppercase alphabetic characters','using only lower-case letters'])
        if typ=='change_case:english_capital':
            more=one(r'\b(?:your (?:entire )?(?:reply|answer|response) (?:must|should) (?:be|appear) in (?:all )?(?:upper(?:-| )?case|capital) letters|(?:write|respond|reply) (?:only |entirely )?in (?:all )?(?:upper(?:-| )?case|capital) letters|only use capital letters|capital letters only)\b',
                ['using only capital letters', 'without any lowercase letters',
                 'in uppercase letters throughout', 'with every alphabetic character capitalized'])
            if more[0] is not None:return more
        if typ=='change_case:english_capital':
            return one(r'\b(?:in all capital letters|all capital letters|in uppercase letters|all upper case letters)\b',
                ['using uppercase letters exclusively','in uppercase only',
                 'with no lowercase alphabetic characters','entirely in capital letters'])
        if typ=='detectable_format:json_format':
            return one(r'\bin JSON format\b',
                ['in valid JSON format','as valid JSON','in syntactically valid JSON','in properly formatted JSON'])
        if typ=='startend:quotation':
            more=one(r'\b(?:your entire response (?:must|should) be enclosed in double quotation marks|put the entire answer inside double quotes|place double quotes around your entire (?:answer|reply|response)|surround the answer with quotation marks)\b',
                ['enclose your entire reply in double quotation marks',
                 'put double quote marks around the entire response',
                 'surround the full answer with double quotation marks',
                 'begin and end your entire answer with double quotation marks'])
            if more[0] is not None:return more
        if typ=='startend:quotation':
            return one(r'\b(?:wrap the entire response with double quotation marks|wrap the entire response in double quotation marks|enclose the entire response in double quotation marks)\b',
                ['surround the complete response with double quotation marks',
                 'put double quotation marks at the beginning and end of your entire reply',
                 'enclose the full answer in double quotes',
                 'start and finish the whole response with double quotation marks'])
        if typ=='keywords:forbidden_words':
            words=[str(w) for w in kw.get('forbidden_words',[])]
            if not words:return None,'missing_forbidden_words'
            patterns=[r'\bDo not include the following keywords\s*:\s*[^.\n]+',
                      r'\bDo not use any of the following (?:words|keywords)\s*:\s*[^.\n]+']
            found=[]
            for pattern in patterns:found+=list(re.finditer(pattern,text,re.I))
            if len(found)!=1:return None,'forbidden_words_unmatched'
            m=found[0];span=m.group(0)
            if not all(re.search(r'(?<!\w)'+re.escape(w)+r'(?!\w)',span,re.I) for w in words):
                return None,'verifier_words_mismatch'
            quoted=', '.join(json.dumps(w,ensure_ascii=False) for w in words)
            return (m.start(),m.end(),dict(zip(VARIANTS,[
                f'Do not write any of these words: {quoted}',
                f'Exclude every one of these expressions from your answer: {quoted}',
                f'Your complete response must not contain: {quoted}',
                f'Forbidden vocabulary in the answer: {quoted}. Use none of these words']))),''
        if typ=='length_constraints:number_words':
            n=kw.get('num_words');rel=kw.get('relation')
            if not isinstance(n,int) or rel not in ('at least','less than'):return None,'unsupported_word_count_metadata'
            patterns = ([(rf'\bat least\s+{n}\s+words\b',
                          [f'no fewer than {n} words',f'{n} or more words',
                           f'a minimum of {n} words',f'not less than {n} words'])]
                        if rel=='at least' else
                        [(rf'\bless than\s+{n}\s+words\b',
                          [f'fewer than {n} words',f'under {n} words',
                           f'strictly below {n} words',f'not as many as {n} words']),
                         (rf'\bat most\s+{n-1}\s+words\b',
                          [f'fewer than {n} words',f'under {n} words',
                           f'strictly below {n} words',f'not as many as {n} words'])])
            candidates=[]
            for pattern,words in patterns:
                mat=list(re.finditer(pattern,text,re.I))
                candidates += [(m,words) for m in mat]
            if len(candidates)!=1:return None,'ambiguous_word_count_instruction'
            m,words=candidates[0]
            return (m.start(),m.end(),dict(zip(VARIANTS,words))),''
        if typ=='startend:end_checker':
            end=str(kw.get('end_phrase',''))
            if len(end)<4 or text.count(end)!=1:return None,'end_phrase_missing_or_ambiguous'
            patterns=[r'\bFinish (?:the (?:poem|response|answer|story|letter)) with this exact phrase:',
                      r'\bEnd (?:the (?:poem|response|answer|story|letter)) with this exact phrase:',
                      r'\bThe very last sentence of your response should be']
            found=[]
            for pat in patterns:found+=list(re.finditer(pat,text,re.I))
            if len(found)!=1:return None,'end_clause_unsupported'
            m=found[0]
            if not (m.end() < text.index(end) and text.index(end)-m.end()<80):return None,'end_phrase_not_near_clause'
            return (m.start(),m.end(),dict(zip(VARIANTS,[
                'Make these the precise final words of your answer:',
                'Your response must end with these exact words:',
                'The final text of your response must be exactly this phrase:',
                'Conclude your answer using these words verbatim:']))),''
        return None,'unsupported_instruction_type'

    candidates=[];selection=[]
    for r in raw:
        key=int(r['key']);n=len(r['instruction_id_list'])
        if n!=1:
            selection.append({'key':key,'type':';'.join(r['instruction_id_list']),
                              'reason':'multi_constraint_not_rewritten','selected':False})
            continue
        patch,reason=patch_for(r)
        if patch is None:
            selection.append({'key':key,'type':r['instruction_id_list'][0],
                              'reason':reason,'selected':False})
            continue
        a,b,words=patch;original_span=r['prompt'][a:b]
        pp={v:r['prompt'][:a]+words[v]+r['prompt'][b:] for v in VARIANTS}
        if len({r['prompt'],*pp.values()})!=5 or any(not p.strip() for p in pp.values()):
            selection.append({'key':key,'type':r['instruction_id_list'][0],
                              'reason':'duplicate_or_blank_rewrite','selected':False})
            continue
        
        for v,text in pp.items():
            assert text.startswith(r['prompt'][:a]) and text.endswith(r['prompt'][b:])
        candidates.append({'key':key,'type':r['instruction_id_list'][0],
            'original_prompt':r['prompt'],'source_span':original_span,'kwargs_json':cj(r['kwargs']),
            **{v:pp[v] for v in VARIANTS},**{v+'_span':words[v] for v in VARIANTS}})
        selection.append({'key':key,'type':r['instruction_id_list'][0],
                          'reason':'conservative_exact_span_match','selected':True})
    selection=pd.DataFrame(selection);selection.to_csv(DATA/'rewrite_coverage_all541.csv',index=False)
    review=pd.DataFrame(candidates);review.to_csv(DATA/'rewrite_review_all.csv',index=False)
    print('Conservatively rewritable prompts:',len(review),'of 541')
    if len(review):display(review.type.value_counts().to_frame('eligible_prompts'))
    print('Review file:',DATA/'rewrite_review_all.csv')
    
    freeze(DATA/'rewriting_fingerprint.json',{'version':REWRITE_VERSION,
         'review_sha256':shafile(DATA/'rewrite_review_all.csv'),'source_sha256':SOURCE_SHA})


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    count=review.type.value_counts() if len(review) else pd.Series(dtype=int)
    ELIGIBLE_TYPES=sorted(count[count>=MIN_PER_TYPE].index.tolist())
    assert len(ELIGIBLE_TYPES)>=2, ('Too few conservative rule-verified categories. '
        'Inspect rewrite_coverage_all541.csv and add HUMAN-REVIEWED exact-span patterns. '
        'Do not weaken the criteria or use append-only reminders as paraphrases.')
    record_by_key={int(r['key']):r for r in raw}
    review_by_key={int(r.key):r for r in review.itertuples(index=False)}
    keys=sorted(int(k) for k in review[review.type.isin(ELIGIBLE_TYPES)].key)
    
    texts=[record_by_key[k]['prompt'] for k in keys]
    vec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=1).fit_transform(texts)
    sims=cosine_similarity(vec).astype(np.float32)
    parent=list(range(len(keys)))
    def uf(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def join(a,b):
        a,b=uf(a),uf(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for i in range(len(keys)):
        for j in range(i+1,len(keys)):
            if sims[i,j]>=0.91:join(i,j)
    clusters={i:uf(i) for i in range(len(keys))}
    components=defaultdict(list)
    for i,k in enumerate(keys):components[clusters[i]].append(k)
    GROUP={k:min(v) for v in components.values() for k in v}
    assert len(GROUP)==len(keys)
    cluster_ids=sorted(set(GROUP.values()))
    rng=np.random.default_rng(SEED+811)
    pilot_set=set()
    for typ in ELIGIBLE_TYPES:
        available=sorted({GROUP[k] for k in keys if record_by_key[k]['instruction_id_list'][0]==typ})
        
        n_take=min(max(4,int(round(PILOT_FRACTION*len(available)))),max(1,len(available)-5))
        pilot_set.update(map(int,rng.choice(available,size=n_take,replace=False)))
    PILOT_GROUPS=sorted(pilot_set)
    TESTABLE_GROUPS=sorted(set(cluster_ids)-pilot_set)
    assert len(TESTABLE_GROUPS)>=FOLDS
    freeze(DATA/'eligible_and_groups.json',{'types':ELIGIBLE_TYPES,'keys':keys,
        'group_map':{str(k):GROUP[k] for k in keys},'pilot_groups':PILOT_GROUPS,
        'evaluation_groups':TESTABLE_GROUPS})
    print('Eligible types:',ELIGIBLE_TYPES)
    print('Eligible keys:',len(keys),' | near-duplicate groups:',len(cluster_ids),
          '| pilot groups:',len(PILOT_GROUPS),' | CV groups:',len(TESTABLE_GROUPS))





if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    GEN_CACHE=CACHE/'generation';STATE_CACHE=CACHE/'states'
    GEN_CACHE.mkdir(exist_ok=True);STATE_CACHE.mkdir(exist_ok=True)
    MODEL_CONTEXT={'model_id':MODEL_ID,'revision':MODEL_REV,'max_input_tokens':MAX_INPUT_TOKENS,
        'transformers_version':'4.53.3','attention':'native_sdpa',
        'chat_template':'user_then_generation_prompt','greedy':True,
        'verifier':VER_HASH}
    model=None;tokenizer=None;INPUT_DEVICE=None
    old_phase3_gen={};old_phase3_states={}
    
    if (OLD_PHASE3/'generation_cache').is_dir():
        for f in (OLD_PHASE3/'generation_cache').glob('*.json'):
            try:
                obj=json.loads(f.read_text());a=obj.get('cache_args',{})
                if a.get('budget')==1024 and a.get('model')==MODEL_ID and a.get('revision')==MODEL_REV\
                        and a.get('max_input_tokens') in (1024,MAX_INPUT_TOKENS) and a.get('verifier_hashes')==VER_HASH:
                    old_phase3_gen[(a['prompt_sha256'],a['constraint_sha256'],a['instruction_type'])]=obj
            except (ValueError,KeyError,TypeError):pass
    if (OLD_PHASE3/'state_cache').is_dir():
        for f in (OLD_PHASE3/'state_cache').glob('*.json'):
            try:
                a=json.loads(f.read_text())
                if a.get('architecture')==MODEL_ID and a.get('revision')==MODEL_REV and\
                        a.get('max_input_tokens') in (1024,MAX_INPUT_TOKENS) and tuple(a.get('layers',[]))==LAYERS:
                    old_phase3_states[a['prompt_sha256']]=f.with_suffix('.npz')
            except (ValueError,KeyError,TypeError):pass
    print('Compatible external candidates:',len(old_phase3_gen),'responses,',len(old_phase3_states),'states')

    def rec_of(key,variant):
        r=record_by_key[int(key)];typ=r['instruction_id_list'][0]
        if variant=='original':p=r['prompt']
        else:p=getattr(review_by_key[int(key)],variant)
        return {'key':int(key),'id':f'ifeval_{key}_{variant}',
            'group':int(GROUP[key]) if key in GROUP else int(key),'variant':variant,
            'prompt':p,'instruction_ids':r['instruction_id_list'],'kwargs':r['kwargs'],
            'type':typ,'n_constraints':len(r['instruction_id_list'])}

    def gen_fingerprint(r):
        return {'prompt_sha256':sha(r['prompt'].encode()),'instructions':r['instruction_ids'],
                'kwargs_hash':sha(cj(r['kwargs']).encode()),'budget':GEN_BUDGET,'model':MODEL_CONTEXT}
    def state_fingerprint(r):
        return {'prompt_sha256':sha(r['prompt'].encode()),'model_id':MODEL_ID,
                'revision':MODEL_REV,'layers':list(LAYERS),'max_input_tokens':MAX_INPUT_TOKENS,
                'template':'user_then_generation_prompt','extract':'all_hidden_states_at_prompt_final_token'}
    def load_gen(r):
        a=gen_fingerprint(r);path=GEN_CACHE/(sha(cj(a).encode())+'.json')
        if path.exists():
            item=json.loads(path.read_text());assert item['fingerprint']==a
            flags,y=strict_labels(r['instruction_ids'],r['kwargs'],item['response'],r['prompt'])
            assert item['label']==y and item['per_instruction']==flags
            return item,True
        if len(r['instruction_ids'])==1:
            old=old_phase3_gen.get((sha(r['prompt'].encode()),sha(cj(r['kwargs'][0]).encode()),r['type']))
            if old is not None and 0<int(old.get('input_tokens',0))<=1024:
                flags,y=strict_labels(r['instruction_ids'],r['kwargs'],old['response'],r['prompt'])
                assert y==int(old['label'])
                item={'response':old['response'],'label':y,'per_instruction':flags,
                      'input_tokens':int(old['input_tokens']),'output_tokens':int(old['output_tokens']),
                      'capped':bool(old['hit_token_cap']),
                      'fingerprint':a,'source':'verified_phase3_exact_prompt'}
                write_json(path,item);return item,True
        return None,False

    def load_state(r):
        a=state_fingerprint(r);path=STATE_CACHE/(sha(cj(a).encode())+'.npz');meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            assert json.loads(meta.read_text())==a
            with np.load(path,allow_pickle=False) as q: z=np.asarray(q['states'],np.float32)
            assert z.shape==(3,HIDDEN_WIDTH) and np.isfinite(z).all()
            return z,True
        old=old_phase3_states.get(sha(r['prompt'].encode()))
        if old is not None and old.is_file():
            with np.load(old,allow_pickle=False) as q:z=np.asarray(q['states'],np.float32)
            assert z.shape==(3,HIDDEN_WIDTH)
            with open(path.with_suffix('.tmp'),'wb') as f:np.savez_compressed(f,states=z.astype(np.float16))
            os.replace(path.with_suffix('.tmp'),path);write_json(meta,a)
            return z,True
        return None,False

    def encode(p):
        enc=tokenizer.apply_chat_template([{'role':'user','content':p}],
            add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors='pt')
        n=int(enc['input_ids'].shape[-1]);
        if n>MAX_INPUT_TOKENS:raise ValueError(f'Prompt has {n} tokens > allowed {MAX_INPUT_TOKENS}; raise budget in a NEW run')
        return {k:v.to(INPUT_DEVICE) for k,v in enc.items()},n

    def load_model():
        global model,tokenizer,INPUT_DEVICE
        if model is not None:return
        assert torch.cuda.is_available(),'Select a GPU runtime to generate missing responses and activations.'
        import transformers
        assert transformers.__version__=='4.53.3',f'Expected 4.53.3, found {transformers.__version__}'
        from transformers import AutoTokenizer,AutoModelForCausalLM
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False)
        model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,device_map='auto',
             torch_dtype=dtype,trust_remote_code=False,attn_implementation='sdpa')
        model.eval()
        global STOP_IDS
        STOP_IDS=prepare_tokenizer(SPEC,tokenizer,model);INPUT_DEVICE=model.get_input_embeddings().weight.device
        assert INPUT_DEVICE.type=='cuda' and len(model.model.layers)==SPEC.num_layers and model.config.hidden_size==HIDDEN_WIDTH
        assert model.__class__.__module__.startswith('transformers.models.'+SPEC.family+'.')
        
        probe,_=encode('Say hello.')
        with torch.inference_mode():
            out=model.generate(**probe,max_new_tokens=3,do_sample=False,use_cache=True,
                                **generation_stop_kwargs(SPEC,tokenizer,STOP_IDS))
            assert out.shape[1]>probe['input_ids'].shape[1]
        del probe,out
        print('Native',SPEC.short_name,'SDPA smoke test passed, model revision',MODEL_REV[:16])

    @torch.inference_mode()
    def infer(r,need_state=False):
        load_model();enc,n=encode(r['prompt']);states=None
        if need_state:
            out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
            states=np.stack([out.hidden_states[i][0,-1].float().cpu().numpy() for i in LAYERS])
            del out
        tokens=model.generate(**enc,max_new_tokens=GEN_BUDGET,do_sample=False,
                      use_cache=True,**generation_stop_kwargs(SPEC,tokenizer,STOP_IDS))
        seq=tokens[0,n:];sz=int(seq.numel());eos=end_token_ids(SPEC,tokenizer,STOP_IDS)
        cap=bool(sz>=GEN_BUDGET and (sz==0 or int(seq[-1]) not in eos))
        ans=tokenizer.decode(seq,skip_special_tokens=True).strip()
        del enc,tokens,seq
        return {'response':ans,'input_tokens':n,'output_tokens':sz,'capped':cap},states

    def get_result(r,with_state=False):
        item,reused=load_gen(r)
        zz,z_reused=(load_state(r) if with_state else (None,True))
        if item is not None and (not with_state or zz is not None):return item,zz,True
        if not RUN_GPU:raise RuntimeError(f'Cache incomplete for {r["id"]}. Set RUN_GPU=True for generation.')
        
        if item is not None:
            load_model();enc,_=encode(r['prompt'])
            with torch.inference_mode():
                out=model(**enc,output_hidden_states=True,use_cache=False,return_dict=True)
                zz=np.stack([out.hidden_states[i][0,-1].float().cpu().numpy() for i in LAYERS]);del out,enc
        else:
            data,states=infer(r,need_state=with_state and zz is None)
            flags,y=strict_labels(r['instruction_ids'],r['kwargs'],data['response'],r['prompt'])
            item={**data,'label':y,'per_instruction':flags,'fingerprint':gen_fingerprint(r),'source':SPEC.derived_source_label}
            write_json(GEN_CACHE/(sha(cj(gen_fingerprint(r)).encode())+'.json'),item)
            if states is not None:zz=states
        if with_state and zz is not None and not z_reused:
            a=state_fingerprint(r);p=STATE_CACHE/(sha(cj(a).encode())+'.npz')
            tmp=p.with_suffix('.tmp')
            with open(tmp,'wb') as f:np.savez_compressed(f,states=np.asarray(zz,np.float16))
            os.replace(tmp,p);write_json(p.with_suffix('.json'),a)
        if with_state:assert zz.shape==(3,HIDDEN_WIDTH)
        return item,zz,False


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    pilot_rows=[]
    for key in tqdm(keys,desc='Pilot original + P1'):
        if GROUP[key] not in PILOT_GROUPS:continue
        for v in ('original','P1'):
            r=rec_of(key,v)
            item,_,_=get_result(r)
            pilot_rows.append({'key':key,'group':GROUP[key],'type':r['type'],'variant':v,
                    'label':item['label'],'capped':item['capped']})
    pilot_df=pd.DataFrame(pilot_rows)
    pilot_df.to_csv(RESULTS/'pilot_screen.csv',index=False)
    print('Pilot groups:',len(PILOT_GROUPS),'pilot examples:',len(pilot_df))
    summary=pilot_df.groupby('type').agg(n=('label','size'),success=('label','sum'),cap_rate=('capped','mean'))
    summary['failure']=summary.n-summary.success
    summary['eligible']=(summary.success>=MIN_PILOT_PER_LABEL)&(summary.failure>=MIN_PILOT_PER_LABEL)&(summary.cap_rate<=MAX_TRAIN_CAP_RATE)
    summary.to_csv(RESULTS/'pilot_type_screen.csv');display(summary)
    PRIMARY_TYPES=sorted(summary.index[summary.eligible].tolist())
    freeze(OUT/'selected_types.json',PRIMARY_TYPES)
    assert len(PRIMARY_TYPES)>=2, ('Insufficient binary signal among safely rewritable types. '
        'See pilot_type_screen.csv; this is a data limitation, not a reason to include leaked or invalid cases.')
    TYPE_TO_ID={s:i for i,s in enumerate(PRIMARY_TYPES)}
    SELECTED_KEYS=sorted(k for k in keys if record_by_key[k]['instruction_id_list'][0] in PRIMARY_TYPES)
    CV_KEYS=[k for k in SELECTED_KEYS if GROUP[k] in TESTABLE_GROUPS]
    assert len(CV_KEYS)>=25 and len(set(GROUP[k] for k in CV_KEYS))>=FOLDS*2
    print('Selected for controlled H1/H2:',len(CV_KEYS),'evaluation-task rows,',len(PRIMARY_TYPES),'types.')


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    training_items=[];training_states=[]
    for key in tqdm(CV_KEYS,desc='Training-only IFEval generation and activations'):
        for v in ('original','P1','P2'):
            r=rec_of(key,v)
            item,z,hit=get_result(r,with_state=True)
            training_items.append({k:r[k] for k in ('id','key','group','variant','prompt','type','instruction_ids','kwargs')}
                             |{k:item[k] for k in ('label','per_instruction','response','capped','input_tokens','output_tokens')})
            training_states.append(z)
    train_full=pd.DataFrame(training_items)
    Ztrain=np.stack(training_states).astype(np.float32)
    assert train_full.id.is_unique and Ztrain.shape==(len(train_full),3,HIDDEN_WIDTH)
    assert train_full.groupby('key').variant.nunique().eq(3).all()
    train_full.to_json(RESULTS/'training_forms_1024_scored.jsonl',orient='records',lines=True,force_ascii=False)
    np.save(RESULTS/'training_states.npy',Ztrain.astype(np.float16))
    training_cap=train_full.groupby(['type','variant']).agg(n=('label','size'),success=('label','mean'),cap_rate=('capped','mean'))
    training_cap.to_csv(RESULTS/'training_compliance_cap_rates.csv')
    display(training_cap.round(3))
    if float(train_full.capped.mean())>.10:
        print('WARNING: Training output is still frequently capped. Report this limitation and consider a new budget-pinned run.')
    for r in tqdm(train_full.itertuples(index=False),total=len(train_full),desc='Verify training labels'):
        flags,y=strict_labels(r.instruction_ids,r.kwargs,r.response,r.prompt)
        assert y==r.label and flags==r.per_instruction,r.id


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    from scipy.special import expit
    METHODS=('B0','B2','MT0','MT2')
    N_TYPES=len(PRIMARY_TYPES)
    TR=train_full.reset_index(drop=True).copy()
    X=Ztrain[:,LAYERS.index(LAYER)].astype(np.float32)
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

    def fit_probe(train_groups,method,C,penalty=16.,layer=LAYER,max_iter=90):
        tr=TR.group.isin(train_groups) & (TR.variant.eq('original') if method in ('B0','MT0') else True)
        idx=np.flatnonzero(tr.to_numpy());ixorig=np.flatnonzero((TR.group.isin(train_groups)&TR.variant.eq('original')).to_numpy())
        assert len(idx)>0 and len(ixorig)>0
        sc=StandardScaler().fit(X[ixorig] if layer==LAYER else Ztrain[ixorig,LAYERS.index(layer)].astype(np.float32))
        XX=X[idx] if layer==LAYER else Ztrain[idx,LAYERS.index(layer)].astype(np.float32)
        z=((XX-sc.mean_)/np.maximum(sc.scale_,1e-6)).astype(np.float32)
        y=TR.iloc[idx].label.to_numpy(np.float32)
        tt=TR.iloc[idx].type.map(TYPE_TO_ID).to_numpy(np.int64)
        if len(np.unique(y))<2:
            prior=np.clip(float(y.mean()),1e-5,1-1e-5)
            return Probe(sc.mean_,np.maximum(sc.scale_,1e-6),np.zeros(HIDDEN_WIDTH),
                np.zeros((N_TYPES,HIDDEN_WIDTH)),np.zeros(N_TYPES),np.log(prior/(1-prior)))
        
        g=TR.iloc[idx].groupby('key').size().to_dict()
        wt=np.array([1/g[int(k)] for k in TR.iloc[idx].key],np.float32);wt/=wt.sum()
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        xx=torch.as_tensor(z,device=device); yy=torch.as_tensor(y,device=device)
        ids=torch.as_tensor(tt,device=device);wts=torch.as_tensor(wt,device=device)
        w=torch.nn.Parameter(torch.zeros(HIDDEN_WIDTH,device=device))
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
            delta.detach().cpu().numpy() if mt else np.zeros((N_TYPES,HIDDEN_WIDTH)),
            offsets.detach().cpu().numpy() if mt else np.zeros(N_TYPES),b.detach().cpu().item())
        del xx,yy,ids,wts,opt
        return pp

    def macro_validation(probe,groups,variants,layer=LAYER):
        mask=(TR.group.isin(groups)&TR.variant.isin(variants)).to_numpy();idx=np.flatnonzero(mask)
        if not len(idx):return np.nan
        XX=X if layer==LAYER else Ztrain[:,LAYERS.index(layer)].astype(np.float32)
        ss=[]
        for typ in PRIMARY_TYPES:
            for v in variants:
                sub=idx[(TR.iloc[idx].type.eq(typ)&TR.iloc[idx].variant.eq(v)).to_numpy()]
                if not len(sub):continue
                yy=TR.iloc[sub].label.to_numpy(int)
                if len(np.unique(yy))<2:continue
                pp=probe.predict(XX[sub],np.full(len(sub),TYPE_TO_ID[typ]))
                ss.append(safe_auc(yy,pp))
        return float(np.nanmean(ss)) if ss else np.nan

    FROZEN_FP=sha(cj({'dataset':SOURCE_SHA,'protocol':PROTOCOL,
     'selected_types':PRIMARY_TYPES,'selected_keys':CV_KEYS,
     'label_file_sha':shafile(RESULTS/'training_forms_1024_scored.jsonl'),
     'layers':LAYERS,'methods':METHODS,'software':'torch_LBFGS_v1'}).encode())
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
            feasible=[z for z in scored if base_id is None or z[1]>=base_id-MARGIN_ID]
            best=max(feasible or scored,key=lambda z:(z[0],z[1],-z[2],-z[3]))
            pp=fit_probe(trainval,method,best[2],best[3],max_iter=110)
            pp.save(path)
            info={'fold':fold,'method':method,'fp':FROZEN_FP,'C':best[2],
                  'delta_penalty':best[3],'validation_shift_auc':best[0],'id_auc':best[1],
                  'constraint_feasible':bool(feasible),'train_groups':list(map(int,trainval)),
                  'heldout_groups':list(map(int,test))}
            write_json(meta,info);fitted[(fold,method)]=pp;training_log.append(info)
            if method=='B0':base_id=best[1]
    pd.DataFrame(training_log).drop(columns=['train_groups','heldout_groups']).to_csv(RESULTS/'probe_tuning.csv',index=False)
    assert len(fitted)==FOLDS*len(METHODS)
    print('All fitted probes:',len(fitted),' | frozen before final R1/R2 scoring')





if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    
    freeze(MODELS/'fitted_models_complete.json',{
        'model_fingerprint':FROZEN_FP,'methods':list(METHODS),
        'number_of_fitted_fold_models':len(fitted),'test_templates':['R1','R2']})
    heldout_items=[];heldout_states=[]
    for key in tqdm(CV_KEYS,desc='Previously unseen R1/R2 test formulations'):
        for v in ('R1','R2'):
            r=rec_of(key,v);item,z,_=get_result(r,with_state=True)
            heldout_items.append({k:r[k] for k in ('id','key','group','variant','prompt','type','instruction_ids','kwargs')}
                              |{k:item[k] for k in ('label','per_instruction','response','capped','input_tokens','output_tokens')})
            heldout_states.append(z)
    heldout_full=pd.DataFrame(heldout_items)
    Zheldout=np.stack(heldout_states).astype(np.float32)
    control_df=pd.concat([train_full,heldout_full],ignore_index=True)
    Z=np.concatenate([Ztrain,Zheldout],axis=0)
    assert len(control_df)==len(CV_KEYS)*5 and Z.shape==(len(control_df),3,HIDDEN_WIDTH)
    assert control_df.id.is_unique
    control_df.to_json(RESULTS/'controlled_1024_scored.jsonl',orient='records',lines=True,force_ascii=False)
    np.save(RESULTS/'controlled_states.npy',Z.astype(np.float16))
    control_df.groupby(['type','variant']).agg(n=('label','size'),success=('label','mean'),cap_rate=('capped','mean')).to_csv(RESULTS/'controlled_compliance_cap_rates.csv')
    for r in tqdm(control_df.itertuples(index=False),total=len(control_df),desc='Strict verifier re-audit'):
        flags,y=strict_labels(r.instruction_ids,r.kwargs,r.response,r.prompt)
        assert flags==r.per_instruction and y==r.label,r.id

    evaluation=[]
    for i,row in control_df.iterrows():
        fold=fold_for_group[int(row.group)]
        scores={'key':int(row.key),'group':int(row.group),'fold':fold,
                'type':row.type,'variant':row.variant,'label':int(row.label),
                'capped':bool(row.capped),'output_tokens':int(row.output_tokens)}
        layer_x=Z[i,LAYERS.index(LAYER)]
        for method in METHODS:scores[method]=float(fitted[(fold,method)].predict(layer_x,[TYPE_TO_ID[row.type]])[0])
        evaluation.append(scores)
    all_pred=pd.DataFrame(evaluation)
    assert all_pred.key.nunique()==len(CV_KEYS) and len(all_pred)==len(control_df)
    all_pred.to_csv(RESULTS/'predictions_all_forms.csv',index=False)
    original_pred=all_pred[all_pred.variant.eq('original')].copy()
    R_pred=all_pred[all_pred.variant.isin(['R1','R2'])].copy()
    assert R_pred.key.nunique()==len(CV_KEYS) and original_pred.key.nunique()==len(CV_KEYS)

    def macro_auc(frame,method,within_fold=False):
        groups=['fold','type','variant'] if within_fold else ['type','variant']
        return float(np.nanmean([safe_auc(g.label,g[method])
            for _,g in frame.groupby(groups)]))

    metric_rows=[]
    for which,frame in [('original',original_pred),('unseen_R',R_pred)]:
        for method in METHODS:
            metric_rows.append({'domain':which,'method':method,'n':len(frame),
                'macro_auc':macro_auc(frame,method),'mean_fold_macro_auc':macro_auc(frame,method,True),
                'brier':float(brier_score_loss(frame.label,frame[method])),
                'logloss':float(log_loss(frame.label,frame[method],labels=[0,1]))})
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(RESULTS/'primary_metrics.csv',index=False)
    display(metrics.round(4))
    bytype=[]
    for (typ,v),g in all_pred.groupby(['type','variant']):
        for method in METHODS:
            bytype.append({'type':typ,'variant':v,'method':method,'n':len(g),
                'n_fail':int((g.label==0).sum()),'auc':safe_auc(g.label,g[method]),
                'brier':float(brier_score_loss(g.label,g[method]))})
    per_type=pd.DataFrame(bytype);per_type.to_csv(RESULTS/'per_type_and_formulation.csv',index=False)
    display(per_type[per_type.variant.isin(['R1','R2'])].pivot_table(index=['type','variant'],columns='method',values='auc').round(3))


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    H1=[]
    for typ in PRIMARY_TYPES:
        orig=original_pred[original_pred.type.eq(typ)][['key','label','capped']]
        for v in ('R1','R2'):
            nxt=R_pred[(R_pred.type.eq(typ)) & R_pred.variant.eq(v)][['key','label','capped']]
            g=orig.merge(nxt,on='key',suffixes=('_orig','_R'),validate='one_to_one')
            assert len(g)>0
            p,up,down=exact_mc(g.label_orig,g.label_R)
            H1.append({'type':typ,'variant':v,'n':len(g),
                'original_success':float(g.label_orig.mean()),'shift_success':float(g.label_R.mean()),
                'delta':float((g.label_R-g.label_orig).mean()),'fail_to_success':up,
                'success_to_fail':down,'exact_p':p,'cap_orig':float(g.capped_orig.mean()),
                'cap_R':float(g.capped_R.mean())})
    H1=pd.DataFrame(H1);H1['holm_p']=holm(H1.exact_p);H1.to_csv(RESULTS/'H1_exact_paired.csv',index=False)
    display(H1.round(4))

    def boot_delta(frame,method,baseline='B0',nboot=BOOTSTRAPS,seed=SEED):
        rng=np.random.default_rng(seed)
        groups=np.array(sorted(frame.group.unique()),dtype=int)
        group_index={int(g):i for i,g in enumerate(groups)}
        gid=np.array([group_index[int(g)] for g in frame.group],dtype=int)
        observed=macro_auc(frame,method)-macro_auc(frame,baseline)
        parts=[(g.label.to_numpy(int),g[method].to_numpy(float),g[baseline].to_numpy(float),
                np.array([group_index[int(k)] for k in g.group],int))
               for _,g in frame.groupby(['type','variant'])]
        diffs=[]
        for _ in range(nboot):
            cnt=rng.multinomial(len(groups),np.ones(len(groups))/len(groups))
            vals=[]
            for y,sa,sb,gg in parts:
                w=cnt[gg].astype(float)
                if not np.any(w[y==1]) or not np.any(w[y==0]):continue
                vals.append(float(roc_auc_score(y,sa,sample_weight=w)-roc_auc_score(y,sb,sample_weight=w)))
            if vals:diffs.append(float(np.mean(vals)))
        d=np.asarray(diffs,float)
        return {'delta':float(observed),'ci_lo':float(np.quantile(d,.025)) if len(d) else np.nan,
                'ci_hi':float(np.quantile(d,.975)) if len(d) else np.nan,
                'lower_one_sided_95':float(np.quantile(d,.05)) if len(d) else np.nan,
                'n_valid':len(d)}

    contrasts=[('B2','B0'),('MT0','B0'),('MT2','B0'),('MT2','B2'),('MT2','MT0')]
    checks=[]
    for dom,frame in [('unseen_R',R_pred),('original',original_pred)]:
        for j,(a,b) in enumerate(contrasts):
            checks.append({'domain':dom,'method':a,'baseline':b,
                **boot_delta(frame,a,b,seed=SEED+200*j+(dom=='original'))})
    H2=pd.DataFrame(checks);H2['excludes_zero']=((H2.ci_lo>0)|(H2.ci_hi<0))
    H2['id_noninferior_at_002']=np.where(H2.domain.eq('original'),H2.lower_one_sided_95>-MARGIN_ID,np.nan)
    H2.to_csv(RESULTS/'H2_bootstrap_and_ID_retention.csv',index=False)
    display(H2.round(4))
    print('Primary H2 is MT2 vs B0 on unseen R1/R2. Secondary comparisons show mechanism attribution.')


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    
    all_pred['type_prior']=np.nan
    for j,row in all_pred.iterrows():
        train=GROUPS[outer[int(row.fold)][0]]
        g=TR[(TR.group.isin(train))&(TR.type.eq(row.type))&TR.variant.eq('original')]
        all_pred.at[j,'type_prior']=float((g.label.sum()+1)/(len(g)+2))
    if RUN_TEXT_CONTROL:
        from sklearn.feature_extraction.text import TfidfVectorizer
        all_pred['char_tfidf']=np.nan
        text=control_df.prompt.tolist()
        for fold,(trix,teix) in enumerate(outer):
            trgroups=GROUPS[trix];testgroups=GROUPS[teix]
            mask=(control_df.group.isin(trgroups)&control_df.variant.eq('original')).to_numpy()
            tr_idx=np.flatnonzero(mask)
            ev_idx=np.flatnonzero(control_df.group.isin(testgroups).to_numpy())
            vect=TfidfVectorizer(analyzer='char',ngram_range=(3,5),max_features=6000,min_df=1,sublinear_tf=True)
            A=vect.fit_transform([text[z] for z in tr_idx]);B=vect.transform([text[z] for z in ev_idx])
            y=control_df.iloc[tr_idx].label.to_numpy(int)
            if len(np.unique(y))<2:pr=np.full(len(ev_idx),float(y.mean()))
            else:
                clf=LogisticRegression(C=1.,solver='liblinear',max_iter=500,random_state=SEED)
                clf.fit(A,y);pr=clf.predict_proba(B)[:,1]
            all_pred.loc[ev_idx,'char_tfidf']=pr
    assert all_pred.type_prior.notna().all()
    if RUN_TEXT_CONTROL:assert all_pred.char_tfidf.notna().all()
    all_pred.to_csv(RESULTS/'predictions_all_forms_and_controls.csv',index=False)
    R_controls=all_pred[all_pred.variant.isin(['R1','R2'])]
    control_summary=[]
    for method in list(METHODS)+['type_prior']+(['char_tfidf'] if RUN_TEXT_CONTROL else []):
        control_summary.append({'method':method,'macro_AUROC':macro_auc(R_controls,method),
            'Brier':float(brier_score_loss(R_controls.label,R_controls[method]))})
    pd.DataFrame(control_summary).to_csv(RESULTS/'OOD_controls.csv',index=False)

    flip=[]
    ref=all_pred[all_pred.variant.eq('original')].set_index('key')
    for row in R_controls.itertuples(index=False):
        prev=ref.loc[int(row.key)];delta=int(row.label)-int(prev.label)
        d={'key':int(row.key),'group':int(row.group),'type':row.type,'variant':row.variant,
           'actual_change':delta,'changed':int(delta!=0)}
        for m in METHODS:
            pp=float(getattr(row,m)-prev[m]);d[m+'_score_change']=pp
            d[m+'_right_flip']=(int(pp*delta>0) if delta else np.nan)
            d[m+'_stable_abs']=(abs(pp) if not delta else np.nan)
        flip.append(d)
    flip=pd.DataFrame(flip);flip.to_csv(RESULTS/'behavioral_flips.csv',index=False)
    flip_summary=[]
    for m in METHODS:
        flip_summary.append({'method':m,'n_flip':int(flip.changed.sum()),
           'flip_direction_accuracy':float(flip[m+'_right_flip'].mean()),
           'stable_mean_abs_score_change':float(flip[m+'_stable_abs'].mean())})
    pd.DataFrame(flip_summary).to_csv(RESULTS/'behavioral_flips_summary.csv',index=False)

    if RUN_LAYER_SENSITIVITY:
        other=[]
        for layer in LAYERS:
            if layer==LAYER:continue
            for fold,(trix,teix) in enumerate(outer):
                trgroups=GROUPS[trix];testgroups=GROUPS[teix]
                tr=next(z for z in training_log if z['fold']==fold and z['method']=='B0')
                for m in METHODS:
                    cfg=next(z for z in training_log if z['fold']==fold and z['method']==m)
                    pp=fit_probe(trgroups,m,cfg['C'],cfg['delta_penalty'],layer=layer,max_iter=100)
                    ix=np.flatnonzero(control_df.group.isin(testgroups).to_numpy())
                    for i in ix:
                        rr=control_df.iloc[i]
                        if rr.variant not in ('R1','R2'):continue
                        score=float(pp.predict(Z[i,LAYERS.index(layer)],[TYPE_TO_ID[rr.type]])[0])
                        other.append({'layer':layer,'fold':fold,'group':rr.group,
                           'key':rr.key,'type':rr.type,'variant':rr.variant,'label':rr.label,
                           'method':m,'score':score})
        layer_df=pd.DataFrame(other);layer_df.to_csv(RESULTS/'exploratory_other_layers.csv',index=False)
        print('Alternative layers evaluated (exploratory):',len(layer_df))







if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    
    def finite_or_null(z):
        try:
            x=float(z)
            return x if np.isfinite(x) else None
        except (ValueError,TypeError):return None
    main_gain=H2[(H2.domain.eq('unseen_R'))&(H2.method.eq('MT2'))&(H2.baseline.eq('B0'))].iloc[0]
    augmentation_gain=H2[(H2.domain.eq('unseen_R'))&(H2.method.eq('B2'))&(H2.baseline.eq('B0'))].iloc[0]
    arch_gain=H2[(H2.domain.eq('unseen_R'))&(H2.method.eq('MT2'))&(H2.baseline.eq('B2'))].iloc[0]
    retention=H2[(H2.domain.eq('original'))&(H2.method.eq('MT2'))&(H2.baseline.eq('B0'))].iloc[0]
    report={'model_id':MODEL_ID,'model_revision':MODEL_REV,'dataset_sha256':SOURCE_SHA,
      'eligible_paraphrase_n':len(CV_KEYS),'selected_types':PRIMARY_TYPES,
      'pilot_groups':len(PILOT_GROUPS),'cv_groups':len(GROUPS),'n_controlled_responses':len(control_df),
      'primary_MT2_minus_B0':finite_or_null(main_gain.delta),'primary_ci':[finite_or_null(main_gain.ci_lo),finite_or_null(main_gain.ci_hi)],
      'augmentation_B2_minus_B0':finite_or_null(augmentation_gain.delta),
      'architecture_MT2_minus_B2':finite_or_null(arch_gain.delta),
      'architecture_ci':[finite_or_null(arch_gain.ci_lo),finite_or_null(arch_gain.ci_hi)],
      'ID_MT2_minus_B0':finite_or_null(retention.delta),
      'ID_noninferiority_one_sided':bool(np.isfinite(retention.lower_one_sided_95) and retention.lower_one_sided_95>-MARGIN_ID),
      'primary_OOD_ci_excludes_zero':bool(np.isfinite(main_gain.ci_lo) and main_gain.ci_lo>0),
      'h1_holm_significant_cells':int((H1.holm_p<.05).sum()),
      'notes':['Unseen templates R1/R2 are withheld from tuning/training.',
               'Results are conditional on selected safely rewritable single-constraint examples.',
               'Manual semantic review is necessary before asserting exact paraphrase equivalence.',
               'Near-duplicate groups are kept together, with pilot groups excluded from CV.',
               'No claim of method success is predetermined.']}
    write_json(RESULTS/'RESULT_SUMMARY.json',report)
    print('FINAL SUMMARY:');print(json.dumps(report,indent=2))
    
    archive=str(ULM_WORKDIR/'IFEval_derived_result_archive.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for rootdir in [DATA,RESULTS,MODELS]:
            for path in rootdir.rglob('*'):
                if not path.is_file():continue
                if path.suffix in ('.npy','.npz') or path.name.endswith('responses.jsonl') or path.name=='controlled_1024_scored.jsonl':continue
                z.write(path,arcname=str(path.relative_to(OUT)))
    print('Portable report archive:',archive,'(',round(Path(archive).stat().st_size/1048576,2),'MB )')
    print('Drive results:',OUT)


if not REPLAY_COMPLETE and RUN_IFEVAL_DERIVED:
    
    import gc, torch
    if 'model' in globals() and model is not None:
        del model
        model = None
    if 'tokenizer' in globals() and tokenizer is not None:
        del tokenizer
        tokenizer = None
    gc.collect(); torch.cuda.empty_cache()
    IFEVAL_RESULTS = str(RESULTS)
    print('IFEval-derived results:', IFEVAL_RESULTS)
    DERIVED_CONTEXT={"fitted":fitted.copy(),"fold_for_group":fold_for_group.copy(),"types":tuple(PRIMARY_TYPES),"probe_dir":str(MODELS)}
