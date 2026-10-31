"""Dependencies, workspace, CCT input checks, model settings and revision pinning.

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""






if not REPLAY_COMPLETE:
    # Install missing packages for a new run.
    import importlib.metadata as _im, sys as _sys, subprocess as _sp
    _needed={'transformers':'4.53.3','accelerate':None,'huggingface-hub':None,
     'sentencepiece':None,'scikit-learn':None,'pandas':None,'scipy':None,'tqdm':None,
     'nltk':None,'immutabledict':None,'langdetect':None,'absl-py':None,'requests':None,
     'matplotlib':None,'emoji':None}
    _missing=[]
    for _pkg,_exact in _needed.items():
        try:
            _got=_im.version(_pkg)
            if _exact is not None and _got!=_exact:_missing.append((_pkg,_got,_exact))
        except _im.PackageNotFoundError:_missing.append((_pkg,'missing',_exact))
    if _missing:
        print('Installing missing/incompatible dependencies:',_missing)
        import shlex as _shlex
        _sp.check_call([_sys.executable,'-m','pip']+_shlex.split("-q install 'transformers==4.53.3' 'accelerate>=1.2,<2' 'huggingface-hub>=0.30,<1' 'sentencepiece>=0.2' 'scikit-learn>=1.4,<1.8' 'pandas>=2,<3' 'scipy>=1.12,<2' 'tqdm>=4.66' 'nltk>=3.8,<4' 'immutabledict>=4' 'langdetect>=1.0.9' 'absl-py>=2' 'requests>=2.31' 'matplotlib>=3.7' 'emoji>=2.10,<3'"))
    else:
        print('Python dependencies already satisfied; no installation or network required.')


if not REPLAY_COMPLETE:
    from pathlib import Path
    import os, zipfile, json, hashlib
    
    _REPOSITORY_ROOT = (Path('..') if Path('../data/cct_benchmark.zip').exists()
         else Path('.') if Path('data/cct_benchmark.zip').exists() else Path('..'))
    _DEFAULT_WORKDIR = _REPOSITORY_ROOT / 'results' / ('ulm_'+SPEC.tag+'_server_results')
    ULM_WORKDIR = Path(os.environ.get('ULM_WORKDIR',str(_DEFAULT_WORKDIR))).expanduser().resolve()
    ULM_WORKDIR.mkdir(parents=True,exist_ok=True)
    
    _NOTEBOOK_CCT_CANDIDATES=[Path('data/cct_benchmark.zip'),
     Path('../data/cct_benchmark.zip'),Path('./cct_benchmark.zip')]
    _CCT_AUTO=next((x for x in _NOTEBOOK_CCT_CANDIDATES if x.is_file()),_NOTEBOOK_CCT_CANDIDATES[0])
    CCT_ZIP = Path(os.environ.get('ULM_CCT_ZIP',str(_CCT_AUTO))).expanduser().resolve()
    assert CCT_ZIP.is_file(), ("Pass run.py --cct-zip, place the ZIP at data/cct_benchmark.zip, or set ULM_CCT_ZIP="
                                "/absolute/path/to/cct_benchmark.zip. No other inputs are required.")
    with zipfile.ZipFile(CCT_ZIP) as z:
        assert z.testzip() is None, 'Corrupted CCT benchmark ZIP'
        expected={'benchmark_protocol.json','SHA256SUMS.json','unified/CCT_counterfactual_prompts.jsonl'}
        assert expected.issubset(z.namelist()), ("Wrong input: expected the CCT benchmark data ZIP, not the"
                                               " CCT final-results ZIP",sorted(expected-set(z.namelist())))
        declared=json.loads(z.read('SHA256SUMS.json'))
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in declared.items()), 'CCT ZIP member hash mismatch'
    _PINNED_CHECKER_CANDIDATES=[Path('data/official_checker/instruction_following_eval'),
        Path('../data/official_checker/instruction_following_eval')]
    _LOCAL_CHECKER=next((d for d in _PINNED_CHECKER_CANDIDATES if (d/'instructions.py').is_file()), None)
    if _LOCAL_CHECKER is not None and not os.environ.get('ULM_CHECKER_SOURCE_DIR'):
        os.environ['ULM_CHECKER_SOURCE_DIR']=str(_LOCAL_CHECKER.resolve())
    print('Server output folder:',ULM_WORKDIR)
    print('CCT input:',CCT_ZIP,'| SHA256:',hashlib.sha256(CCT_ZIP.read_bytes()).hexdigest())
    print('CCT manifest member hash checks: PASS')


if not REPLAY_COMPLETE:
    
    MODEL = SPEC.key
    RUN_TAG = 'final_'+SPEC.tag+'_simple_v1'  
    RUN_TEXT_CONTROL = True
    RUN_OTHER_LAYERS = False    
    N_BOOT = 2000
    SEED = 20261017

    
    MODEL_CONFIGS = {
        SPEC.key: {'model_id':SPEC.model_id,
                'layers':SPEC.layers,'expected_hidden':SPEC.hidden_width,'expected_layers':SPEC.num_layers,
                'family':SPEC.family},
    }
    assert MODEL in MODEL_CONFIGS, f'Unknown MODEL={MODEL!r}'
    CFG = MODEL_CONFIGS[MODEL]
    MODEL_ID = CFG['model_id']
    LAYERS = CFG['layers']
    PRIMARY_LAYER = SPEC.primary_layer
    GENERATION_BUDGET = 1024             
    MAX_INPUT_TOKENS = 1024
    FOLDS = 5
    Cs = (0.03,0.1,0.3)
    DELTA_PENALTIES = (4.0,16.0)
    ORIG_MARGIN = 0.02
    METHODS = ('B0','B2','MT0','MT2')
    TYPES = ['keywords:forbidden_words','keywords:existence','keywords:frequency',
             'detectable_content:number_placeholders','startend:end_checker']
    
    PRIMARY_TYPES = ('keywords:forbidden_words','detectable_content:number_placeholders','startend:end_checker')
    R_TEMPLATES = ('R1','R2')
    HIDDEN_WIDTH=SPEC.hidden_width
    SPEC_LAYERS=SPEC.layers
    SPEC_PRIMARY_LAYER=SPEC.primary_layer
    assert LAYERS==SPEC_LAYERS and PRIMARY_LAYER==SPEC_PRIMARY_LAYER
    print('Chosen model:', MODEL, '| Hugging Face repo:', MODEL_ID)


if not REPLAY_COMPLETE:
    
    import os, sys, gc, math, json, time, itertools, hashlib, warnings, zipfile, shutil, random, importlib
    from pathlib import Path
    from collections import Counter
    from datetime import datetime, timezone
    import numpy as np, pandas as pd, torch, requests, nltk, transformers
    from tqdm.auto import tqdm
    from ulm.compat import display
    from sklearn.model_selection import KFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    from scipy.special import expit
    from scipy.stats import binomtest
    from huggingface_hub import HfApi
    from ulm.models import prepare_tokenizer, generation_stop_kwargs, end_token_ids

    assert transformers.__version__ == '4.53.3', ("Wrong loaded Transformers version: ",transformers.__version__," Restart runtime and rerun first 2 cells.")
    assert torch.cuda.is_available(), 'CUDA GPU required for the frozen BF16 inference protocol.'
    assert torch.cuda.is_bf16_supported(), 'BF16 GPU required to reproduce our standard protocol; choose A100 or better.'
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    DTYPE=torch.bfloat16
    GPU_NAME=torch.cuda.get_device_name(0)
    print('GPU',GPU_NAME,'| torch',torch.__version__,'| transformers',transformers.__version__)

    
    SIMPLE_ROOT=ULM_WORKDIR/'IFEval-Simple'
    PHI_LEGACY_ROOT=SIMPLE_ROOT  
    DRIVE_BASE=SIMPLE_ROOT/'source_replication'  
    MODEL_ROOT=DRIVE_BASE/MODEL
    DATA_DIR=SIMPLE_ROOT/'data'
    for p in (MODEL_ROOT,DATA_DIR):p.mkdir(parents=True,exist_ok=True)
    REV_RECORD=MODEL_ROOT/'selected_revision.json'
    requested_revision=os.environ.get('ULM_MODEL_REV', '').strip() or (SPEC.pinned_revision or '')
    if REV_RECORD.exists():
        saved=json.loads(REV_RECORD.read_text())
        assert saved['model_id']==MODEL_ID, 'Existing revision pin is for a different checkpoint'
        MODEL_REV=saved['revision']
        assert not requested_revision or requested_revision==MODEL_REV, 'Requested revision conflicts with frozen run'
    else:
        MODEL_REV=requested_revision or HfApi(token=os.getenv('HF_TOKEN')).model_info(MODEL_ID).sha
        assert len(MODEL_REV)==40 and all(c in '0123456789abcdef' for c in MODEL_REV), 'Expected a full HF commit hash'
        REV_RECORD.write_text(json.dumps({'model_id':MODEL_ID,'revision':MODEL_REV},indent=2))
    assert not SPEC.pinned_revision or MODEL_REV==SPEC.pinned_revision, 'Model revision changed in this workspace'
    print('Pinned',SPEC.short_name,'checkpoint revision:',MODEL_REV)


    def sha(buf):return hashlib.sha256(buf).hexdigest()
    def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)
    def shafile(p):
        h=hashlib.sha256()
        with open(p,'rb') as f:
            for buf in iter(lambda:f.read(1024*1024),b''):h.update(buf)
        return h.hexdigest()
    def atomic_json(p,obj):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.tmp')
        tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')
        os.replace(tmp,p)
    def atomic_npz(p,**arrays):
        p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_name(p.name+'.tmp')
        with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
        os.replace(tmp,p)
    def freeze_json(p,obj):
        p=Path(p)
        if p.exists(): assert json.loads(p.read_text(encoding='utf-8'))==json.loads(canonical(obj)), f'Frozen protocol mismatch at {p}. Change RUN_TAG to start a new run.'
        else:atomic_json(p,obj)
    def safe_auc(y,p):
        y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
        return float(roc_auc_score(y,p)) if len(np.unique(y))==2 else float('nan')
    auc=safe_auc  
    for pkg in ('punkt','punkt_tab'):assert nltk.download(pkg,quiet=True),pkg


if not REPLAY_COMPLETE:
    
    from ulm.models import MODEL_SPECS
    assert MODEL_ID == SPEC.model_id
    assert HIDDEN_WIDTH == SPEC.hidden_width and SPEC_LAYERS == SPEC.layers
    assert MODEL_REV == json.loads(REV_RECORD.read_text())['revision']
    _OTHER_TAGS=[s.tag for s in MODEL_SPECS.values() if s.tag!=SPEC.tag]
    assert not any(t+suffix in str(ULM_WORKDIR).lower() for t in _OTHER_TAGS for suffix in ('_server_results','_final')) if os.environ.get('ULM_WORKDIR') is None else True
    print(SPEC.display_name,'configured; revision',MODEL_REV, '| width',HIDDEN_WIDTH, '| layers',SPEC_LAYERS)
