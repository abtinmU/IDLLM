"""Verify an already completed run against stored file metadata and hashes (fast replay).

Pure stdlib: runs before dependency installation.

Generated from notebooks/*_run.ipynb; runs in the shared namespace set up by
ulm/pipeline.py, so names defined by earlier stages are available here.
"""



# Check saved results before loading heavy dependencies.
import os,sys,hashlib,json
from pathlib import Path
_MODEL_LABEL=SPEC.model_id
_CODE_RELEASE_SHA=SPEC.code_release_sha
_REPO_ROOT=Path('..') if Path('../data/cct_benchmark.zip').exists() else Path('.')
_WORKDIR_DEFAULT=_REPO_ROOT/'results'/('ulm_'+SPEC.tag+'_server_results')
_ULM_OUT=Path(os.environ.get('ULM_WORKDIR',str(_WORKDIR_DEFAULT))).expanduser().resolve()
_ULM_DATA=Path(os.environ.get('ULM_CCT_ZIP',str(_REPO_ROOT/'data/cct_benchmark.zip'))).expanduser().resolve()
_RUN_MARKER=_ULM_OUT/'COMPLETED_RUN.json'
def _fingerprint_file(path):
    h=hashlib.sha256()
    with path.open('rb') as reader:
        for part in iter(lambda:reader.read(1024*1024),b''):h.update(part)
    return h.hexdigest()
def _replay_valid():
    if not _RUN_MARKER.is_file() or not _ULM_DATA.is_file():return False,'No complete marker or CCT input'
    try:
        m=json.loads(_RUN_MARKER.read_text())
        if m['version']!='verified-v2' or m['model']!=_MODEL_LABEL or m['notebook_sha256']!=_CODE_RELEASE_SHA:return False,'Protocol changed'
        if m['cct_sha256']!=_fingerprint_file(_ULM_DATA):return False,'CCT changed'
        entries=m['files']
        if not entries:return False,'Missing output inventory'
        for relative,rec in entries.items():
            p=_ULM_OUT/relative
            if not p.is_file() or p.stat().st_size!=rec['size']:return False,'Missing or truncated '+relative
            if rec.get('sha256') and _fingerprint_file(p)!=rec['sha256']:return False,'Changed '+relative
        return True,'Validated '+str(len(entries))+' saved files'
    except Exception as e:return False,'Corrupt/incompatible marker: '+str(e)
REPLAY_COMPLETE,_REPLAY_REASON=_replay_valid()
print('MODEL:',_MODEL_LABEL,'| completed-run replay:',REPLAY_COMPLETE,'|',_REPLAY_REASON)
if REPLAY_COMPLETE:
    print('All expected study artifacts found. Skipping package installation, downloads, inference, training, and analysis.')
    print('Results:',_ULM_OUT/'final_tables')
    print('CCT:',_ULM_OUT/'CCT_independent_refit')
