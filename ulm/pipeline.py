"""Run the stage scripts in one shared namespace, like Run All in the notebooks."""
import sys
import time
import types
from pathlib import Path

STAGE_DIR = Path(__file__).resolve().parent / 'stages'


# Stages run in this order.
STAGES = (
    ('replay', 'replay.py', 'check for a completed run (fast replay)'),
    ('setup', 'setup.py', 'dependencies, workspace, CCT input, model settings, revision pin'),
    ('ifeval-simple', 'ifeval_simple.py', 'Part I: IFEval-Simple'),
    ('ifeval-derived', 'ifeval_derived.py', 'Part II: IFEval-derived (only with --run-ifeval-derived)'),
    ('followbench', 'followbench.py', 'Part III: FollowBench-derived'),
    ('export', 'final_export.py', 'Part IV: unified final-paper export'),
    ('cct', 'cct.py', 'Part V: Counterfactual Compliance Transfer'),
    ('complete', 'complete.py', 'write the verified completion marker'),
)
STAGE_NAMES = tuple(name for name, _, _ in STAGES)


def run(spec, until=None, run_ifeval_derived=False):
    """Execute stages in order and return the session module holding their globals.

    Stages share one namespace because they were notebook cells; it is registered in
    sys.modules so classes defined in a stage stay importable by name.
    """
    session = types.ModuleType('ulm_session')
    session.__dict__.update(SPEC=spec, RUN_IFEVAL_DERIVED=run_ifeval_derived)
    sys.modules['ulm_session'] = session
    for name, filename, _ in STAGES:
        path = STAGE_DIR / filename
        print(f'\n===== [{spec.tag}] stage {name} ({filename}) =====', flush=True)
        start = time.time()
        exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), session.__dict__)
        print(f'===== [{spec.tag}] stage {name} done in {time.time() - start:.1f}s =====', flush=True)
        if name == until:
            break
    return session
