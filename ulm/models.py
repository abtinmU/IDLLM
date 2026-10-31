"""Per-model protocol constants and tokenizer/stop-token policies.

Everything that differed between the Phi-3, Qwen2.5 and Llama 3.2 notebooks lives
here; the stage scripts are shared verbatim. This module is stdlib-only because the
replay check runs before dependency installation.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    key: str                  
    tag: str                  
    camel: str                
    model_id: str
    display_name: str
    short_name: str
    family: str               
    layers: Tuple[int, int, int]
    primary_layer: int
    hidden_width: int
    num_layers: int
    
    
    code_release_sha: str
    
    pinned_revision: Optional[str] = None
    gated: bool = False       
    stop_policy: str = 'default'          
    expected_eos_token: Optional[str] = None
    expected_model_type: Optional[str] = None
    check_chat_template: bool = False
    smoke_test: bool = False  
    
    
    derived_source_label: str = 'phi3_1024'


MODEL_SPECS = {
    'phi3': ModelSpec(
        key='phi', tag='phi3', camel='Phi3',
        model_id='microsoft/Phi-3-mini-128k-instruct',
        display_name='Phi-3-mini-128k-instruct', short_name='Phi-3', family='phi3',
        layers=(14, 26, 32), primary_layer=14, hidden_width=3072, num_layers=32,
        code_release_sha='a869aacaba8a6ab62463910d8abcb96fd83c8b99edfdc89817591a5b40d34f1c',
        pinned_revision='f3c06aed622e14ca0abf5115094e4fc9a9948f36',
    ),
    'qwen25': ModelSpec(
        key='qwen25', tag='qwen25', camel='Qwen25',
        model_id='Qwen/Qwen2.5-3B-Instruct',
        display_name='Qwen2.5-3B-Instruct', short_name='Qwen', family='qwen2',
        layers=(16, 29, 36), primary_layer=16, hidden_width=2048, num_layers=36,
        code_release_sha='c29b75bb26cb94bb13c690578e0657b9224773f215f9c451f4cd42b61b145289',
        expected_eos_token='<|im_end|>', expected_model_type='qwen2', check_chat_template=True,
    ),
    'llama32': ModelSpec(
        key='llama32', tag='llama32', camel='Llama32',
        model_id='meta-llama/Llama-3.2-3B-Instruct',
        display_name='Llama-3.2-3B-Instruct', short_name='Llama', family='llama',
        layers=(12, 23, 28), primary_layer=12, hidden_width=3072, num_layers=28,
        code_release_sha='25334ad52390d252958138389990adc92db9a00c2cd05761236be4ebbbd5b82d',
        gated=True, stop_policy='llama', expected_model_type='llama', check_chat_template=True,
        smoke_test=True, derived_source_label='llama32_1024',
    ),
}

ALIASES = {'phi': 'phi3', 'phi3': 'phi3', 'qwen': 'qwen25', 'qwen25': 'qwen25',
           'llama': 'llama32', 'llama32': 'llama32'}


def get_spec(name):
    try:
        return MODEL_SPECS[ALIASES[name.lower()]]
    except KeyError:
        raise ValueError(f'Unknown model {name!r}; choose from {sorted(ALIASES)}') from None


def llama_stop_ids(tokenizer, model):
    """Llama chat turns end with <|eot_id|>; also accept the generation config's EOS list."""
    base = getattr(model.generation_config, 'eos_token_id', None)
    ids = set(int(x) for x in base) if isinstance(base, (list, tuple)) else ({int(base)} if base is not None else set())
    for marker in ('<|eot_id|>', '<|end_of_text|>', '<|eom_id|>'):
        idx = tokenizer.convert_tokens_to_ids(marker)
        if isinstance(idx, int) and 0 <= idx < len(tokenizer):
            ids.add(idx)
    assert tokenizer.convert_tokens_to_ids('<|eot_id|>') in ids, 'Missing Llama assistant turn terminator'
    return sorted(ids)


def prepare_tokenizer(spec, tokenizer, model):
    """Apply the model's post-load tokenizer checks; return explicit stop ids or None."""
    stop_ids = None
    if spec.stop_policy == 'llama':
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        stop_ids = llama_stop_ids(tokenizer, model)
    if spec.expected_eos_token is not None:
        assert tokenizer.eos_token == spec.expected_eos_token, (spec.short_name + ' assistant EOS changed', tokenizer.eos_token)
        assert tokenizer.pad_token_id is not None
    if spec.check_chat_template:
        assert tokenizer.chat_template is not None, spec.short_name + ' chat template unavailable'
    if spec.expected_model_type is not None:
        assert model.config.model_type == spec.expected_model_type, model.config.model_type
    return stop_ids


def generation_stop_kwargs(spec, tokenizer, stop_ids):
    """pad/eos arguments passed to model.generate()."""
    if spec.stop_policy == 'llama':
        return {'pad_token_id': tokenizer.pad_token_id, 'eos_token_id': stop_ids}
    return {'pad_token_id': tokenizer.eos_token_id}


def end_token_ids(spec, tokenizer, stop_ids):
    """Token ids that count as a natural end when deciding whether output hit the cap."""
    if spec.stop_policy == 'llama':
        return set(stop_ids)
    ends = {int(tokenizer.eos_token_id)}
    ends.update(map(int, getattr(tokenizer, 'additional_special_tokens_ids', []) or []))
    return ends
