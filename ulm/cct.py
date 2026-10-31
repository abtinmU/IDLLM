"""Render the CCT helper modules that the CCT stage writes into ``cct_modules/``."""
from pathlib import Path

TEMPLATE_DIR = Path(__file__).resolve().parent / 'cct_templates'

_DEFAULT = {
    'POST_LOAD': "  self.model.eval();assert len(self.model.model.layers)=={num_layers} and self.model.config.hidden_size=={hidden}",
    'STOP_IDS_METHOD': "",
    'GEN_STOP_KW': "pad_token_id=self.tok.eos_token_id",
    'END_IDS': ("  end_ids=set([int(self.tok.eos_token_id)])\n"
                "  end_ids.update(int(v) for v in getattr(self.tok,'additional_special_tokens_ids',[]) or [])"),
}
_LLAMA = {
    'POST_LOAD': ("  self.model.eval()\n"
                  "  if self.tok.pad_token_id is None:self.tok.pad_token=self.tok.eos_token\n"
                  "  self.stop_ids=self._stop_ids()\n"
                  "  assert self.tok.chat_template is not None\n"
                  "  assert self.model.config.model_type=='llama'\n"
                  "  assert len(self.model.model.layers)=={num_layers} and self.model.config.hidden_size=={hidden}"),
    'STOP_IDS_METHOD': (" def _stop_ids(self):\n"
                        "  base=self.model.generation_config.eos_token_id\n"
                        "  ids=set(map(int,base)) if isinstance(base,(list,tuple)) else ({int(base)} if base is not None else set())\n"
                        "  for marker in ('<|eot_id|>','<|end_of_text|>','<|eom_id|>'):\n"
                        "   idx=self.tok.convert_tokens_to_ids(marker)\n"
                        "   if isinstance(idx,int) and 0<=idx<len(self.tok):ids.add(idx)\n"
                        "  assert self.tok.convert_tokens_to_ids('<|eot_id|>') in ids\n"
                        "  return sorted(ids)\n"),
    'GEN_STOP_KW': "pad_token_id=self.tok.pad_token_id,eos_token_id=self.stop_ids",
    'END_IDS': "  end_ids=set(self.stop_ids)",
}


def _fill(text, values):
    for name, value in values.items():
        marker = '@@' + name + '@@'
        assert marker in text, marker
        text = text.replace(marker, value)
    assert '@@' not in text, 'Unfilled CCT template marker'
    return text


def render_common(spec, revision):
    text = (TEMPLATE_DIR / 'cct_common.py.tmpl').read_text()
    return _fill(text, {'SHORT_NAME': spec.short_name, 'MODEL_REVISION': revision})


def render_inference(spec):
    text = (TEMPLATE_DIR / 'cct_inference.py.tmpl').read_text()
    policy = _LLAMA if spec.stop_policy == 'llama' else _DEFAULT
    fmt = {'num_layers': spec.num_layers, 'hidden': spec.hidden_width}
    values = {name: value.format(**fmt) if name == 'POST_LOAD' else value for name, value in policy.items()}
    values.update({
        'DISPLAY_NAME': spec.display_name,
        'MODEL_ID': spec.model_id,
        'LAYERS_LIST': str(list(spec.layers)).replace(' ', ''),
        'LAYERS_TUPLE': str(tuple(spec.layers)).replace(' ', ''),
        'HIDDEN_WIDTH': str(spec.hidden_width),
    })
    return _fill(text, values)
