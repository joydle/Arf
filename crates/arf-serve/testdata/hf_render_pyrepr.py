# Reference render for arf-serve's jinja_chat `qwen3_coder_python_str_matches_hf` test: Jinja2 set up
# exactly as HF transformers' `_compile_jinja_template` does, with tools and history
# that exercise Python str(): a type list, anyOf, float/bool/None defaults, a list-of-dicts argument.
import json, sys
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
env.filters["tojson"] = tojson
env.globals["raise_exception"] = raise_exception

tools = json.loads(open(sys.argv[2]).read())
history = json.loads(open(sys.argv[3]).read())
# vLLM passes arguments to the template as dicts (json.loads of the OpenAI string).
for m in history:
    for c in m.get("tool_calls", []):
        c["function"]["arguments"] = json.loads(c["function"]["arguments"])
t = env.from_string(open(sys.argv[1]).read())
out = t.render(messages=history, tools=tools, add_generation_prompt=True, bos_token="", eos_token="<|im_end|>")
open(sys.argv[4], "w").write(out)
print(len(out), "bytes ->", sys.argv[4])
# Regenerate (Jinja2 3.1.6):
#   python hf_render_pyrepr.py qwen3_coder_chat_template.jinja pyrepr_tools.json \
#       pyrepr_history.json qwen3_coder_python_str.hf.txt
