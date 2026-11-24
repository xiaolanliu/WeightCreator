"""Image-only noun matching for an instruction-specified planar push."""
import argparse
import json
import re
from pathlib import Path
from semantic_grounding import load,match_score
from semantic_phrasing import visual_phrase

p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();folder=a.capture
instruction=json.loads((folder/'instruction.json').read_text())['instruction'].lower()
match=re.fullmatch(r'push (?:the )?(.+?) to the front of (?:the )?(.+)',instruction)
if not match:raise ValueError('Unsupported push instruction: '+instruction)
source,target=match.groups();candidates=json.loads((folder/'candidates.json').read_text())['candidates']
model,processor=load('0.8B');model.requires_grad_(False)
scores={noun:[match_score(model,processor,folder/f"crop_{c['id']}.png",visual_phrase(noun)) for c in candidates] for noun in [source,target]}
selected=max(range(len(candidates)),key=lambda j:scores[source][j]);destination=max((j for j in range(len(candidates)) if j!=selected),key=lambda j:scores[target][j])
plan=dict(skill='push',source_id=candidates[selected]['id'],target_id=candidates[destination]['id'],semantic_scores=scores,
          provenance='Frozen Qwen3.5-0.8B crop margins for instruction nouns; camera-relative front geometry')
(folder/'condition-push.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
