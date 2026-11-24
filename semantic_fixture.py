"""Image-based fixture choice and instruction ordinal; no simulator identities."""
import argparse
import json
import re
from pathlib import Path
from semantic_grounding import load,match_score

p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();folder=a.capture
instruction=json.loads((folder/'instruction.json').read_text())['instruction'].lower()
match=re.fullmatch(r'open the (top|middle|bottom) drawer(?: of the (?:wooden )?cabinet)?',instruction)
if not match:raise ValueError('Unsupported fixture instruction: '+instruction)
level=match.group(1);rows=json.loads((folder/'fixture_proposals.json').read_text())
eligible=[r for r in rows if r['count']>= (3 if level=='middle' else 1)]
if not eligible:raise ValueError('No visible handle geometry supports the requested drawer ordinal')
model,processor=load('0.8B');model.requires_grad_(False)
scores={i:match_score(model,processor,folder/f'crop_{i}.png','a cabinet with drawers') for i in {r['cabinet_id'] for r in eligible}}
chosen=max(eligible,key=lambda r:scores[r['cabinet_id']]);assert scores[chosen['cabinet_id']]>0,'Fixture semantics uncertain'
index={'bottom':0,'middle':len(chosen['handles'])//2,'top':len(chosen['handles'])-1}[level]
plan=dict(skill='drawer_open',level=level,cabinet_id=chosen['cabinet_id'],handle=chosen['handles'][index],normal=chosen['normal'],
          fixture_extent=chosen['fixture_extent'],front_plane=chosen['plane'],handle_heights=[h['position'][2] for h in chosen['handles']],semantic_scores=scores,provenance='Frozen Qwen3.5-0.8B cabinet crop margins + visible protruding bars + instruction ordinal')
(folder/'condition-fixture.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
