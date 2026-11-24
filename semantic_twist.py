"""Ground a full rotary assembly, then select its visible raised grip bar."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from semantic_grounding import load,match_score

p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--weights',type=Path,required=True);a=p.parse_args();folder=a.capture
instruction=json.loads((folder/'instruction.json').read_text())['instruction'].lower()
if instruction!='turn on the stove':raise ValueError('Unsupported twist instruction: '+instruction)
candidates=json.loads((folder/'candidates.json').read_text())['candidates']
limits=torch.load(a.weights,weights_only=True)['handle_geometry'].tolist();pairs=[]
for body in candidates:
    extent=np.asarray(body['upper'])-body['lower']
    if body.get('layer')!='full' or not(limits[0]<min(extent[:2]) and max(extent[:2])<limits[1] and limits[2]<extent[2]<limits[3]):continue
    if max(extent[:2])/min(extent[:2])>limits[4]:continue
    for bar in candidates:
        be=np.asarray(bar['upper'])-bar['lower']
        if not bar.get('layer','').startswith('above_'):continue
        if max(be[:2])/max(min(be[:2]),1e-6)<limits[7] or be[2]<.012:continue
        if np.linalg.norm(np.asarray(bar['position'])[:2]-np.asarray(body['position'])[:2])>limits[6]:continue
        if bar['lower'][2]<=body['lower'][2]+.01 or bar['upper'][2]>body['upper'][2]+.005:continue
        if max(be[:2])>max(extent[:2])+limits[6]:continue
        pairs.append((body,bar))
if not pairs:raise ValueError('No visible circular control base with raised narrow handle')
model,processor=load('0.8B');model.requires_grad_(False)
stove_scores=[match_score(model,processor,folder/f"crop_{c['id']}.png",'an electric stove burner') for c in candidates]
stove=candidates[int(np.argmax(stove_scores))]
eligible=[pair for pair in pairs if np.linalg.norm(np.asarray(pair[0]['position'])[:2]-np.asarray(stove['position'])[:2])<limits[5]]
if not eligible:raise ValueError('No visible rotary control near the visually selected stove')
body_scores={body['id']:match_score(model,processor,folder/f"crop_{body['id']}.png",'a stove control knob') for body,_ in eligible}
body,bar=max(eligible,key=lambda pair:body_scores[pair[0]['id']])
plan=dict(skill='twist',source_id=bar['id'],control_body_id=body['id'],stove_id=stove['id'],body_scores=body_scores,stove_scores=stove_scores,
          provenance='Frozen Qwen3.5-0.8B full-assembly semantics plus observed circular base and raised bar; no simulator identity')
(folder/'condition-twist.json').write_text(json.dumps(plan,indent=2)+'\n');print(json.dumps(plan),flush=True)
