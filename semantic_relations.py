"""Frozen VLM noun scores conditioned by the authored relation expert."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from semantic_grounding import load,match_score,phrases
from relation_expert import RelationExpert
from semantic_phrasing import visual_phrase


def parse_relation(source):
    inside=re.fullmatch(r'(.+?) in (?:the )?(?:top|middle|bottom) drawer of (?:the )?(.+)',source)
    if inside:return inside.group(1),'inside',[inside.group(2)]
    matches=[('between',r'(.+?) between (?:the )?(.+?) and (?:the )?(.+)'),
             ('near',r'(.+?) next to (?:the )?(.+)'),
             ('on',r'(.+?) on (?:top of )?(?:the )?(.+)'),
             ('center',r'(.+?) from table center')]
    for mode,pattern in matches:
        m=re.fullmatch(pattern,source)
        if m:return m.group(1),mode,list(m.groups()[1:])
    return source,'none',[]


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path)
    p.add_argument('--weights',type=Path,default=Path(__file__).parent/'weights/relation_v2.pt')
    p.add_argument('--canonical-phrases',action='store_true')
    a=p.parse_args();folder=a.capture
    instruction=json.loads((folder/'instruction.json').read_text())['instruction']
    source,target=phrases(instruction);noun,mode,refs=parse_relation(source)
    record=json.loads((folder/'candidates.json').read_text());candidates=record['candidates']
    model,processor=load('0.8B');model.requires_grad_(False)
    scores={}
    for phrase in dict.fromkeys([noun,target,*refs]):
        query=visual_phrase(phrase) if a.canonical_phrases else phrase
        scores[phrase]=[match_score(model,processor,folder/f"crop_{c['id']}.png",query) for c in candidates]
        print(phrase,scores[phrase],flush=True)
        (folder/'relation_vlm_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    references=[int(np.argmax(scores[r])) for r in refs]
    center=None
    if mode=='center':
        sensor=np.load(folder/'sensors.npz');xyz=sensor['world']
        points=xyz[(abs(xyz[...,2]-record['table_height'])<.003)&(abs(xyz[...,0])<.45)&(abs(xyz[...,1])<.5)]
        center=((np.quantile(points[:,:2],.02,axis=0)+np.quantile(points[:,:2],.98,axis=0))/2).tolist()
    relation=RelationExpert(a.weights)
    index,combined,features=relation(candidates,scores[noun],mode,references,bowl='bowl' in noun,center=center)
    target_index=max((j for j in range(len(candidates)) if j!=index),key=lambda j:scores[target][j])
    plan=dict(source_id=candidates[index]['id'],target_id=candidates[target_index]['id'],source_kind='bowl' if 'bowl' in noun else 'other',skill='pick_place',
              provenance='frozen Qwen3.5-0.8B image noun margins + analytically authored relation_v0',source_phrase=source,target_phrase=target,relation=mode,
              relation_weights=str(a.weights),relation_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest())
    plan['visual_phrases']={phrase:visual_phrase(phrase) for phrase in scores} if a.canonical_phrases else None
    (folder/'relation_debug.json').write_text(json.dumps(dict(reference_ids=[candidates[j]['id'] for j in references],combined=combined.tolist(),features=features.tolist()),indent=2)+'\n')
    (folder/'condition-relation.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan),flush=True)


if __name__=='__main__':main()
