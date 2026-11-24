"""One instruction-conditioned grounding path for single and double pick/place."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
import numpy as np
import torch
from program_parser import parse_program
from relation_expert import RelationExpert
from semantic_grounding import load,match_score
from semantic_phrasing import visual_phrase
from semantic_relations import parse_relation
from robot_observation_features import robot_overlap


def target_relation(text):
    match=re.fullmatch(r'(left|right) of (?:the )?(.+)',text)
    if match:return match.group(2),'offset_'+match.group(1)
    match=re.fullmatch(r'back compartment of (?:the )?(.+)',text)
    if match:return match.group(1),'compartment_back'
    match=re.fullmatch(r'(left|right) (.+)',text)
    if match:return match.group(2),match.group(1)
    return text,'none'


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--weights',type=Path,required=True);p.add_argument('--relations',type=Path,required=True)
    a=p.parse_args();folder=a.capture;instruction=json.loads((folder/'instruction.json').read_text())['instruction']
    program=parse_program(instruction);record=json.loads((folder/'candidates.json').read_text());candidates=record['candidates']
    state=torch.load(a.weights,weights_only=True);gains=state['assignment_gain'].tolist();relation=RelationExpert(a.relations)
    overlap=robot_overlap(np.load(folder/'sensors.npz'),candidates,relation.robot_boxes.numpy(),float(relation.robot_gate[2])) if hasattr(relation,'robot_boxes') else [0.]*len(candidates)
    (folder/'joint_robot_overlap.json').write_text(json.dumps([dict(id=c['id'],robot_fraction=f) for c,f in zip(candidates,overlap)],indent=2)+'\n')
    descriptions=[];phrases=[]
    for clause in program:
        noun,mode,refs=parse_relation(clause['source']);target,side=target_relation(clause['target'])
        descriptions.append(dict(noun=noun,mode=mode,refs=refs,target=target,side=side))
        phrases.extend([noun,target,*refs])
    model,processor=load('0.8B');model.requires_grad_(False);scores={};view_scores={}
    for phrase in dict.fromkeys(phrases):
        primary=[match_score(model,processor,folder/f"crop_{c['id']}.png",visual_phrase(phrase)) for c in candidates]
        if hasattr(relation,'view_weights'):
            wrist=[match_score(model,processor,folder/f"wrist_crop_{c['id']}.png",visual_phrase(phrase)) if (folder/f"wrist_crop_{c['id']}.png").exists() else float('nan') for c in candidates]
            scores[phrase]=relation.fuse_views(list(zip(primary,wrist))).tolist()
            view_scores[phrase]=[dict(id=c['id'],primary=p,wrist=None if np.isnan(w) else w) for c,p,w in zip(candidates,primary,wrist)]
        else:scores[phrase]=primary
        scores[phrase]=relation.exclude_robot(scores[phrase],overlap).tolist()
        print(phrase,visual_phrase(phrase),scores[phrase],flush=True)
        (folder/'joint_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
        (folder/'joint_view_scores.json').write_text(json.dumps(view_scores,indent=2)+'\n')
    source_scores=[];debug=[]
    for clause in descriptions:
        refs=[int(np.argmax(scores[r])) for r in clause['refs']];center=None
        if clause['mode']=='center':
            xyz=np.load(folder/'sensors.npz')['world'];points=xyz[(abs(xyz[...,2]-record['table_height'])<.003)&(abs(xyz[...,0])<.45)&(abs(xyz[...,1])<.5)]
            center=((np.quantile(points[:,:2],.02,axis=0)+np.quantile(points[:,:2],.98,axis=0))/2).tolist()
        _,combined,features=relation(candidates,scores[clause['noun']],clause['mode'],refs,bowl='bowl' in clause['noun'],center=center)
        source_scores.append(combined.tolist());debug.append(dict(references=refs,combined=combined.tolist(),features=features.tolist()))

    def distinct(indices):
        if len(set(indices))!=len(indices):return False
        if len(indices)==1:return True
        aa,bb=[candidates[i]['box'] for i in indices]
        overlap=max(0,min(aa[2],bb[2])-max(aa[0],bb[0]))*max(0,min(aa[3],bb[3])-max(aa[1],bb[1]))
        area=min((aa[2]-aa[0])*(aa[3]-aa[1]),(bb[2]-bb[0])*(bb[3]-bb[1]))
        return overlap/max(area,1)<.5

    choices=(ids for ids in itertools.product(range(len(candidates)),repeat=len(program)) if distinct(ids))
    selected=max(choices,key=lambda ids:sum(gains[k]*source_scores[k][i] for k,i in enumerate(ids)))
    steps=[]
    for k,clause in enumerate(descriptions):
        options=[j for j in range(len(candidates)) if j not in selected]
        if clause['target'] in ('caddy','basket') and hasattr(relation,'container_bounds'):
            bounds=relation.container_bounds.tolist()
            options=[j for j in options if (lambda extent: min(extent[:2])>=bounds[0] and max(extent[:2])<=bounds[1] and extent[2]<=bounds[2])(np.asarray(candidates[j]['upper'])-candidates[j]['lower'])]
        if clause['side'] in ('left','right'):
            best=max(scores[clause['target']][j] for j in options)
            options=[j for j in options if scores[clause['target']][j]>=max(0.,best-3.)]
            target=min(options,key=lambda j:(1 if clause['side']=='left' else -1)*sum(candidates[j]['box'][::2]))
        else:target=max(options,key=lambda j:scores[clause['target']][j])
        steps.append(dict(source_id=candidates[selected[k]]['id'],target_id=candidates[target]['id'],
                          source_kind='bowl' if 'bowl' in clause['noun'] else 'other',skill='pick_place',
                          support_mode=program[k].get('support_mode','other'),
                          source_phrase=program[k]['source'],target_phrase=program[k]['target'],relation=clause['mode'],placement_relation=clause['side']))
    plan=dict(steps=steps,visual_phrases={s:visual_phrase(s) for s in scores},
              provenance='Frozen Qwen3.5-0.8B visual margins, authored phrases and geometric relation expert, distinct-region assignment',
              cameras=['agentview','robot0_eye_in_hand'] if hasattr(relation,'view_weights') else ['agentview'],
              action_weight_sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),relation_weight_sha256=hashlib.sha256(a.relations.read_bytes()).hexdigest())
    (folder/'condition-program.json').write_text(json.dumps(plan,indent=2)+'\n')
    (folder/'joint_debug.json').write_text(json.dumps(debug,indent=2)+'\n');print(json.dumps(plan),flush=True)


if __name__=='__main__':main()
