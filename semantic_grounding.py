"""Use frozen VLM image-conditioned Yes/No logits to ground instruction phrases."""
import argparse
import json
import re
import sys
from pathlib import Path
import torch
from backbone import load


def phrases(instruction):
    text=instruction.lower().strip()
    if text.startswith('pick up '):
        pattern=r'pick up (?:the )?(.+?) and (?:place|put) it (?:on top of|on|into|in) (?:the )?(.+)$'
    else:
        pattern=r'put (?:the )?(.+?) (?:on top of|on|into|in) (?:the )?(.+)$'
    m=re.fullmatch(pattern,text)
    if not m or ' and ' in m.group(2):
        raise ValueError(f'Unsupported instruction grammar: {text}')
    return m.group(1),m.group(2)


@torch.inference_mode()
def match_score(model,processor,image,phrase):
    prompt=f'Is the main object in this image {phrase}? Answer Yes or No.'
    inputs=processor.apply_chat_template([{'role':'user','content':[
        {'type':'image','image':str(image.resolve())},{'type':'text','text':prompt}]}],
        tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to(model.device)
    logits=model(**inputs).logits[0,-1].float()
    yes=processor.tokenizer.encode('Yes',add_special_tokens=False)
    no=processor.tokenizer.encode('No',add_special_tokens=False)
    assert len(yes)==len(no)==1
    return float(logits[yes[0]]-logits[no[0]])


def ground_folder(model,processor,folder):
    instruction=json.loads((folder/'instruction.json').read_text())['instruction']
    source_phrase,target_phrase=phrases(instruction)
    candidates=json.loads((folder/'candidates.json').read_text())['candidates']
    scores={}
    for role,phrase in [('source',source_phrase),('target',target_phrase)]:
        scores[role]={}
        for c in candidates:
            score=match_score(model,processor,folder/f"crop_{c['id']}.png",phrase)
            scores[role][c['id']]=score
            print(role,c['id'],round(score,3),flush=True)
    source=max(scores['source'],key=scores['source'].get)
    target=max((i for i in scores['target'] if i!=source),key=scores['target'].get)
    kind=next((k for k in ['bowl','bottle','box'] if k in source_phrase),'other')
    plan=dict(source_id=source,target_id=target,source_kind=kind,skill='pick_place',
              provenance='frozen Qwen3.5-0.8B image-conditioned Yes/No logit margins; generic instruction grammar',
              source_phrase=source_phrase,target_phrase=target_phrase)
    (folder/'grounding_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    (folder/'condition-grounding.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path,nargs='*')
    p.add_argument('--serve',action='store_true');args=p.parse_args()
    model,processor=load('0.8B');model.requires_grad_(False)
    if args.serve:
        for line in sys.stdin:
            folder=Path(json.loads(line)['capture'])
            try:
                ground_folder(model,processor,folder)
                status=dict(capture=str(folder),ok=True)
            except Exception as error:
                status=dict(capture=str(folder),ok=False,error=repr(error))
            print('GROUNDING_RESULT '+json.dumps(status),flush=True)
        return
    for folder in args.capture:
        print(str(folder),flush=True)
        ground_folder(model,processor,folder)


if __name__=='__main__':main()
