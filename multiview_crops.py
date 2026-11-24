"""Associate visible wrist-camera points with primary-camera geometric regions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();folder=a.capture
sensor=np.load(folder/'wrist_sensors.npz');world=sensor['world'];rgb=sensor['rgb'];record=json.loads((folder/'candidates.json').read_text());rows=[]
for candidate in record['candidates']:
    lower=np.asarray(candidate['lower'])-.008;upper=np.asarray(candidate['upper'])+.008
    mask=np.all((world>=lower)&(world<=upper),axis=2);yy,xx=np.where(mask)
    row=dict(id=candidate['id'],visible_pixels=int(len(xx)))
    if len(xx)>=25:
        box=[max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(rgb.shape[1],int(xx.max())+9),min(rgb.shape[0],int(yy.max())+9)]
        if min(box[2]-box[0],box[3]-box[1])>=16:
            Image.fromarray(rgb[box[1]:box[3],box[0]:box[2]]).save(folder/f"wrist_crop_{candidate['id']}.png");row['box']=box
    rows.append(row)
(folder/'multiview_visibility.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
