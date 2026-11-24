"""Deterministic RGB-D proposals; no object labels or simulator instances."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw, ImageFont


def propose(sensor, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    xyz=sensor['world']; rgb=sensor['rgb']; z=xyz[...,2]
    hist,edges=np.histogram(z,bins=np.arange(0,2,.004))
    plane_bin=hist.argmax()
    plane=float(np.median(z[(z>=edges[plane_bin]) & (z<edges[plane_bin+1])]))
    mask=(z>plane+.007)&(z<plane+.40)&(xyz[...,0]>-.6)&(xyz[...,0]<.45)&(abs(xyz[...,1])<.5)
    # Metric XYZ connectivity preserves height separation when silhouettes overlap.
    yy_valid,xx_valid=np.where(mask)
    cells=np.floor((xyz[mask]-np.array([-.6,-.6,plane]))/.004).astype(int)
    occupied=np.zeros((301,301,101),dtype=bool)
    occupied[cells[:,0],cells[:,1],cells[:,2]]=True
    grid_labels,n=ndimage.label(occupied,structure=np.ones((3,3,3)))
    labels=np.zeros(mask.shape,dtype=int)
    labels[yy_valid,xx_valid]=grid_labels[cells[:,0],cells[:,1],cells[:,2]]
    image=Image.fromarray(rgb); draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',18)
    candidates=[]
    for lab in range(1,n+1):
        yy,xx=np.where(labels==lab)
        if len(xx)<65: continue
        points=xyz[yy,xx]; lo,hi=np.quantile(points,[.02,.98],axis=0)
        if max(hi[:2]-lo[:2])>.6: continue
        center=(lo+hi)/2
        box=[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]
        cid=len(candidates)+1
        candidates.append(dict(id=cid,box=box,position=center.tolist(),lower=lo.tolist(),upper=hi.tolist(),pixels=len(xx)))
        draw.rectangle(box,outline='red',width=2)
        x,y=box[:2]; draw.rectangle([x,y,x+24,y+23],fill='white'); draw.text((x+2,y),str(cid),fill='black',font=font)
        Image.fromarray(rgb[max(0,box[1]-8):box[3]+9,max(0,box[0]-8):box[2]+9]).save(out/f'crop_{cid}.png')
    image.save(out/'candidates.png')
    report=dict(method='xyz_4mm_26connectivity_v3_wide_containers',table_height=plane,candidates=candidates)
    (out/'candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);args=p.parse_args()
    print(json.dumps(propose(np.load(args.capture/'sensors.npz'),args.capture),indent=2))
