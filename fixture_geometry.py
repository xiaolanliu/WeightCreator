"""Visible protruding handle geometry inside an image-selected fixture proposal."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw


def handle_hypotheses(sensor,candidate):
    x0,y0,x1,y1=candidate['box'];xyz=sensor['world'];yy,xx=np.mgrid[y0:y1+1,x0:x1+1]
    points=xyz[y0:y1+1,x0:x1+1].reshape(-1,3);pixels=np.column_stack([xx.ravel(),yy.ravel()])
    lo=np.asarray(candidate['lower']);hi=np.asarray(candidate['upper'])
    valid=np.all((points[:,:2]>lo[:2]-.025)&(points[:,:2]<hi[:2]+.025),axis=1)&(points[:,2]>lo[2]+.015)&(points[:,2]<hi[2]-.025)
    points=points[valid];pixels=pixels[valid]
    if len(points)<20:return []
    rows=[]
    for normal in [np.array([1.,0.,0.]),np.array([-1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,-1.,0.])]:
        tangent=np.array([-normal[1],normal[0],0.]);projection=points@normal
        edges=np.arange(projection.min()-.004,projection.max()+.008,.004)
        hist,_=np.histogram(projection,bins=edges)
        # Keep the bin containing the threshold; testing only its left edge can
        # exclude the dominant front plane and incorrectly use the handle peak.
        eligible=edges[1:]>=np.quantile(projection,.60)
        if not eligible.any():continue
        mode=int(np.argmax(hist*eligible));band=(projection>=edges[mode])&(projection<edges[mode+1])
        if not band.any():continue
        plane=float(np.median(projection[band]));outside=(projection>plane+.008)&(projection<plane+.065)
        visible=points[outside];px=pixels[outside]
        if len(visible)<20:continue
        origin=visible.min(0)-.004;cells=np.floor((visible-origin)/.004).astype(int)
        occupancy=np.zeros(tuple(cells.max(0)+2),bool);occupancy[tuple(cells.T)]=True
        labels,_=ndimage.label(occupancy,structure=np.ones((3,3,3)));ids=labels[tuple(cells.T)]
        handles=[]
        for label in np.unique(ids):
            which=ids==label;part=visible[which]
            if len(part)<20:continue
            lower,upper=np.quantile(part,[.02,.98],axis=0)
            horizontal=part@tangent;depth=part@normal
            length=float(np.quantile(horizontal,.98)-np.quantile(horizontal,.02));thickness=float(upper[2]-lower[2])
            if not (.045<length<.20 and .004<thickness<.045 and np.ptp(depth)<.065):continue
            box=[int(px[which,0].min()),int(px[which,1].min()),int(px[which,0].max()),int(px[which,1].max())]
            handles.append(dict(position=((lower+upper)/2).tolist(),lower=lower.tolist(),upper=upper.tolist(),box=box,
                                points=len(part),length=length,thickness=thickness))
        handles.sort(key=lambda x:x['position'][2])
        rows.append(dict(normal=normal.tolist(),plane=plane,handles=handles,count=len(handles)))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('--candidate',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    record=json.loads((a.capture/'candidates.json').read_text());candidate=next(c for c in record['candidates'] if c['id']==a.candidate)
    sensor=np.load(a.capture/'sensors.npz');rows=handle_hypotheses(sensor,candidate);a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'handles.json').write_text(json.dumps(rows,indent=2)+'\n')
    image=Image.fromarray(sensor['rgb']);draw=ImageDraw.Draw(image)
    for i,row in enumerate(rows):
        for j,h in enumerate(row['handles']):draw.rectangle(h['box'],outline='red',width=3);draw.text(h['box'][:2],f'{i}:{j}',fill='red')
    image.save(a.out/'handles.png');print(json.dumps(rows,indent=2))
