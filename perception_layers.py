"""Deterministic height-layer proposals for touching/stacked visible objects."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw,ImageFont


def propose(sensor,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    xyz=sensor['world'];rgb=sensor['rgb'];z=xyz[...,2]
    hist,edges=np.histogram(z,bins=np.arange(0,2,.004));i=hist.argmax()
    plane=float(np.median(z[(z>=edges[i])&(z<edges[i+1])]))
    workspace=(xyz[...,0]>-.6)&(xyz[...,0]<.45)&(abs(xyz[...,1])<.5)
    base=workspace&(z>plane+.007)&(z<plane+.4)
    candidates=[]

    def collect(mask,label):
        yy,xx=np.where(mask)
        cells=np.floor((xyz[mask]-[-.6,-.6,plane])/.004).astype(int)
        voxels=np.zeros((301,301,101),bool)
        voxels[cells[:,0],cells[:,1],cells[:,2]]=True
        components,n=ndimage.label(voxels,structure=np.ones((3,3,3)))
        ids=components[cells[:,0],cells[:,1],cells[:,2]]
        for j in range(1,n+1):
            select=ids==j
            if select.sum()<65:continue
            py,px=yy[select],xx[select];points=xyz[py,px]
            lo,hi=np.quantile(points,[.02,.98],axis=0)
            if max(hi[:2]-lo[:2])>.6:continue
            box=np.array([px.min(),py.min(),px.max(),py.max()])
            duplicate=False
            for prior in candidates:
                pb=np.array(prior['box']);wh=np.maximum(0,np.minimum(box[2:],pb[2:])-np.maximum(box[:2],pb[:2]))
                overlap=float(np.prod(wh));union=float(np.prod(box[2:]-box[:2])+np.prod(pb[2:]-pb[:2])-overlap)
                if overlap/max(union,1)>.8 and abs(hi[2]-prior['upper'][2])<.015:
                    duplicate=True;break
            if duplicate:continue
            candidates.append(dict(id=len(candidates)+1,box=box.tolist(),position=((lo+hi)/2).tolist(),
                                   lower=lo.tolist(),upper=hi.tolist(),pixels=int(select.sum()),layer=label))

    collect(base,'full')
    for offset in [.03,.07,.14,.24,.32]:
        collect(base&(z>plane+offset),f'above_{offset}')
    # Remove raised-object footprints from the low band to expose a nearby flat plate.
    high=base&(z>plane+.03)
    cells=np.floor((xyz[high,:2]+.6)/.004).astype(int)
    occupied=np.zeros((301,301),bool);occupied[cells[:,0],cells[:,1]]=True
    occupied=ndimage.binary_dilation(ndimage.binary_fill_holes(occupied),iterations=1)
    cells_all=np.floor((xyz[...,:2]+.6)/.004).astype(int).clip(0,300)
    covered=occupied[cells_all[...,0],cells_all[...,1]]
    collect(base&(z<plane+.03)&~covered,'low_uncovered')
    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',max(18,rgb.shape[0]//40))
    for c in candidates:
        x0,y0,x1,y1=c['box'];draw.rectangle(c['box'],outline='red',width=2)
        draw.rectangle([x0,y0,x0+40,y0+30],fill='white');draw.text((x0+2,y0),str(c['id']),fill='black',font=font)
        Image.fromarray(rgb[max(0,y0-8):y1+9,max(0,x0-8):x1+9]).save(out/f"crop_{c['id']}.png")
    image.save(out/'candidates.png')
    report=dict(method='xyz_height_layers_v2_wide_containers',table_height=plane,candidates=candidates)
    (out/'candidates.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args()
    print(json.dumps(propose(np.load(a.capture/'sensors.npz'),a.capture),indent=2))
