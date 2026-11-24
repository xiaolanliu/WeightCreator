"""Hand-authored semantic/geometric referent selection; no fitting or optimization."""
from pathlib import Path
import torch
from torch import nn


class RelationExpert(nn.Module):
    MODES=('none','near','between','on','center','inside')

    def __init__(self,checkpoint):
        super().__init__()
        for name,value in torch.load(checkpoint,weights_only=True).items():
            self.register_parameter(name,nn.Parameter(value,requires_grad=False))

    @torch.no_grad()
    def fuse_views(self,margins):
        values=torch.as_tensor(margins,dtype=torch.float32)
        present=torch.isfinite(values)
        weights=present*self.view_weights
        mean=(values.nan_to_num()*weights).sum(-1)/weights.sum(-1).clamp_min(1e-8)
        if hasattr(self,'view_reduce'):
            maximum=values.masked_fill(~present,float('-inf')).amax(-1)
            return torch.stack((mean,maximum),dim=-1)@self.view_reduce
        return mean

    @torch.no_grad()
    def exclude_robot(self,semantic,overlap):
        values=torch.as_tensor(semantic,dtype=torch.float32)
        if hasattr(self,'robot_gate'):
            values=values-(torch.as_tensor(overlap)>self.robot_gate[0]).float()*self.robot_gate[1]
        return values

    @torch.no_grad()
    def forward(self,candidates,semantic,mode,references,bowl=False,center=None):
        pos=torch.tensor([c['position'] for c in candidates]);lo=torch.tensor([c['lower'] for c in candidates]);hi=torch.tensor([c['upper'] for c in candidates])
        sem=torch.tensor(semantic);zero=torch.zeros(len(candidates))
        distance=zero.clone();outside=zero.clone();height=zero.clone()
        if mode in ('near','on','inside'):
            anchor=pos[references[0]]
            distance=torch.linalg.vector_norm(pos[:,:2]-anchor[:2],dim=1)
            if mode=='on':height=torch.relu(hi[references[0],2]-hi[:,2])
            if mode=='inside':height=torch.relu(hi[:,2]-hi[references[0],2])
        elif mode=='between':
            a,b=pos[references[:2],:2];vector=b-a
            coordinate=((pos[:,:2]-a)@vector)/(vector@vector).clamp_min(1e-8)
            projection=a+coordinate.clamp(0,1).unsqueeze(1)*vector
            distance=torch.linalg.vector_norm(pos[:,:2]-projection,dim=1)
            outside=torch.relu(-coordinate)+torch.relu(coordinate-1)
        elif mode=='center':
            distance=torch.linalg.vector_norm(pos[:,:2]-torch.as_tensor(center),dim=1)
        ext=(hi-lo)[:,:2].clamp_min(.001)
        shape=torch.relu(torch.log(ext.max(1).values/ext.min(1).values)-torch.log(torch.tensor(1.5)))*float(bowl)
        features=torch.stack([sem,-distance,-outside,-height,-shape],dim=1)
        score=features@self.feature_weights[self.MODES.index(mode)]
        if references:score[references]-=self.reference_exclusion
        if bowl and hasattr(self,'bowl_bounds'):
            extent=hi-lo
            plausible=(extent[:,:2].min(1).values>=self.bowl_bounds[0]) & (extent[:,:2].max(1).values<=self.bowl_bounds[1])
            plausible &= (extent[:,2]>=self.bowl_bounds[2]) & (extent[:,2]<=self.bowl_bounds[3])
            plausible &= sem>=self.semantic_floor
            score=score-(~plausible).float()*self.reference_exclusion
        return int(score.argmax()),score,features


def author(path):
    # Logit units vs distances in meters; no example data are used to obtain these values.
    weights=torch.tensor([[1,0,0,0,3],[1,40,0,0,3],[1,80,40,0,3],[1,50,0,50,3],[1,15,0,0,3]],dtype=torch.float32)
    torch.save(dict(feature_weights=weights,reference_exclusion=torch.tensor(1000.)),path)


if __name__=='__main__':author(Path(__file__).parent/'weights/relation_v0.pt')
