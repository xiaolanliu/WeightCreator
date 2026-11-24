"""Eight authored phases for pinching and rotating a visible control handle."""
import torch
from torch import nn


class TwistActionExpert(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        for k,v in torch.load(checkpoint,weights_only=True).items():
            self.register_parameter(k,nn.Parameter(v,requires_grad=False))

    @torch.no_grad()
    def prepare(self,position,upper):
        point=torch.tensor(position,dtype=torch.float32);point[2]=float(upper[2])+self.grasp_z
        self.condition=torch.cat([point,torch.ones(1)])
        self.phase=torch.nn.functional.one_hot(torch.tensor(0),8).float();self.age=torch.tensor(0.)
        self.finished=False

    @torch.no_grad()
    def forward(self,position,rotation_error):
        target=torch.einsum('s,sij,j->i',self.phase,self.target_weight,self.condition)
        error=target-torch.as_tensor(position,dtype=torch.float32)
        rotation=torch.as_tensor(rotation_error,dtype=torch.float32)
        limit=self.phase@self.action_limit
        action=(self.servo@torch.cat([error,rotation])).clamp(-limit,limit)
        action=torch.cat([action,(self.phase@self.gripper).reshape(1)])
        tolerance=self.phase@self.tolerance
        near=(error.abs().max()<tolerance[0]) & (rotation.abs().max()<tolerance[1])
        self.age=torch.where(near,self.age+1,torch.zeros_like(self.age))
        advance=self.age>=self.phase@self.dwell
        if advance:
            self.finished=int(self.phase.argmax())==7
            self.phase=self.transition@self.phase;self.age=torch.tensor(0.)
        return action,target,error
