"""Analytically parameterized recurrent action expert. No training operations."""
from pathlib import Path
import torch
from torch import nn


class WeightActionExpert(nn.Module):
    # Eight recurrent modes, each with an affine Cartesian target and dwell gate.
    PHASES=('approach','descend','close','lift','transfer','lower','release','retreat')

    def __init__(self, checkpoint):
        super().__init__()
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        for name,value in state.items():
            self.register_parameter(name,nn.Parameter(value,requires_grad=False))

    @torch.no_grad()
    def prepare(self,source,target,kind,grasp_shift=None):
        self.source_kind=kind
        s=torch.tensor(source['position'],dtype=torch.float32)
        t=torch.tensor(target['position'],dtype=torch.float32)
        s[2]=float(source['upper'][2]); t[2]=float(target['upper'][2])
        bowl=float(kind=='bowl')
        extent=torch.tensor(source['upper'])-torch.tensor(source['lower'])
        self.solid_grasp=bool(not bowl and hasattr(self,'tall_grasp') and extent[2]>self.tall_grasp[0]*extent[:2].max())
        grasp_delta=self.grasp_offset[0]+bowl*self.grasp_offset[1]*extent
        if hasattr(self,'tall_grasp') and not bowl and extent[2]>self.tall_grasp[0]*extent[:2].max():
            grasp_delta[2]=torch.minimum(grasp_delta[2],-self.tall_grasp[1]*extent[2])
        if grasp_shift is not None:
            grasp_delta=grasp_delta+torch.as_tensor(grasp_shift,dtype=torch.float32)
        s=s+grasp_delta
        if hasattr(self,'carry_offset'):
            t=t+self.carry_offset@grasp_delta
        if hasattr(self,'placement_height'):
            grasp_height=s[2]-float(source['lower'][2])
            clearance=torch.maximum(self.placement_height[0],grasp_height+self.placement_height[1])
            # Transfer/release heights preserve the observed grasp-to-base distance.
            t[2]+=clearance-self.target_weight[5,2,6]
        self.condition=torch.cat((s,t,torch.ones(1)))
        self.phase=torch.nn.functional.one_hot(torch.tensor(0),8).float()
        self.age=torch.tensor(0.)
        self.previous_pos=None
        self.stall_age=torch.tensor(0.)
        self.release_aperture=None
        self.approach_height=None

    @torch.no_grad()
    def compartment_grip(self,gripper_qpos,phase,eef_z,rim_z):
        width=torch.as_tensor(gripper_qpos,dtype=torch.float32).abs().mean()
        if self.release_aperture is None:self.release_aperture=width+self.compartment_release[0]
        if phase==7 and eef_z>rim_z+float(self.compartment_release[2]):return -1.
        delta=width-self.release_aperture
        return float(torch.sign(delta)) if abs(delta)>self.compartment_release[1] else 0.

    @torch.no_grad()
    def grasp_yaw(self,geometry,initial_closing_axis):
        if not hasattr(self,'yaw_config') or self.source_kind=='bowl':
            return 0.
        axis=torch.as_tensor(geometry['minor_axis'],dtype=torch.float32)
        initial=torch.as_tensor(initial_closing_axis,dtype=torch.float32)
        angle=torch.atan2(axis[1],axis[0])-torch.atan2(initial[1],initial[0])
        # Parallel gripper jaws are invariant to a half-turn.
        angle=torch.remainder(angle+torch.pi/2,torch.pi)-torch.pi/2
        gate=float(geometry['aspect']>self.yaw_config[1])
        return float(self.yaw_config[0]*gate*angle.clamp(-self.yaw_config[2],self.yaw_config[2]))

    @torch.no_grad()
    def choose_grasp_offset(self,features):
        values=torch.as_tensor(features,dtype=torch.float32)
        scores=values[:,:2]@self.clearance_score
        if values.shape[1]==3:
            scores=scores+self.bowl_view_score*values[:,2]
        return int(scores.argmax()),scores.tolist()

    @torch.no_grad()
    def forward(self,eef_pos,rotation_error):
        matrix=torch.einsum('s,sij->ij',self.phase,self.target_weight)
        target=matrix@self.condition
        if hasattr(self,'level_bowl_transfer') and self.source_kind=='bowl':
            gate=self.phase[4]*self.level_bowl_transfer
            clearance=self.bowl_transfer_height if hasattr(self,'bowl_transfer_height') else self.target_weight[3,2,6]
            height=torch.maximum(target[2],self.condition[2]+clearance)
            target[2]=(1-gate)*target[2]+gate*height
        if hasattr(self,'bowl_lift_cap') and self.source_kind=='bowl' and self.approach_height is not None and float(self.phase[3]+self.phase[4])>.5:
            target[2]=torch.minimum(target[2],self.condition.new_tensor(self.approach_height)+self.bowl_lift_cap)
        error=target-torch.as_tensor(eef_pos,dtype=torch.float32)
        error6=torch.cat((error,torch.as_tensor(rotation_error,dtype=torch.float32)))
        limit=self.phase@self.phase_action_limit if hasattr(self,'phase_action_limit') else self.action_limit
        gain=self.phase@self.phase_servo_gain if hasattr(self,'phase_servo_gain') else 1.
        native=((self.servo@error6)*gain).clamp(-limit,limit)
        grip=self.phase@self.gripper
        action=torch.cat((native,grip.reshape(1)))
        tolerance=self.phase@self.tolerance
        if hasattr(self,'solid_grasp_tolerance') and self.solid_grasp:
            gate=self.phase[1]+self.phase[2]
            tolerance=(1-gate)*tolerance+self.phase[1:3]@self.solid_grasp_tolerance
        near=(error.abs().max()<tolerance).float()
        if hasattr(self,'rotation_gate'):
            near=near*(torch.as_tensor(rotation_error,dtype=torch.float32).abs().max()<self.phase@self.rotation_gate).float()
        if hasattr(self,'stall_config'):
            current_pos=torch.as_tensor(eef_pos,dtype=torch.float32)
            movement=torch.tensor(float('inf')) if self.previous_pos is None else (current_pos-self.previous_pos).abs().max()
            slow=(movement<self.stall_config[0]).float()
            self.stall_age=slow*(self.stall_age+1)
            eligible=(self.phase@self.stall_phase_mask)>.5
            contact_like=(self.stall_age>=self.stall_config[1]) & (error.abs().max()<self.stall_config[2]) & (error[:2].abs().max()<self.stall_config[3]) & eligible
            near=torch.maximum(near,contact_like.float())
            self.previous_pos=current_pos.clone()
        dwell=self.phase@self.dwell
        self.age=torch.where(near>0,self.age+1,torch.zeros_like(self.age))
        advance=(self.age>=dwell).float()
        if bool(advance) and float(self.phase[0])>.5:self.approach_height=float(torch.as_tensor(eef_pos)[2])
        self.phase=(1-advance)*self.phase+advance*(self.transition@self.phase)
        self.age=(1-advance)*self.age
        self.stall_age=(1-advance)*self.stall_age
        return action,target,error


def author_weights(out):
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    targets=torch.zeros(8,3,7)
    for i in range(8):
        targets[i,:,0 if i<4 else 3:3 if i<4 else 6]=torch.eye(3)
    targets[:,2,6]=torch.tensor([.16,0,0,.18,.18,.08,.08,.22])
    # Source top -> grasp point: shift onto a bowl rim, descend 15 mm below top.
    offsets=torch.tensor([[0,0,-.015],[-.30,0,0]],dtype=torch.float32)
    transition=torch.zeros(8,8)
    for i in range(8): transition[min(i+1,7),i]=1
    state=dict(target_weight=targets,grasp_offset=offsets,
               servo=torch.diag(torch.tensor([12.,12.,12.,4.,4.,4.])),
               action_limit=torch.tensor([.8,.8,.8,.5,.5,.5]),
               gripper=torch.tensor([-1.,-1.,1.,1.,1.,1.,-1.,-1.]),
               tolerance=torch.tensor([.01,.008,.018,.012,.012,.012,.018,.02]),
               dwell=torch.tensor([2.,2.,12.,2.,2.,2.,10.,2.]),transition=transition)
    torch.save(state,out)
    return {k:dict(shape=list(v.shape),numel=v.numel()) for k,v in state.items()}


if __name__=='__main__':
    import json
    path=Path(__file__).parent/'weights'/'expert_v0.pt'
    print(json.dumps(author_weights(path),indent=2))
