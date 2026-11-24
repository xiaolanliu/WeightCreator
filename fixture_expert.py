"""Authored drawer-opening dynamics using only robot pose and finger positions."""
import torch
from expert import WeightActionExpert


class FixtureActionExpert(WeightActionExpert):
    @torch.no_grad()
    def prepare_fixture(self,handle,normal,depth):
        source=torch.tensor(handle,dtype=torch.float32);direction=torch.tensor(normal,dtype=torch.float32)
        stroke=(self.stroke_config[0]*depth).clamp(self.stroke_config[1],self.stroke_config[2])
        self.condition=torch.cat((source,source+direction*stroke,torch.ones(1)))
        self.phase=torch.nn.functional.one_hot(torch.tensor(0),8).float()
        self.age=torch.tensor(0.);self.stall_age=torch.tensor(0.);self.previous_pos=None

    @torch.no_grad()
    def forward(self,eef_pos,rotation_error,gripper_position):
        before=int(self.phase.argmax())
        action,target,error=super().forward(eef_pos,rotation_error)
        if before in (0,1,6,7):
            gap=torch.as_tensor(gripper_position,dtype=torch.float32).abs().sum()
            difference=gap-self.pregrip_config[0]
            action[-1]=torch.where(difference.abs()>self.pregrip_config[1],difference.sign(),torch.tensor(0.))
        if before==3 and int(self.phase.argmax())==4:
            # Proprioceptive stop point, not the fixture's hidden joint coordinate.
            self.condition[3:6]=torch.as_tensor(eef_pos,dtype=torch.float32)
        return action,target,error
