"""Authored recurrent skill selector; receives motor completion, never success."""
import torch
from torch import nn


class SequenceActionExpert(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        for name,value in torch.load(checkpoint,weights_only=True).items():
            self.register_parameter(name,nn.Parameter(value,requires_grad=False))
        self.state=torch.tensor([1.,0.,0.]);self.age=torch.tensor(0.)

    @property
    def index(self):return int(self.state.argmax())

    @torch.no_grad()
    def observe(self,motor_finished):
        self.age=self.age+1 if motor_finished else torch.zeros_like(self.age)
        if self.age<self.completion_dwell:return False
        self.state=self.transition@self.state;self.age=torch.zeros_like(self.age)
        return True
