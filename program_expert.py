"""Authored recurrent program state above the eight-phase motor expert."""
import torch
from expert import WeightActionExpert


class ProgramActionExpert(WeightActionExpert):
    def start(self,count):
        assert 1<=count<=2
        self.program_count=count
        self.program_state=torch.nn.functional.one_hot(torch.tensor(0),3).float()
        self.program_age=torch.tensor(0.)

    @property
    def program_index(self):return int(self.program_state.argmax())

    @torch.no_grad()
    def observe_completion(self,error,phase):
        last=self.program_index==self.program_count-1
        near=(phase==7) and float(error.abs().max())<float(self.program_gate[0])
        self.program_age=self.program_age+1 if near else torch.tensor(0.)
        if self.program_age<float(self.program_gate[2 if last else 1]):return False
        self.program_state=self.program_transition@self.program_state
        self.program_age=torch.tensor(0.)
        return True

    @torch.no_grad()
    def placement_offset(self,extent,shared_target):
        return (self.program_state[:2]@self.target_slots)*torch.as_tensor(extent,dtype=torch.float32)*float(shared_target)
