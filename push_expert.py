"""Authored planar pushing geometry and recurrent control, no object truth."""
import numpy as np
import torch
from expert import WeightActionExpert


class PushActionExpert(WeightActionExpert):
    @torch.no_grad()
    def prepare_push(self,source,target,camera_position):
        s=np.asarray(source['position']);t=np.asarray(target['position'])
        se=np.asarray(source['upper'])-source['lower'];te=np.asarray(target['upper'])-target['lower']
        view=np.asarray(camera_position)[:2]-t[:2];front=np.zeros(2)
        axis=int(np.argmax(abs(view)));front[axis]=np.sign(view[axis])
        endpoint=t[:2]+front*(.5*abs(front)@te[:2]+.5*abs(front)@se[:2]+float(self.push_geometry[0]))
        direction=endpoint-s[:2];direction/=max(np.linalg.norm(direction),1e-6)
        radius=.5*np.sqrt(np.sum((direction*se[:2])**2))
        offset=direction*(radius+float(self.push_geometry[1]));z=source['upper'][2]+float(self.push_geometry[2])
        start=np.r_[s[:2]-offset,z];finish=np.r_[endpoint-offset,z]
        self.condition=torch.tensor(np.r_[start,finish,1.],dtype=torch.float32)
        self.phase=torch.nn.functional.one_hot(torch.tensor(0),8).float();self.age=torch.tensor(0.)
        self.previous_pos=None;self.stall_age=torch.tensor(0.)
        return dict(start=start.tolist(),finish=finish.tolist(),plate_destination=endpoint.tolist(),front_axis=front.tolist(),direction=direction.tolist())
