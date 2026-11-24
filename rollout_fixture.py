"""Native LIBERO evaluator for the authored drawer primitive."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from capture import make_env
from fixture_expert import FixtureActionExpert

p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--suite',default='libero_goal');p.add_argument('--task',type=int,required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--weights',type=Path,required=True)
p.add_argument('--split',default='development',choices=['development','frozen-heldout'])
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
episode=json.loads((a.capture/'episode.json').read_text());assert (episode['suite'],episode['task'],episode['trial'])==(a.suite,a.task,a.trial)
plan=json.loads((a.capture/'condition-fixture.json').read_text());assert plan['skill']=='drawer_open'
normal=np.asarray(plan['normal']);depth=float(abs(normal)@np.asarray(plan['fixture_extent']))
head=FixtureActionExpert(a.weights);head.prepare_fixture(plan['handle']['position'],normal,depth)
z_axis=-normal;y_axis=np.array([0.,0.,1.]);x_axis=np.cross(y_axis,z_axis);desired=np.column_stack([x_axis,y_axis,z_axis])
env,obs,instruction=make_env(a.suite,a.task,a.trial,size=256);assert instruction==json.loads((a.capture/'instruction.json').read_text())['instruction']
trace=[];first=None;success=False;settled=0;limit={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}[a.suite]
try:
    for step in range(limit):
        phase=int(head.phase.argmax());current=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        action,target,error=head(obs['robot0_eef_pos'],Rotation.from_matrix(desired@current.T).as_rotvec(),obs['robot0_gripper_qpos'])
        obs,_,_,_=env.step(action.numpy());success=bool(env.check_success())
        if success and first is None:first=step
        trace.append(dict(step=step,phase=phase,eef=obs['robot0_eef_pos'].tolist(),gripper=obs['robot0_gripper_qpos'].tolist(),action=action.tolist(),target=target.tolist(),success=success))
        if step%10==0 or success:
            Image.fromarray(obs['agentview_image'][::-1]).save(a.out/f'frame_{step:04}.png');Image.fromarray(obs['robot0_eye_in_hand_image'][::-1]).save(a.out/f'wrist_{step:04}.png')
        if step%20==0:print(step,phase,error.tolist(),obs['robot0_gripper_qpos'].tolist(),flush=True)
        settled=settled+1 if phase==7 and float(error.abs().max())<.02 else 0
        if settled>=20:break
finally:env.close()
result=dict(success=success,first_success_step=first,steps=len(trace),native_step_limit=limit,primitive_finished=settled>=20,released=any(t['phase']>=6 for t in trace),
            suite=a.suite,task=a.task,trial=a.trial,instruction=instruction,condition=plan,weights=str(a.weights),sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
            parameter_count=sum(p.numel() for p in head.parameters()),capture_path=str(a.capture.resolve()),split=a.split,sensor_protocol='RGB-D + calibrated camera + proprioception; no object joint state or evaluator input')
(a.out/'trace.json').write_text(json.dumps(trace,indent=2)+'\n');(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
