"""Native simulator evaluation of the observable planar-push expert."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from capture import make_env
from push_expert import PushActionExpert

p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
p.add_argument('--suite',required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--weights',type=Path,required=True)
p.add_argument('--split',default='development');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
episode=json.loads((a.capture/'episode.json').read_text());assert (episode['suite'],episode['task'],episode['trial'])==(a.suite,a.task,a.trial)
plan=json.loads((a.capture/'condition-push.json').read_text());candidates={c['id']:c for c in json.loads((a.capture/'candidates.json').read_text())['candidates']}
sensor=np.load(a.capture/'sensors.npz');head=PushActionExpert(a.weights)
geometry=head.prepare_push(candidates[plan['source_id']],candidates[plan['target_id']],sensor['camera_position'])
env,obs,instruction=make_env(a.suite,a.task,a.trial,size=256);assert instruction==json.loads((a.capture/'instruction.json').read_text())['instruction']
reference=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix();trace=[];first=None;success=False;settled=0
if hasattr(head,'push_yaws'):
    point=np.asarray(geometry['start']);bounds=head.palm_bounds.numpy();points=sensor['world'].reshape(-1,3)
    points=points[(np.linalg.norm(points[:,:2]-point[:2],axis=1)<.25)&(points[:,2]>point[2]+bounds[2])&(points[:,2]<point[2]+bounds[3])]
    points=np.unique(np.floor(points/.004).astype(int),axis=0).astype(float)*.004+.002;delta=points[:,:2]-point[:2]
    scores=[];orientations=[]
    for yaw in head.push_yaws.tolist():
        orientation=Rotation.from_euler('z',yaw).as_matrix()@reference;jaw=(orientation@head.closing_axis.numpy())[:2];jaw/=max(np.linalg.norm(jaw),1e-6);tangent=np.array([-jaw[1],jaw[0]])
        scores.append(float(head.clearance_gain)*np.count_nonzero((abs(delta@jaw)<bounds[0])&(abs(delta@tangent)<bounds[1])));orientations.append(orientation)
    selected=int(np.argmax(scores));reference=orientations[selected]
    geometry.update(palm_clearance_scores=scores,selected_yaw=float(head.push_yaws[selected]))
limit={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}[a.suite]
try:
    for step in range(limit):
        phase=int(head.phase.argmax());current=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        action,target,error=head(obs['robot0_eef_pos'],Rotation.from_matrix(reference@current.T).as_rotvec())
        obs,_,_,_=env.step(action.numpy());success=bool(env.check_success())
        if success and first is None:first=step
        trace.append(dict(step=step,phase=phase,eef=obs['robot0_eef_pos'].tolist(),gripper=obs['robot0_gripper_qpos'].tolist(),action=action.tolist(),target=target.tolist(),success=success))
        if step%10==0 or success:
            Image.fromarray(obs['agentview_image'][::-1]).save(a.out/f'frame_{step:04}.png');Image.fromarray(obs['robot0_eye_in_hand_image'][::-1]).save(a.out/f'wrist_{step:04}.png')
        if step%20==0:print(step,phase,error.tolist(),flush=True)
        settled=settled+1 if phase==7 and float(error.abs().max())<.02 else 0
        if settled>=20:break
finally:env.close()
result=dict(success=success,first_success_step=first,steps=len(trace),primitive_finished=settled>=20,released=any(t['phase']>=5 for t in trace),
            suite=a.suite,task=a.task,trial=a.trial,instruction=instruction,condition=plan,geometry=geometry,weights=str(a.weights),sha256=hashlib.sha256(a.weights.read_bytes()).hexdigest(),
            parameter_count=sum(p.numel() for p in head.parameters()),capture_path=str(a.capture),split=a.split,sensor_protocol='RGB-D + camera calibration + robot proprioception only')
(a.out/'trace.json').write_text(json.dumps(trace,indent=2)+'\n');(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
