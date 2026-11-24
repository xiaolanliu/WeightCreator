"""Native-budget cross-skill evaluator sharing one uninterrupted environment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from capture import make_env
from composite_parser import parse_composite
from grasp_features import source_axes,palm_clearance_features,bowl_clearance_features
from program_expert import ProgramActionExpert
from sequence_expert import SequenceActionExpert
from twist_expert import TwistActionExpert


class TwistPolicy:
    def __init__(self,capture,weights,obs):
        self.plan=json.loads((capture/'condition-twist.json').read_text())
        candidates=json.loads((capture/'candidates.json').read_text())['candidates']
        source=next(c for c in candidates if c['id']==self.plan['source_id'])
        self.geometry=source_axes(np.load(capture/'sensors.npz'),source)
        self.head=TwistActionExpert(weights);self.head.prepare(source['position'],source['upper'])
        self.reference=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        jaw=self.reference@self.head.closing_axis.numpy();axis=self.geometry['minor_axis']
        self.base=float((np.arctan2(axis[1],axis[0])-np.arctan2(jaw[1],jaw[0])+np.pi/2)%np.pi-np.pi/2)
        self.finished=False;self.released=False

    def action(self,obs):
        phase=int(self.head.phase.argmax())
        desired=Rotation.from_euler('z',self.base+float(self.head.phase@self.head.yaw)).as_matrix()@self.reference
        current=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        action,target,error=self.head(obs['robot0_eef_pos'],Rotation.from_matrix(desired@current.T).as_rotvec())
        self.finished=self.head.finished;self.released|=phase>=5
        return action,target,error,phase


class PickPlacePolicy:
    def __init__(self,capture,weights,obs,target_override=None,reference_override=None,target_sensor=None):
        self.plan=json.loads((capture/'condition-program.json').read_text())
        assert self.plan['action_weight_sha256']==hashlib.sha256(weights.read_bytes()).hexdigest()
        assert len(self.plan['steps'])==1
        clause=self.plan['steps'][0]
        assert clause.get('placement_relation','none')=='none'
        candidates={c['id']:c for c in json.loads((capture/'candidates.json').read_text())['candidates']}
        source=candidates[clause['source_id']];target=candidates[clause['target_id']]
        if target_override is not None:
            target=target_override
            self.plan['observed_target_override']=target_override
        sensor=np.load(capture/'sensors.npz')
        self.head=ProgramActionExpert(weights);self.head.start(1);self.head.prepare(source,target,clause['source_kind'])
        self.geometry=source_axes(sensor,source)
        reference=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix() if reference_override is None else reference_override
        yaw=self.head.grasp_yaw(self.geometry,(reference@self.head.closing_axis.numpy())[:2])
        self.desired=Rotation.from_euler('z',yaw).as_matrix()@reference
        self.geometry['desired_yaw']=yaw
        if clause['source_kind']=='bowl' and hasattr(self.head,'bowl_candidates'):
            clearance_sensor=target_sensor if target_sensor is not None and hasattr(self.head,'source_live_clearance') else sensor
            shifts,poses,features,details=bowl_clearance_features(clearance_sensor,source,target,reference,self.head,target_sensor=target_sensor)
            choice,scores=self.head.choose_grasp_offset(features);self.head.prepare(source,target,clause['source_kind'],grasp_shift=shifts[choice]);self.desired=poses[choice]
            self.geometry.update(bowl_candidate=choice,bowl_candidates=details,bowl_scores=scores,chosen_offset=shifts[choice].tolist(),desired_yaw=details[choice]['yaw'])
        elif clause['source_kind']!='bowl' and self.geometry['aspect']>float(self.head.yaw_config[1]):
            closing=self.desired@self.head.closing_axis.numpy()
            shifts,features=palm_clearance_features(sensor,source,self.geometry,closing[:2],self.head.offset_fractions.numpy(),self.head.palm_bounds.numpy())
            choice,scores=self.head.choose_grasp_offset(features)
            self.head.prepare(source,target,clause['source_kind'],grasp_shift=shifts[choice])
            self.geometry.update(chosen_offset=shifts[choice].tolist(),clearance_scores=scores)
        self.finished=False;self.released=False

    def action(self,obs):
        phase=int(self.head.phase.argmax());current=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        action,target,error=self.head(obs['robot0_eef_pos'],Rotation.from_matrix(self.desired@current.T).as_rotvec())
        self.finished=self.head.observe_completion(error,phase);self.released|=phase>=6
        return action,target,error,phase


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--suite',required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--trial',type=int,required=True)
    for name in ('sequence','program','twist'):p.add_argument('--'+name+'-weights',type=Path,required=True)
    p.add_argument('--split',default='development');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    episode=json.loads((a.capture/'episode.json').read_text());assert (episode['suite'],episode['task'],episode['trial'])==(a.suite,a.task,a.trial)
    instruction=json.loads((a.capture/'instruction.json').read_text())['instruction'];stages=parse_composite(instruction)
    assert [s['kind'] for s in stages]==['twist','program']
    sequence=SequenceActionExpert(a.sequence_weights)
    env,obs,actual=make_env(a.suite,a.task,a.trial,size=256);assert actual==instruction
    policies=[]
    def prepare():
        index=sequence.index;kind=stages[index]['kind'];capture=a.capture/f'stage{index}'
        cls=TwistPolicy if kind=='twist' else PickPlacePolicy
        policy=cls(capture,getattr(a,kind+'_weights'),obs);policies.append(policy);return policy
    policy=prepare();trace=[];first=None;success=False;finished=False
    limit={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}[a.suite]
    try:
        for step in range(limit):
            index=sequence.index
            action,target,error,phase=policy.action(obs)
            obs,_,_,_=env.step(action.numpy());success=bool(env.check_success())
            if success and first is None:first=step
            trace.append(dict(step=step,skill=index,kind=stages[index]['kind'],phase=phase,eef=obs['robot0_eef_pos'].tolist(),
                              gripper=obs['robot0_gripper_qpos'].tolist(),action=action.tolist(),target=target.tolist(),success=success))
            if step%10==0 or success:
                Image.fromarray(obs['agentview_image'][::-1]).save(a.out/f'frame_{step:04}.png');Image.fromarray(obs['robot0_eye_in_hand_image'][::-1]).save(a.out/f'wrist_{step:04}.png')
            if step%20==0:print(step,index,stages[index]['kind'],phase,error.tolist(),flush=True)
            if sequence.observe(policy.finished):
                if sequence.index==len(stages):finished=True;break
                policy=prepare()
    finally:env.close()
    bank={name:dict(path=str(getattr(a,name+'_weights')),sha256=hashlib.sha256(getattr(a,name+'_weights').read_bytes()).hexdigest()) for name in ('sequence','program','twist')}
    result=dict(success=success,first_success_step=first,steps=len(trace),native_step_limit=limit,primitive_finished=finished,
                released=len(policies)==len(stages) and all(p.released for p in policies),completed_skills=sequence.index,
                suite=a.suite,task=a.task,trial=a.trial,instruction=instruction,condition=[p.plan for p in policies],geometry=[p.geometry for p in policies],
                weight_bank=bank,sha256=bank['sequence']['sha256'],parameter_count=sum(p.numel() for p in sequence.parameters())+sum(p.numel() for x in policies for p in x.head.parameters()),
                capture_path=str(a.capture),split=a.split,sensor_protocol='Initial calibrated RGB-D from two cameras + robot proprioception + instruction; no object truth or evaluator success enters policy')
    (a.out/'trace.json').write_text(json.dumps(trace,indent=2)+'\n');(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
