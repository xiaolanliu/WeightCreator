"""Native-budget evaluation of the authored two-primitive recurrent expert."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from capture import make_env,sensors
from grasp_features import source_axes,palm_clearance_features,bowl_clearance_features
from program_expert import ProgramActionExpert
from support_features import support_surface,support_normal
from compartment_features import observed_back_center
from payload_features import carried_center


def main():
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--suite',default='libero_10');p.add_argument('--task',type=int,required=True);p.add_argument('--trial',type=int,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--split',default='development')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    episode=json.loads((a.capture/'episode.json').read_text());assert (episode['suite'],episode['task'],episode['trial'])==(a.suite,a.task,a.trial)
    plan=json.loads((a.capture/'condition-program.json').read_text());steps=plan['steps']
    digest=hashlib.sha256(a.weights.read_bytes()).hexdigest();assert plan['action_weight_sha256']==digest
    record=json.loads((a.capture/'candidates.json').read_text());candidates={c['id']:c for c in record['candidates']}
    sensor=np.load(a.capture/'sensors.npz');head=ProgramActionExpert(a.weights);head.start(len(steps))
    env,obs,instruction=make_env(a.suite,a.task,a.trial,size=256)
    assert instruction==json.loads((a.capture/'instruction.json').read_text())['instruction']
    reference=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
    geometries=[]

    def prepare():
        clause=steps[head.program_index];source=candidates[clause['source_id']];target=copy.deepcopy(candidates[clause['target_id']])
        container_bounds=(np.asarray(target['lower']).copy(),np.asarray(target['upper']).copy())
        shared=len(steps)>1 and all(s['target_id']==steps[0]['target_id'] and s.get('placement_relation','none')=='none' for s in steps)
        extent=np.asarray(target['upper'])-target['lower']
        slot=head.placement_offset(extent,shared).numpy()
        relation=clause.get('placement_relation','none')
        if relation in ('offset_left','offset_right','compartment_back'):
            assert hasattr(head,'placement_relation'), 'Checkpoint lacks authored relative placement geometry'
            settings=head.placement_relation.numpy();source_extent=np.asarray(source['upper'])-source['lower']
            if relation.startswith('offset_'):
                axis=sensor['camera_rotation'][:2,0].copy();axis/=max(np.linalg.norm(axis),1e-6)
                axis*=1 if relation=='offset_right' else -1
                distance=settings[0]*abs(axis)@extent[:2]+settings[1]*abs(axis)@source_extent[:2]+settings[2]
                slot[:2]+=axis*distance
                target['upper'][2]=record['table_height']
            else:
                view=sensor['camera_position'][:2]-np.asarray(target['position'])[:2];axis=np.zeros(2)
                dimension=int(np.argmax(abs(view)));axis[dimension]=-np.sign(view[dimension])
                slot[:2]+=axis*(settings[3]*abs(axis)@extent[:2])
        target['position']=(np.asarray(target['position'])+slot).tolist()
        support_details=None
        surface_details=None
        if relation=='none' and clause.get('support_mode')=='on' and hasattr(head,'visible_surface'):
            settings=head.visible_surface.numpy();lo,hi=container_bounds
            points=support_surface(sensor,np.asarray(target['position'])[:2],np.repeat(settings[0],2),lo[2]-.005,hi[2]+settings[2],settings[4],settings[5])
            surface_details=dict(points=len(points),original_upper=float(target['upper'][2]),applied=False)
            if len(points)>=settings[3]:
                target['upper'][2]=float(np.quantile(points[:,2],settings[1]))
                surface_details.update(applied=True,observed_support_height=target['upper'][2])
        if relation=='compartment_back' and hasattr(head,'compartment_support'):
            support=head.compartment_support.numpy();half_window=extent[:2]*support[3];half_window[dimension]=extent[dimension]*support[2]
            world=sensor['world'].reshape(-1,3);lo,hi=container_bounds
            inside=np.all(abs(world[:,:2]-np.asarray(target['position'])[:2])<half_window,axis=1)&(world[:,2]>=lo[2]-.005)&(world[:,2]<=hi[2])
            floor=world[inside,2];support_details=dict(points=int(len(floor)),half_window=half_window.tolist(),original_upper=float(hi[2]))
            if len(floor)>=20:
                target['upper'][2]=float(np.quantile(floor,float(support[0]))+support[1])
                support_details['observed_support_height']=target['upper'][2]
        head.prepare(source,target,clause['source_kind'])
        geometry=source_axes(sensor,source);closing=reference@head.closing_axis.numpy()
        yaw=head.grasp_yaw(geometry,closing[:2]);desired=Rotation.from_euler('z',yaw).as_matrix()@reference
        geometry.update(desired_yaw=yaw,target_slot=slot.tolist(),placement_relation=relation,compartment_support=support_details,
                        visible_surface=surface_details,
                        target_center=target['position'],container_bounds=[x.tolist() for x in container_bounds])
        if clause['source_kind']=='bowl' and hasattr(head,'bowl_candidates'):
            shifts,poses,features,details=bowl_clearance_features(sensor,source,target,reference,head,target_sensor=sensors(env,obs) if head.program_index>0 else sensor)
            choice,scores=head.choose_grasp_offset(features);head.prepare(source,target,clause['source_kind'],grasp_shift=shifts[choice]);desired=poses[choice]
            geometry.update(bowl_candidate=choice,bowl_candidates=details,bowl_scores=scores,chosen_offset=shifts[choice].tolist(),desired_yaw=details[choice]['yaw'])
        elif clause['source_kind']!='bowl' and geometry['aspect']>float(head.yaw_config[1]):
            closing=desired@head.closing_axis.numpy()
            shifts,features=palm_clearance_features(sensor,source,geometry,closing[:2],head.offset_fractions.numpy(),head.palm_bounds.numpy())
            choice,scores=head.choose_grasp_offset(features)
            head.prepare(source,target,clause['source_kind'],grasp_shift=shifts[choice])
            geometry.update(chosen_offset=shifts[choice].tolist(),clearance_features=features.tolist(),clearance_scores=scores)
        placement=desired
        if relation in ('offset_left','offset_right','compartment_back') and hasattr(head,'destination_yaws'):
            # A later placement sees objects placed by earlier primitives through RGB-D.
            observed=sensor if relation=='compartment_back' and head.program_index==0 else sensors(env,obs)
            world=observed['world'].reshape(-1,3);points=world
            base=head.condition[3:6].numpy().copy();delta=head.condition[:2].numpy()-np.asarray(source['position'])[:2]
            bounds=head.palm_bounds.numpy();height=base[2]+float(head.target_weight[5,2,6])
            points=points[(np.linalg.norm(points[:,:2]-base[:2],axis=1)<.3)&(points[:,2]>height+bounds[2])&(points[:,2]<height+bounds[3])]
            points=np.unique(np.floor(points/.004).astype(int),axis=0).astype(float)*.004+.002
            compartment=None
            if relation=='compartment_back' and hasattr(head,'compartment_clearance'):
                settings=head.compartment_clearance.numpy();lo,hi=container_bounds
                inside=np.all((world[:,:2]>=lo[:2])&(world[:,:2]<=hi[:2]),axis=1)&(world[:,2]>lo[2]+settings[0])&(world[:,2]<hi[2]+.005)
                compartment=np.unique(np.floor(world[inside]/.004).astype(int),axis=0).astype(float)*.004+.002
            scores=[];poses=[];targets=[];object_collisions=[]
            angles=head.destination_yaws.tolist()
            aligned_axes=None
            if relation=='compartment_back' and hasattr(head,'compartment_alignment'):
                minor=np.asarray(geometry['minor_axis']);major=np.array([-minor[1],minor[0]])
                target_major=np.eye(2)[int(np.argmax(np.asarray(container_bounds[1])[:2]-np.asarray(container_bounds[0])[:2]))]
                alignment=(np.arctan2(target_major[1],target_major[0])-np.arctan2(major[1],major[0])+np.pi/2)%np.pi-np.pi/2
                angles=[float(head.compartment_alignment)*float(alignment)]
                aligned_axes=(minor,major)
                geometry['compartment_axis_alignment']=dict(source_major=major.tolist(),target_major=target_major.tolist(),yaw=angles[0])
            pose_candidates=[(angle,0.) for angle in angles]
            if relation.startswith('offset_') and hasattr(head,'relative_clearance_offsets'):
                pose_candidates=[(angle,extra) for extra in head.relative_clearance_offsets.tolist() for angle in angles]
            for angle,extra in pose_candidates:
                turn=Rotation.from_euler('z',angle).as_matrix();pose=turn@desired;target_xy=base[:2]+turn[:2,:2]@delta-delta
                if relation.startswith('offset_'):target_xy=target_xy+axis*extra
                jaw=(pose@head.closing_axis.numpy())[:2];jaw/=max(np.linalg.norm(jaw),1e-6);tangent=np.array([-jaw[1],jaw[0]])
                distance=points[:,:2]-target_xy;count=np.count_nonzero((abs(distance@jaw)<bounds[0])&(abs(distance@tangent)<bounds[1]))
                object_count=0
                if compartment is not None:
                    relative=(compartment[:,:2]-np.asarray(target['position'])[:2])@turn[:2,:2]
                    if aligned_axes is not None and 'minor_width' in geometry:
                        relative=relative@np.column_stack(aligned_axes)
                        half=np.array([geometry['minor_width'],geometry['major_width']])*.5*settings[1]
                    else:half=(np.asarray(source['upper'])[:2]-source['lower'][:2])*.5*settings[1]
                    object_count=np.count_nonzero(np.all(abs(relative)<half,axis=1))
                scores.append(float(head.clearance_score[0])*count+float(head.clearance_score[1])*extra-(float(settings[2])*object_count if compartment is not None else 0.));poses.append(pose);targets.append(target_xy);object_collisions.append(int(object_count))
            choice=int(np.argmax(scores));placement=poses[choice];head.condition[3:5]=head.condition.new_tensor(targets[choice])
            chosen_angle,chosen_extra=pose_candidates[choice]
            if relation.startswith('offset_'):
                geometry['target_center'][:2]=(np.asarray(geometry['target_center'])[:2]+axis*chosen_extra).tolist()
            geometry.update(destination_clearance_scores=scores,destination_object_collisions=object_collisions,destination_yaw=float(chosen_angle),destination_outward_offset=float(chosen_extra),destination_grasp_xy=targets[choice].tolist(),destination_observation_resolution=observed['rgb'].shape[:2])
        geometry['observed_grasp_to_base']=float(head.condition[2])-float(source['lower'][2])
        source_extent=np.asarray(source['upper'])-source['lower']
        if hasattr(head,'surface_align') and surface_details and surface_details['applied'] and source_extent[2]>float(head.tall_grasp[0])*max(source_extent[:2]):
            alignment=head.surface_align.numpy();settings=head.visible_surface.numpy()
            normal,count=support_normal(sensor,np.asarray(target['position'])[:2],np.repeat(settings[0],2),surface_details['observed_support_height'],alignment[1],alignment[2],settings[5])
            if normal is not None and normal[2]<alignment[0]:
                uphill=np.array([0.,0.,1.])-normal*normal[2];uphill/=np.linalg.norm(uphill)
                jaw=np.cross(normal,uphill);jaw/=np.linalg.norm(jaw)
                if jaw@(desired@head.closing_axis.numpy())<0:jaw=-jaw
                z=-uphill;placement=np.column_stack([np.cross(jaw,z),jaw,z])
                support_point=np.array([*target['position'][:2],surface_details['observed_support_height']])
                radius=.5*max(source_extent[:2])+float(alignment[3])
                grip_point=support_point+uphill*geometry['observed_grasp_to_base']+normal*radius
                head.condition[3:6]=head.condition.new_tensor(grip_point)-head.target_weight[5,:,6]
                geometry['surface_alignment']=dict(points=count,normal=normal.tolist(),uphill=uphill.tolist(),grip_target=grip_point.tolist(),radius_clearance=radius,rotation=placement.tolist())
        geometries.append(geometry)
        return desired,placement

    grasp_pose,placement_pose=prepare();trace=[];first=None;success=False;released=set();finished=False;support_refreshed=set();contact_history={};payload_refreshed=set()
    limit={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}[a.suite]
    try:
        for index in range(limit):
            part=head.program_index;phase=int(head.phase.argmax())
            geometry=geometries[part]
            if phase==4 and part not in payload_refreshed and hasattr(head,'payload_tracking'):
                payload_refreshed.add(part);settings=head.payload_tracking.numpy()
                source=candidates[steps[part]['source_id']];extent=np.asarray(source['upper'])-source['lower']
                gap=float(np.mean(abs(obs['robot0_gripper_qpos']))*2)
                eligible=(steps[part]['source_kind']!='bowl' and gap<settings[0]*min(extent[:2]) and extent[2]>settings[1]*max(extent[:2]) and 'surface_alignment' not in geometry)
                detail=dict(eligible=bool(eligible),grasp_gap=gap,source_extent=extent.tolist(),applied=False)
                if eligible:
                    observations=[]
                    for camera in ('agentview','robot0_eye_in_hand'):
                        current=sensors(env,obs,camera);observations.append(current)
                        np.savez_compressed(a.out/f'payload_{part}_{camera}.npz',**current)
                        Image.fromarray(current['rgb']).save(a.out/f'payload_{part}_{camera}.png')
                    center,measured=carried_center(observations,obs['robot0_eef_pos'],geometry['observed_grasp_to_base'],extent,settings)
                    detail.update(measured)
                    if center is not None:
                        offset=np.r_[np.asarray(obs['robot0_eef_pos'])[:2]-center,0.]
                        rotation=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
                        offset=(placement_pose@rotation.T@offset)[:2]
                        if np.max(abs(offset))<settings[7]:
                            head.condition[3:5]=head.condition.new_tensor(np.asarray(geometry['target_center'])[:2]+offset)
                            detail.update(applied=True,grip_offset=offset.tolist(),eef_target=head.condition[3:5].tolist(),step=index)
                geometry['payload_centering']=detail
            if phase==4 and geometry['placement_relation']=='compartment_back' and hasattr(head,'compartment_contact') and head.release_aperture is None:
                head.release_aperture=head.condition.new_tensor(float(np.mean(abs(obs['robot0_gripper_qpos']))))+head.compartment_release[0]
            if phase==5 and part not in support_refreshed and geometry['placement_relation']=='compartment_back' and hasattr(head,'compartment_live_support'):
                settings=head.compartment_live_support.numpy();support=head.compartment_support.numpy()
                details=geometry['compartment_support'];lo,hi=np.asarray(geometry['container_bounds'])
                observed_points=[];views={}
                carried_points=[];wall_observations=[sensor];old_target_center=np.asarray(geometry['target_center']).copy()
                for camera in ('agentview','robot0_eye_in_hand'):
                    current=sensors(env,obs,camera)
                    wall_observations.append(current)
                    np.savez_compressed(a.out/f'support_{part}_{camera}.npz',**current)
                    Image.fromarray(current['rgb']).save(a.out/f'support_{part}_{camera}.png')
                    points=support_surface(current,np.asarray(geometry['target_center'])[:2],np.asarray(details['half_window']),lo[2]-.005,hi[2],settings[0],settings[1])
                    views[camera]=dict(points=len(points),minimum_z=float(points[:,2].min()) if len(points) else None)
                    observed_points.extend(points[:,2].tolist())
                    if hasattr(head,'compartment_tracking'):
                        tracking=head.compartment_tracking.numpy();world=current['world'].reshape(-1,3);eef=np.asarray(obs['robot0_eef_pos'])
                        visible=np.linalg.norm(world[:,:2]-eef[:2],axis=1)<tracking[2]
                        visible&=(world[:,2]>max(eef[2]-geometry['observed_grasp_to_base']-tracking[0],hi[2]+tracking[0]))&(world[:,2]<eef[2]-tracking[1])
                        carried_points.extend(world[visible].tolist())
                if hasattr(head,'compartment_wall_geometry'):
                    refined,wall_details=observed_back_center(wall_observations,lo,hi,geometry['target_center'],geometry['target_slot'],head.compartment_wall_geometry.numpy())
                    geometry['target_center']=refined.tolist();geometry['observed_compartment_walls']=wall_details
                update=dict(step=index,views=views,points=len(observed_points),applied=False)
                if len(observed_points)>=settings[2]:
                    height=float(np.quantile(observed_points,float(support[0]))+support[1])
                    old=details.get('observed_support_height',float(hi[2]))
                    if height<old:
                        head.condition[5]+=height-old
                        update.update(applied=True,old_height=old,new_height=height)
                geometry['live_support']=update;support_refreshed.add(part)
                if hasattr(head,'compartment_release_height'):
                    minimum=hi[2]+float(head.compartment_release_height)
                    current=float(head.condition[5]+head.target_weight[5,2,6])
                    head.condition[5]+=max(0.,minimum-current)
                    geometry['rim_clear_release']=dict(rim=float(hi[2]),minimum_grip_height=minimum,previous_grip_height=current)
                if hasattr(head,'compartment_tracking'):
                    tracking=head.compartment_tracking.numpy();centering=dict(points=len(carried_points),applied=False)
                    if len(carried_points)>=tracking[4]:
                        cloud=np.asarray(carried_points);center=(np.quantile(cloud[:,:2],.05,axis=0)+np.quantile(cloud[:,:2],.95,axis=0))/2
                        correction=np.clip(np.asarray(geometry['target_center'])[:2]-center,-tracking[3],tracking[3])
                        if hasattr(head,'compartment_absolute_center'):
                            anchor=np.asarray(obs['robot0_eef_pos'])[:2]
                            head.condition[3:5]=head.condition.new_tensor(anchor+correction)
                            centering['eef_anchor']=anchor.tolist()
                        else:
                            head.condition[3:5]+=head.condition.new_tensor(correction)
                        centering.update(applied=True,observed_center=center.tolist(),correction=correction.tolist(),eef_target=head.condition[3:5].tolist())
                    elif hasattr(head,'compartment_wall_geometry'):
                        head.condition[3:5]+=head.condition.new_tensor(np.asarray(geometry['target_center'])[:2]-old_target_center[:2])
                    geometry['live_centering']=centering
            desired=grasp_pose if phase<4 else placement_pose
            rotation=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
            action,target,error=head(obs['robot0_eef_pos'],Rotation.from_matrix(desired@rotation.T).as_rotvec())
            if phase==5 and geometry['placement_relation']=='compartment_back' and hasattr(head,'compartment_contact'):
                contact=head.compartment_contact.numpy();position=np.asarray(obs['robot0_eef_pos'])
                previous,age=contact_history.get(part,(None,0))
                age=age+1 if previous is not None and np.max(abs(position-previous))<contact[0] else 0
                contact_history[part]=(position.copy(),age)
                if age>=contact[1] and float(error.abs().max())<contact[2] and float(error[:2].abs().max())<contact[3] and int(head.phase.argmax())==5:
                    head.condition[3:6]=head.condition.new_tensor(position)-head.target_weight[5,:,6]
                    head.phase.zero_();head.phase[6]=1;head.age.zero_()
                    geometry['contact_release']=dict(step=index,position=position.tolist(),stationary_steps=age,proprioceptive_error=error.tolist())
            if phase>=6 and geometry['placement_relation']=='compartment_back' and hasattr(head,'compartment_release'):
                action[-1]=head.compartment_grip(obs['robot0_gripper_qpos'],phase,float(obs['robot0_eef_pos'][2]),geometry['container_bounds'][1][2])
            obs,_,_,_=env.step(action.numpy())
            success=bool(env.check_success())
            if success and first is None:first=index
            if phase>=6:released.add(part)
            trace.append(dict(step=index,program=part,phase=phase,eef=obs['robot0_eef_pos'].tolist(),
                              gripper=obs['robot0_gripper_qpos'].tolist(),target=target.tolist(),action=action.tolist(),success=success))
            if index%10==0 or success:
                Image.fromarray(obs['agentview_image'][::-1]).save(a.out/f'frame_{index:04}.png')
                Image.fromarray(obs['robot0_eye_in_hand_image'][::-1]).save(a.out/f'wrist_{index:04}.png')
            if index%20==0:print(index,'primitive',part,head.PHASES[phase],error.tolist(),flush=True)
            # Only proprioceptive error and authored state enter policy transitions.
            if head.observe_completion(error,phase):
                if head.program_index==len(steps):finished=True;break
                grasp_pose,placement_pose=prepare()
    finally:env.close()
    result=dict(success=success,first_success_step=first,steps=len(trace),native_step_limit=limit,
                program_finished=finished,primitive_finished=finished,completed_primitives=head.program_index,
                released=len(released)==len(steps),suite=a.suite,task=a.task,trial=a.trial,instruction=instruction,
                condition=plan,weights=str(a.weights),sha256=digest,parameter_count=sum(p.numel() for p in head.parameters()),
                capture_path=str(a.capture.resolve()),perception_method=record.get('method'),grasp_geometry=geometries,
                sensor_protocol='RGB-D + calibrated cameras + proprioception; initial proposals and phase-boundary visual observations, no object truth or evaluator success input',split=a.split)
    (a.out/'trace.json').write_text(json.dumps(trace,indent=2)+'\n');(a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
