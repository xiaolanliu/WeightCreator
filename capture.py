"""Native LIBERO observation acquisition; no policy implementation is imported."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def make_env(suite_name, task_id, trial, size=384):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                            camera_heights=size, camera_widths=size,
                            camera_depths=True, hard_reset=False)
    env.seed(7)
    env.reset()
    # Existing trusted simulator initialization assets, never supplied to policy.
    original = torch.load
    torch.load = lambda *a, **kw: original(*a, **dict(kw, weights_only=False))
    try:
        states = suite.get_task_init_states(task_id)
    finally:
        torch.load = original
    obs = env.set_init_state(states[trial])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return env, obs, task.language


def sensors(env, obs, camera='agentview'):
    rgb = np.ascontiguousarray(obs[camera + '_image'][::-1])
    depth_buffer = obs[camera + '_depth'][::-1].squeeze()
    near = env.sim.model.vis.map.znear * env.sim.model.stat.extent
    far = env.sim.model.vis.map.zfar * env.sim.model.stat.extent
    depth = near / (1 - depth_buffer * (1 - near / far))
    idx = env.sim.model.camera_name2id(camera)
    rotation = env.sim.data.cam_xmat[idx].reshape(3, 3).copy()
    position = env.sim.data.cam_xpos[idx].copy()
    h, w = depth.shape
    focal = h / (2 * np.tan(np.deg2rad(env.sim.model.cam_fovy[idx]) / 2))
    v, u = np.mgrid[:h, :w]
    local = np.stack([(u-(w-1)/2)*depth/focal,
                      -(v-(h-1)/2)*depth/focal, -depth], axis=-1)
    world = local @ rotation.T + position
    return dict(rgb=rgb, depth=depth, world=world, camera_rotation=rotation,
                camera_position=position, focal=np.array(focal),
                eef_pos=obs['robot0_eef_pos'], eef_quat=obs['robot0_eef_quat'],
                gripper_qpos=obs['robot0_gripper_qpos'])


def save_capture(out, env, obs, instruction):
    out.mkdir(parents=True, exist_ok=True)
    sensor = sensors(env, obs)
    np.savez_compressed(out/'sensors.npz', **sensor)
    Image.fromarray(sensor['rgb']).save(out/'rgb.png')
    (out/'instruction.json').write_text(json.dumps({'instruction':instruction}, indent=2)+'\n')
    return sensor


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--suite',default='libero_goal')
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--trial',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--size',type=int,default=384)
    args=p.parse_args()
    env,obs,instruction=make_env(args.suite,args.task,args.trial,size=args.size)
    try:
        save_capture(args.out,env,obs,instruction)
        (args.out/'episode.json').write_text(json.dumps(dict(suite=args.suite,task=args.task,trial=args.trial,seed=7,image_size=args.size),indent=2)+'\n')
        print(instruction, obs['robot0_eef_pos'], flush=True)
    finally:
        env.close()
