"""Acquire both existing RGB-D cameras at the same native initialized state."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from capture import make_env,save_capture,sensors

p=argparse.ArgumentParser();p.add_argument('--suite',required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--trial',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--size',type=int,default=1024);p.add_argument('--split',default='development');a=p.parse_args()
env,obs,instruction=make_env(a.suite,a.task,a.trial,size=a.size)
try:
    save_capture(a.out,env,obs,instruction)
    wrist=sensors(env,obs,'robot0_eye_in_hand');np.savez_compressed(a.out/'wrist_sensors.npz',**wrist);Image.fromarray(wrist['rgb']).save(a.out/'wrist_rgb.png')
    (a.out/'episode.json').write_text(json.dumps(dict(suite=a.suite,task=a.task,trial=a.trial,seed=7,image_size=a.size,cameras=['agentview','robot0_eye_in_hand'],split=a.split),indent=2)+'\n')
finally:env.close()
