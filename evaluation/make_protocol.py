"""Create a portable complete LIBERO evaluation manifest."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--simulation-python',required=True);p.add_argument('--model-python',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
assert not (out/'protocol.json').exists(),'Preserve existing evaluation protocols'
limits={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}
bank={'program':'expert_v52','relation':'relation_v7','fixture':'fixture_v1','twist':'twist_v1','push':'push_v2','sequence':'sequence_v0'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=dict(code=str(root),source_sha256={p.name:sha(p) for p in root.glob('*.py')},policies={k:dict(weights=str(root/'weights'/(v+'.pt')),weight_sha256=sha(root/'weights'/(v+'.pt'))) for k,v in bank.items()},simulation_python=a.simulation_python,model_python=a.model_python,camera_size=1024,native_limits=limits,max_episodes_per_producer=2000,episodes=[dict(suite=s,task=t,trial=i) for i in range(50) for t in range(10) for s in limits],scope='WeightCreator v1 full RGB-D rerun; includes development-used states',split='frozen-full-evaluation')
(out/'protocol.json').write_text(json.dumps(d,indent=2)+'\n');print(out/'protocol.json')
