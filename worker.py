"""Frozen all-suite evaluator; skill routing receives only the observation instruction."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    protocol_path,worker=map(Path,sys.argv[1:]);p=json.loads(protocol_path.read_text());root=protocol_path.parent;code=Path(p['code'])
    worker.mkdir(parents=True,exist_ok=True)
    for name,digest in p['source_sha256'].items():assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,name
    for policy in p['policies'].values():assert hashlib.sha256(Path(policy['weights']).read_bytes()).hexdigest()==policy['weight_sha256']
    sys.path.insert(0,str(code));from dispatch import route
    owner=dict(host=socket.gethostname(),pid=os.getpid(),worker=str(worker),started_at=time.time(),executor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(worker/'started.json',owner);completed=[];infrastructure=[]
    for episode in p['episodes']:
        if len(completed)>=p.get('max_episodes_per_producer',6):break
        name=f"{episode['suite']}_task{episode['task']}_trial{episode['trial']}";run=root/name;run.mkdir(exist_ok=True)
        if (run/'status.json').exists():continue
        try:(run/'claim').mkdir()
        except FileExistsError:continue
        save(run/'claim/owner.json',owner);capture=run/'capture'
        common=['--suite',episode['suite'],'--task',str(episode['task']),'--trial',str(episode['trial'])]
        status=dict(episode=episode,success=False,native_final_success=False,split='frozen-full-evaluation',producer=str(worker),started_at=time.time())

        def execute(stage,command):
            status['stage']=stage;save(run/'progress.json',status)
            with (run/f'{stage}.log').open('w') as log:response=subprocess.run(command,stdout=log,stderr=log,timeout=1200)
            if response.returncode:
                detail=(run/f'{stage}.log').read_text(errors='replace')[-3500:]
                if stage.startswith('grounding') and any(t in detail for t in ['Unsupported','Fixture semantics uncertain','No visible','max() arg is an empty sequence']):
                    raise ValueError(detail)
                raise RuntimeError(f'{stage} exited{response.returncode}: {detail}')

        try:
            execute('capture',[p['simulation_python'],str(code/'capture_views.py'),*common,'--size',str(p['camera_size']),'--out',str(capture),'--split','frozen-full-evaluation'])
            # No suite, task index, initialization index or native goal is passed here.
            instruction=json.loads((capture/'instruction.json').read_text())['instruction'];choice=route(instruction)
            status['route']=choice;kind=choice['kind'];policy=p['policies'][kind]
            if kind=='sequence':
                execute('prepare_sequence',[sys.executable,str(code/'prepare_composite.py'),str(capture)])
                execute('perception_twist',[p['simulation_python'],str(code/'perception_layers.py'),str(capture/'stage0')])
                execute('grounding_twist',[p['model_python'],str(code/'semantic_twist.py'),str(capture/'stage0'),'--weights',p['policies']['twist']['weights']])
                execute('perception_pick',[p['simulation_python'],str(code/'perception.py'),str(capture/'stage1')])
                execute('views_pick',[p['simulation_python'],str(code/'multiview_crops.py'),str(capture/'stage1')])
                grounding=[p['model_python'],str(code/'semantic_joint.py'),str(capture/'stage1'),'--weights',p['policies']['program']['weights'],'--relations',p['policies']['relation']['weights']]
            else:
                execute('perception',[p['simulation_python'],str(code/choice['perception']),str(capture)])
            if kind=='fixture':
                execute('handles',[p['simulation_python'],str(code/'fixture_proposals.py'),str(capture)])
                grounding=[p['model_python'],str(code/'semantic_fixture.py'),str(capture)]
            elif kind=='program':
                execute('views',[p['simulation_python'],str(code/'multiview_crops.py'),str(capture)])
                grounding=[p['model_python'],str(code/'semantic_joint.py'),str(capture),'--weights',policy['weights'],'--relations',p['policies']['relation']['weights']]
            elif kind!='sequence':
                grounding=[p['model_python'],str(code/f'semantic_{kind}.py'),str(capture)]
                if kind=='twist':grounding+=['--weights',policy['weights']]
            execute('grounding',grounding)
            runtime_split='frozen-heldout' if kind=='fixture' else 'frozen-full-evaluation'
            status['runtime_split']=runtime_split
            rollout=[p['simulation_python'],str(code/f'rollout_{kind}.py'),*common,'--capture',str(capture),'--out',str(run/'rollout'),'--split',runtime_split]
            if kind=='sequence':
                for bank in ('sequence','program','twist'):rollout+=['--'+bank+'-weights',p['policies'][bank]['weights']]
            else:rollout+=['--weights',policy['weights']]
            execute('rollout',rollout)
            result=json.loads((run/'rollout/result.json').read_text());trace=json.loads((run/'rollout/trace.json').read_text())
            assert result['sha256']==policy['weight_sha256']
            assert len(trace)==result['steps']<=p['native_limits'][episode['suite']]
            assert (result['suite'],result['task'],result['trial'])==(episode['suite'],episode['task'],episode['trial'])
            assert bool(trace[-1]['success'])==bool(result['success']) and result['split']==runtime_split
            stable=len(trace)>=10 and all(t['success'] for t in trace[-10:])
            status.update(stage='completed',success=bool(result['success']),native_final_success=result['success'],released=result['released'],
                          stable_success_last10=stable,primitive_finished=result['primitive_finished'],steps=result['steps'],
                          result_sha256=hashlib.sha256((run/'rollout/result.json').read_bytes()).hexdigest())
        except ValueError as error:
            status.update(stage='completed',failure_type='unsupported_or_selection_failure',error=str(error))
        except Exception as error:
            status.update(failure_type='infrastructure_error',error=repr(error));infrastructure.append(name)
        status['finished_at']=time.time();save(run/'status.json',status);completed.append(name)
        print(name,status['stage'],status['success'],flush=True)
        if infrastructure:break
    save(worker/'result.json',dict(completed=completed,infrastructure_errors=infrastructure,finished_at=time.time(),**owner))
    return bool(infrastructure)


if __name__=='__main__':sys.exit(main())
