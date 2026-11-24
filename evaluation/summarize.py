"""Read-only metrics over the full planned denominator; never reruns episodes."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('protocol',type=Path);a=p.parse_args()
d=json.loads(a.protocol.read_text());suites={s:dict(planned=0,completed=0,native_success=0,released_success=0,stable_success=0,full_success=0,infrastructure_errors=0) for s in d['native_limits']}
for e in d['episodes']:
 row=suites[e['suite']];row['planned']+=1
 f=a.protocol.parent/f"{e['suite']}_task{e['task']}_trial{e['trial']}"/'status.json'
 if not f.exists():continue
 s=json.loads(f.read_text());row['infrastructure_errors']+=s.get('failure_type')=='infrastructure_error'
 if s.get('stage')!='completed':continue
 row['completed']+=1;ok=bool(s['success']);row['native_success']+=ok
 for metric,key in [('released_success','released'),('stable_success','stable_success_last10'),('full_success','primitive_finished')]:row[metric]+=ok and bool(s.get(key))
total={k:sum(r[k] for r in suites.values()) for k in next(iter(suites.values()))}
complete=total['completed']==total['planned'] and not total['infrastructure_errors']
print(json.dumps(dict(complete=complete,by_suite=suites,total=total,final_success_rate=total['native_success']/total['planned'] if complete else None,platform_terminal_status_verified=False),indent=2))
