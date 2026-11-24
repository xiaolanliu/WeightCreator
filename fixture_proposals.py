"""Geometry-only handle proposals, followed separately by frozen VLM selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from fixture_geometry import handle_hypotheses

p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args()
sensor=np.load(a.capture/'sensors.npz');candidates=json.loads((a.capture/'candidates.json').read_text())['candidates'];rows=[]
for candidate in candidates:
    extent=np.asarray(candidate['upper'])-candidate['lower']
    if extent[2]<.12 or min(extent[:2])<.08:continue
    hypotheses=handle_hypotheses(sensor,candidate)
    for h in hypotheses:
        if h['count']>=1:rows.append(dict(cabinet_id=candidate['id'],fixture_extent=extent.tolist(),**h))
(a.capture/'fixture_proposals.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
