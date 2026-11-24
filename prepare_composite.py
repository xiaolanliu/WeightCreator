"""Make sub-instruction views of the same observed initial sensor packet."""
import argparse
import json
from pathlib import Path
from composite_parser import parse_composite

p=argparse.ArgumentParser();p.add_argument('capture',type=Path);a=p.parse_args();folder=a.capture
instruction=json.loads((folder/'instruction.json').read_text())['instruction']
stages=parse_composite(instruction)
for index,stage in enumerate(stages):
    out=folder/f'stage{index}';out.mkdir()
    for name in ('sensors.npz','rgb.png','wrist_sensors.npz','wrist_rgb.png','episode.json'):
        if (folder/name).exists():(out/name).symlink_to(folder/name)
    (out/'instruction.json').write_text(json.dumps(dict(instruction=stage['instruction']),indent=2)+'\n')
    (out/'parent_instruction.json').write_text(json.dumps(dict(instruction=instruction,stage=index,provenance='Instruction grammar only'),indent=2)+'\n')
(folder/'composition.json').write_text(json.dumps(dict(instruction=instruction,stages=stages),indent=2)+'\n')
