"""Instruction-only cross-skill decomposition, without evaluator metadata."""
import re


def parse_composite(instruction):
    text=instruction.lower().strip().rstrip('.')
    match=re.fullmatch(r'put the (.+?) in the (top|middle|bottom) drawer of the (?:wooden )?cabinet and close it',text)
    if match:
        source,level=match.groups()
        return [dict(kind='program',instruction=f'put the {source} in the cabinet',placement='open_drawer'),
                dict(kind='fixture_close',instruction=f'close the {level} drawer of the cabinet',level=level)]
    match=re.fullmatch(r'open the (top|middle|bottom) drawer and put the (.+?) inside',text)
    if match:
        level,source=match.groups()
        return [dict(kind='fixture',instruction=f'open the {level} drawer',level=level),
                dict(kind='program',instruction=f'put the {source} in the cabinet',placement='opened_drawer')]
    match=re.fullmatch(r'turn on the (stove) and put the (.+?) on (it|the stove)',text)
    if match:
        appliance,source,_=match.groups()
        return [dict(kind='twist',instruction=f'turn on the {appliance}'),
                dict(kind='program',instruction=f'put the {source} on the {appliance}')]
    raise ValueError('Unsupported cross-skill instruction: '+instruction)
