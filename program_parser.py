"""Instruction-only composition of pick/place clauses; no suite or task IDs."""
import re


def support_mode(text,target):
    if re.search(r'\b(?:on top of|on) (?:the )?'+re.escape(target)+r'$',text):return 'on'
    if re.search(r'\b(?:into|in) (?:the )?'+re.escape(target)+r'$',text):return 'in'
    return 'other'


def single_clause(text):
    relative=re.fullmatch(r'put (?:the )?(.+?) to the (right|left) of (?:the )?(.+)',text)
    if relative:return relative.group(1),relative.group(2)+' of '+relative.group(3)
    if text.startswith('pick up '):
        pattern=r'pick up (?:the )?(.+?) and (?:place|put) it (?:on top of|on|into|in) (?:the )?(.+)'
    else:
        pattern=r'put (?:the )?(.+?) (?:on top of|on|into|in) (?:the )?(.+)'
    match=re.fullmatch(pattern,text)
    if not match:raise ValueError(f'Unsupported primitive: {text}')
    return match.groups()


def parse_program(instruction):
    text=instruction.lower().strip().rstrip('.')
    if text.startswith('put both '):
        match=re.fullmatch(r'put both (.+?) (?:on top of|on|into|in) (?:the )?(.+)',text)
        if not match:raise ValueError(f'Unsupported composition: {text}')
        source,target=match.groups()
        if ' and ' in source:
            sources=[part.removeprefix('the ') for part in source.split(' and ',1)]
        else:
            singular=source.removeprefix('the ').removesuffix('s')
            sources=[singular,singular]
        return [dict(source=s,target=target,skill='pick_place',support_mode=support_mode(text,target)) for s in sources]
    clauses=re.split(r' and (?=put )',text)
    if len(clauses)>2:raise ValueError('Program exceeds the two-primitive authored state capacity')
    program=[]
    for clause in clauses:
        source,target=single_clause(clause)
        if 'both ' in source or ' and ' in target:raise ValueError(f'Unsupported composition: {text}')
        program.append(dict(source=source,target=target,skill='pick_place',support_mode=support_mode(clause,target)))
    return program
