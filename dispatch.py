"""Skill routing from instruction text alone, never evaluator metadata."""
import re
from program_parser import parse_program
from composite_parser import parse_composite


def route(instruction):
    text=instruction.lower().strip().rstrip('.')
    if text.startswith('turn on ') and ' and put ' in text:
        parse_composite(text)
        return dict(kind='sequence',perception='perception_layers.py')
    if re.fullmatch(r'open the (top|middle|bottom) drawer(?: of the (?:wooden )?cabinet)?',text):
        return dict(kind='fixture',perception='perception.py')
    if text=='turn on the stove':
        return dict(kind='twist',perception='perception_layers.py')
    if re.fullmatch(r'push (?:the )?(.+?) to the front of (?:the )?(.+)',text):
        return dict(kind='push',perception='perception_layers.py')
    program=parse_program(text)
    layered=any(any(word in step['source'] for word in [' between ',' next to ',' on ',' from table center',' in the ']) for step in program)
    return dict(kind='program',perception='perception_layers.py' if layered else 'perception.py')
