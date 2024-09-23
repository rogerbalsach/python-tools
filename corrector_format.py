#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:50:46 2024

@author: rbalsach
"""
from itertools import count
from random import choice
from string import ascii_lowercase, digits
import re

pattern = re.compile(r'\\(?:eq)?ref\{\S*\}')
begin_pattern = re.compile(r'\\begin\{([^\s\*\}]*)\*?\}')
end_pattern = re.compile(r'\\end\{([^\s\*\}]*)\*?\}')
def string_format(s):
    c = count(1)
    env = []
    punct = ''
    new_s = ''
    prev = ''
    caption = [False, 0]
    append = ''
    for line in s.splitlines():
        next_line = False
        line = line.strip()
        line = line.replace(r'\\', ' ').replace(r'\newline', '')
        if not line: continue
        for match in begin_pattern.finditer(line):
            env.append(match.group(1))
            next_line = True
        for match in end_pattern.finditer(line):
            end = env.pop()
            if match.group(1) != end:
                raise ValueError(line)
            if end in {'equation', 'align'}:
                if prev.startswith('~'):
                    punct = prev[-1]
                new_s += f'{choice(ascii_lowercase)} = {choice(digits)}{punct} '
                punct = ''
            next_line = True
        if next_line: continue
        for match in pattern.finditer(line):
            line = line.replace(match.group(), str(next(c)))
        if not env:
            new_s += line + ' '
        elif env[-1] == 'itemize':
            new_s += line.replace(r'\item', '').strip() + ' '
        elif 'figure' in env[-1]:
            if 'caption' in line:
                caption[0] = True
                caption[1] = line.count('{') - line.count('}')
                line = line.replace(r'\caption{', '')
                append += line.strip() + ' '
            elif caption[1]:
                append += line.strip() + ' '
                caption[1] += line.count('{') - line.count('}')
            if caption == [True, 0]:
                r = append[::-1].find('}')
                append = append[:-r-1].strip() + '\n'
                caption[0] = False
        prev = line

    append = append.replace('\n ', '\n')
    s = new_s.strip() + '\n' + append.strip()
    # s = s.replace('\n', ' ').replace('\t', ' ')
    s = s.replace(r'\\', ' ').replace(r'\newline', '')
    s = s.replace(9*' ', ' ').replace(5*' ', ' ').replace(3*' ', ' ').replace('  ', ' ')
    print(s.strip())
