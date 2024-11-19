#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:42:48 2023

@author: Roger Balsach
"""

from collections.abc import Iterable
import pathlib
import functools
import re
import sys
from typing import Optional, Union

import numpy as np

separate_list = [r'\\quad', r'\\qquad']
arrow_list = [r'\\to', r'\\xrightarrow(\[.*?\])?\{.*?\}', r'\\Longrightarrow']
equal_list = ['=', r'\\equiv', r'\\cong', r'\\neq']
# A backslash is allowed after these characters without space
backslash_prefactor = [r'\(', r'\[', r'\{', r'\s', r'\$', r'\^', r'\\', r'\-',
                       r'\|', '_', '%']
context_list = [r'\\label', r'\\text']
def_list = [r'\def', r'\newcommand', r'\renewcommand']

### Define regex patterns:
pattern_separate = re.compile('|'.join(separate_list))
pattern_arrow = re.compile('|'.join(arrow_list))
pattern_equal = re.compile('|'.join(equal_list))
pattern_backslash = re.compile(
    rf'[^{"".join(backslash_prefactor)}]\\(?!right)'
)
pattern_context = re.compile(rf'({"|".join(context_list)})$')


ParenthesisType = dict[
    tuple[
        Union[
            tuple[Optional[int], int],
            tuple[int, Optional[int]]
        ],
        str
    ],
    'ParenthesisType'
]


class Parenthesis():
    OPEN_PARENTHESIS = '([{'
    CLOSE_PARENTHESIS = ')]}'

    def __init__(self) -> None:
        self.current_struct: ParenthesisType = {}
        self.parenthesis_structure: ParenthesisType = {}

    def add_open_brace(self, idx: int, char: str) -> None:
        self.levels.append((idx, char))
        self.current_struct[((idx, None), char)] = {}
        self.current_struct = self.current_struct[((idx, None), char)]

    @classmethod
    def get_match(cls, char: str) -> str:
        if char in cls.OPEN_PARENTHESIS:
            return cls.CLOSE_PARENTHESIS[cls.OPEN_PARENTHESIS.index(char)]
        elif char in cls.CLOSE_PARENTHESIS:
            return cls.OPEN_PARENTHESIS[cls.CLOSE_PARENTHESIS.index(char)]
        elif char == '$':
            return char
        raise ValueError()

    @classmethod
    def match(cls, first: str, second: str) -> bool:
        return first == cls.get_match(second)

    def process_end_equation(self, char: str, idx: int, start: int) -> bool:
        if char != '$':
            raise ValueError(
                f'Parenthesis mismatch in line: {self.line}'
            )
        elif self.is_escaped(idx):
            return False
        self.in_equation = False
        return True

    def process_not_match(self, idx: int, char: str, start: int
                          ) -> tuple[int, str]:
        if char == ')':
            # Assume that ) is not part of a parenthesis
            return start, ''
        # Check its not a phantom context
        while self.levels:
            start, schar = self.levels.pop()
            if schar == '{' and not self.is_escaped(start):
                break
        else:
            raise Exception(
                f'Parenthesis not well written: {self.line}'
            )
        if char == '}':
            return start, schar
        self.levels.append((start, schar))
        return start, ''

    def process_unmatched(self, unmatched_parenthesis: str, char: str) -> bool:
        schar = unmatched_parenthesis[-1]
        if not self.match(schar, char):
            if char == ')':
                return True
            raise Exception(
                f'Parenthesis not well written in line:\n'
                f'{self.line}\n'
                f'Expected: {self.get_match(schar)}, '
                f'Found: {char}'
            )
        if char == '$':
            self.in_equation = False
        return False

    def parse(self, line: str, unmatched_parenthesis: str) -> ParenthesisType:
        self.line = line
        self.in_equation = '$' in unmatched_parenthesis
        self.parenthesis_structure = {}
        self.levels: list[tuple[int, str]] = []
        self.current_struct = self.parenthesis_structure

        for idx, char in enumerate(line):
            if char in self.OPEN_PARENTHESIS:
                self.add_open_brace(idx, char)

            elif char == '$' and not self.in_equation:
                if self.is_escaped(idx):
                    continue
                self.add_open_brace(idx, '$')
                self.in_equation = True

            elif char in self.CLOSE_PARENTHESIS + '$':
                if self.levels:
                    start, schar = self.levels.pop()
                    if schar == '$':
                        valid = self.process_end_equation(char, idx, start)
                        if not valid:
                            self.levels.append((start, '$'))
                            continue
                    elif not self.match(char, schar):
                        self.levels.append((start, schar))
                        start, schar = self.process_not_match(idx, char, start)
                        if not schar:
                            continue
                    elif char == '}':
                        if self.is_escaped(idx):
                            idx -= 1
                else:
                    if unmatched_parenthesis:
                        escaped = self.process_unmatched(unmatched_parenthesis,
                                                         char)
                        if escaped:
                            continue
                        unmatched_parenthesis = unmatched_parenthesis[:-1]
                    elif char == ')':
                        continue
                    elif char == '}':
                        if self.is_escaped(idx):
                            idx -= 1
                    schar = self.get_match(char)
                    self.parenthesis_structure = {
                        ((None, idx), schar): self.parenthesis_structure
                    }
                    self.current_struct = self.parenthesis_structure
                    continue
                self.current_struct = self.update_structure(start, schar, idx)
        while self.levels:
            start, schar = self.levels.pop()
            self.current_struct = self.update_structure(start, schar, None)

        return self.parenthesis_structure

    def is_escaped(self, idx: int) -> bool:
        return idx > 0 and self.line[idx - 1] == '\\'

    def update_structure(self, start: int, schar: str, idx: Optional[int]
                         ) -> ParenthesisType:
        parent_structure = self.parenthesis_structure
        for _idx, char in self.levels:
            parent_structure = parent_structure[(_idx, None), char]
        parent_structure.pop(((start, None), schar))
        parent_structure[((start, idx), schar)] = self.current_struct
        return parent_structure


# TODO: Add CLI interface
# TODO: Implement read from file properly
class TeXFormatter:
    def __init__(self, content: Union[str, list[str]]) -> None:
        if isinstance(content, str):
            content = content.splitlines(keepends=True)
        self.init_string = content.copy()
        self.reset_context()
        self.multline_parenthesis = ''
        self.indent = ''
        format_content = self._format_spaces(content)
        self.formatted_lines = self.format_tex(format_content, first=True)

    @property
    def context(self) -> str:
        return self._context[-1]

    def update_context(self, line: str) -> None:
        if any(x in line for x in def_list):
            return
        # if self.context == 'text' and '$' in self.multline_parenthesis:
        #     self._context.append('equation')
        if r'\begin' in line:
            if 'equation' in line or 'align' in line or 'eqnarray' in line:
                self._context.append('equation')
            elif 'document' in line or 'figure' in line:
                self._context.append('text')
            else:
                from warnings import warn
                warn(f'unknown environment: {line}')
                self._context.append(self._context[-1])
        elif r'\beq' in line:
            self._context.append('equation')
        if r'\end' in line:
            self._context.pop()
        elif r'\eeq' in line:
            self._context.pop()

    def reset_context(self) -> None:
        self._context = ['text']

    def _format_spaces(self, lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            # Replate all tabs by spaces
            line = line.expandtabs(4)
            # Calculate the indent of the line, remove spaces in the beginning
            # of the line.
            if line.lstrip().startswith('%%% '):
                # Emacs local variable definition.
                indent = 0
                cmt = '%%% '
            elif line.lstrip().startswith('%%%%'):
                # Long comment. Keep it as is
                continue
            elif line.lstrip().startswith('%'):
                # Line is a comment
                indent = (len(line[1:]) - len(line.lstrip(' %'))) // 4 * 4
                cmt = '%'
            else:
                indent = (len(line) - len(line.lstrip())) // 4 * 4
                cmt = ''
            self.indent = cmt + ' ' * indent
            line = self.indent + line.lstrip(' %')
            # Remove double spaces (except for the indent)
            while '  ' in line[indent:]:
                line = self.indent + line.lstrip(' %').replace('  ', ' ')
            # Make sure all the commas are followed by a space, except for ,~
            # and footnotes
            line = re.sub(r',(?!\s|~|\\footnote)', r', ', line)
            # Move "begin" commands to a new line.
            # TODO: The check for "def" commands should be better.
            if (r'\begin' in line.strip(' %')[6:]
                    and all(x not in line for x in def_list)):
                idx = line.index(r'\begin')
                if not ((match := re.search(r'(?<!\\)%', line))
                        and match.start() < idx):
                    new_line = self.indent + line[idx:]
                    line = line[:idx]
                    lines.insert(i+1, new_line)
            self.update_context(line)
            offset = len(self.indent)
            self.indent = ''
            if self.context == 'equation':
                add_space = self._equation_addspace(line.lstrip(' %'), offset)
            elif self.context == 'text':
                add_space = self._text_addspace(line.lstrip(' %'), offset)
            # Add all the spaces found previously
            for space_pos in sorted(set(add_space), reverse=True):
                line = line[:space_pos] + ' ' + line[space_pos:]
            lines[i] = line.rstrip() + '\n'
        self.reset_context()
        return lines

    def _equation_addspace(self, line: str, offset: int = 0) -> list[int]:
        # Find position that need space
        add_space = []
        self.indent = ' ' * (len(line) - len(line.lstrip(' %')))
        skeleton, parenthesis = self.get_skeleton(line)
        # Add a space before '\' except when following ( [ { $ ^ \ or a space
        # or except when prefacing "right".
        if '\\' in skeleton:
            for match in pattern_backslash.finditer(skeleton):
                if re.search(r'(\\left|\\right|\\[bB]igg?)\s*$',
                             skeleton[:match.start() + 1]):
                    continue
                add_space.append(self.get_index_line(match.start(0), line) + 1)
        # Add a space before '&' except when following \ or a space.
        if '&' in skeleton:
            for match in re.finditer(r'[^\\\s]&', skeleton):
                add_space.append(self.get_index_line(match.start(0), line) + 1)
        # Add a space before and after the + - and / operations.
        if ('+' in skeleton or '-' in skeleton or '/' in skeleton
                or '=' in skeleton):
            add_space.extend(
                [self.get_index_line(idx, line)
                 for idx in self._format_spaces_operation(skeleton)]
            )
        # Add a space after ).
        if ')' in skeleton or ']' in skeleton or '}' in skeleton:
            for match in re.finditer(r'[\)\]\}][A-Za-z0-9]', skeleton):
                add_space.append(self.get_index_line(match.start(0), line) + 1)
        # Add a space after super and underscript.
        if '_' in skeleton or '^' in skeleton:
            for match in re.finditer(r'[_^]\w[A-Za-z0-9]', skeleton):
                add_space.append(self.get_index_line(match.end(), line) - 1)
        # Add a space after digits.
        for match in re.finditer(r'\d[A-Za-z]', skeleton):
            add_space.append(self.get_index_line(match.end(), line) - 1)
        self.indent = ''
        for (start, end), char in parenthesis:
            start = start or 0
            if not start:
                self.indent = ''
            if char == '{':
                if pattern_context.search(line[:start]):
                    add_space.extend(self._text_addspace(line[start+1:end],
                                                         start+1))
                    continue
            elif char == '$':
                if end is None:
                    self._context.pop()
                continue
            add_space.extend(self._equation_addspace(line[start+1:end],
                                                     start+1))
        return [offset + n for n in add_space]

    def _text_addspace(self, line: str, offset: int = 0) -> list[int]:
        add_space = []
        list_parenthesis = [
            self._find_parenthesis(line, self.multline_parenthesis)
        ].copy()
        while list_parenthesis:
            parenthesis = list_parenthesis.pop()
            for ((start, end), char), par in parenthesis.items():
                if char != '$':
                    list_parenthesis.append(par)
                    continue
                assert start is not None
                add_space.extend(
                    self._equation_addspace(line[start+1:end], offset=start+1)
                )
                if end is None:
                    self._context.append('equation')
        return [offset + n for n in add_space]

    def _format_spaces_operation(self, line: str, offset: int = 0
                                 ) -> list[int]:
        add_space = []
        # Add a space before an operation, unless preceeded
        # by a parenthesis or exponent (^)
        for match in re.finditer(r'([^\s\(\[\{\^])[\+\-/=\<\>]', line):
            if not match.group(1) == ' ':
                add_space.append(offset + match.start(1) + 1)
            else:
                assert False
        # Add a space after an operation if not preceded by parenthesis
        # or followed by EOL.
        for match in re.finditer(r'[^\{\(\[](\+|/|=|\\neq|\-|\<)(?!\s|$)',
                                 line):
            add_space.append(offset + match.end())
        return add_space

    def format_tex(self, lines: Iterable[str], first: bool = False
                   ) -> list[str]:
        # TODO: Fix split parenthesis from multiple lines
        new_content = []
        for line in map(str.rstrip, lines):
            # print(line)
            # Detect when we are inside an environment
            self.update_context(line)
            # Compute the indent of the line
            self.indent = ' ' * (len(line) - len(line.lstrip()))
            self.commentafter = len(line)
            # TODO: Manage block separators of thef form %%%% TEXT %%%%
            if set(line.strip()) == {'%'}:
                new_content.append(line[:79] + '\n')
                continue
            # TODO: Compute indent from scratch from previous lines.
            if line.strip().startswith('%'):
                level = len(line) - len(line.lstrip(' %'))
                self.indent = '%' + ' ' * (level - 1)
            elif (match := re.search(r'(?<!\\)%(?!$)', line)):
                self.commentafter = match.start()
            if line.replace('$$', '$').count('$') % 2 == 1:
                if (self.context == 'text'
                        and ('$' not in self.multline_parenthesis
                             or line.strip()[0] != '$')
                        and line.partition('%')[0].strip()[-1] != '$'):
                    assert '$' not in self.multline_parenthesis, line
                    idx = line[::-1].find('$')
                    new_content.extend(self.format_tex([line[:-idx]]))
                    self._context.append('equation')
                    new_content.extend(self.format_tex(
                        [self.indent + 4*' ' + line[-idx:].lstrip()]
                    ))
                    continue
                elif self.context == 'equation':
                    idx = line.find('$')
                    if idx > 0:
                        new_content.extend(self.format_tex([line[:idx]]))
                    self._context.pop()
                    # if '$' in self.multline_parenthesis:
                    #     assert self.multline_parenthesis[-1] == '$'
                    #     self.multline_parenthesis = self.multline_parenthesis[:-1]
                    new_content.extend(self.format_tex([line[idx:]]))
                    continue
            # If line is shoft enough, leave it as it is
            if len(line) <= 80:
                if not first and self.context == 'equation':
                    # TODO: If previous lines were splitted, check that all
                    # TODO: operations here have lower priority. For example,
                    # TODO: if the previous line splitted sums, and this line
                    # TODO: contains some sums, we should split them also.
                    # If there are unmatched parenthesis, split the line anyway
                    ret = self.check_unmatched_parenthesis(line)
                    if ret is not None:
                        new_content.extend(ret)
                        continue
                self.update_multiline_parenthesis(line)
                new_content.append(line + '\n')
                continue
            # Format the line according to the actual context
            try:
                if self.context == 'text':
                    new_content.extend(self._format_text(line))
                elif self.context == 'equation':
                    new_content.extend(self._format_equation(line))
            except Exception as e:
                print(type(e))
                print(line)
                sys.exit(1)
        # Combine the lines to avoid lines too short
        if not first:
            new_content = self.combine_lines(new_content)
        return new_content

    def combine_lines(self, content: list[str]) -> list[str]:
        index_mask = [True,] * (len(content) - 1)
        while len(content) > 1:
            lengths = np.asarray(list(map(len, content)))
            comb_len = lengths[1:] + lengths[:-1]
            valid_comb = comb_len[(comb_len <= 80) & index_mask]
            if not valid_comb.size:
                break
            # Substitute for?:
            # idx = np.where(comb_len == min(valid_comb))[0][0]
            # assert index_mask[idx] is True
            indices = np.where(comb_len == min(valid_comb))[0]
            for idx in indices:
                if index_mask[idx]:
                    break
            else:
                assert False
            first = content[idx]
            second = content[idx + 1]
            assert first, first
            assert second, second
            if not self.allow_combine(first, second):
                index_mask[idx] = False
                continue
            content.pop(idx + 1)
            space = ' '
            # why does the first need to be alnum?
            # if not first[-1].isalnum() and second[0] in {'.', ','}:
            if (second.lstrip(' %')[0] in {'.', ','}
                    or first.strip()[-1] == '%' and first.strip()[-2] != '\\'):
                space = ''
            match = re.search(r'^(.*?)(\s|(?<!\\)%)*$', first)
            if match is None:
                raise ValueError()
            first = match.group(1)
            content[idx] = first + space + second.lstrip(' %')
            index_mask.pop(idx)
        return content

    def allow_combine(self, first: str, second: str) -> bool:
        if re.search(r'(?:\w{3,}|\W)\.$', first.strip(' %')):
            return False
        first = first.strip()
        second = second.strip()
        if (first[0] == '%') ^ (second[0] == '%'):
            return False
        first = first.strip(' %')
        second = second.strip(' %')
        if first == '$' or second.count('$') == 1:
            return False
        if pattern_equal.match(second):
            return False
        if pattern_separate.match(second) or pattern_separate.match(first):
            return False
        if pattern_arrow.match(second) or pattern_arrow.match(first):
            return False
        _first = first.endswith
        _second = second.startswith
        if _second('+') or _second('-'):
            return False
        if _first('(') or _first('[') or _first('{'):
            return False
        if _second(')') or _second(']') or _second('}') or _second(r'\}'):
            return False
        if _first('\\left(') or _first('\\left[') or _first('\\left\\{'):
            return False
        if _second('\\right)') or _second('\\right]') or _second('\\right\\}'):
            return False
        # if _first.strip()[-1] == '%' and _first.strip()[-2] != '\\':
        #     return True
        return True

    def line_split(self, line: str, pattern: Union[str, re.Pattern],
                   keep: Union[bool, str] = False) -> list[str]:
        if not isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        skeleton, _ = self.get_skeleton(line, self.multline_parenthesis)
        lines = []
        prev_idx = 0
        for match in pattern.finditer(skeleton):
            start = self.get_index_line(match.start(), line)
            end = self.get_index_line(match.end(), line)
            sEOL = eEOL = '%' if self.context == 'text' else ''
            if end == len(line) or line[end] == ' ':
                eEOL = ''
            if line[start-1] == ' ' or line[start] == ' ':
                sEOL = ''
            if keep == 'first':
                lines.append(line[prev_idx:end] + eEOL)
                prev_idx = end
            elif keep == 'second':
                lines.append(line[prev_idx:start] + sEOL)
                prev_idx = start
            elif keep is True:
                lines.append(line[prev_idx:start] + sEOL)
                lines.append(line[start:end] + eEOL)
                prev_idx = end
            elif keep is False:
                lines.append(line[prev_idx:start] + sEOL)
                prev_idx = end
        lines.append(line[prev_idx:])
        lines = [line for line in lines if line.strip(' %')]
        new_lines = map(lambda s: self.indent + s.lstrip(' %').rstrip() + '\n',
                        lines)
        return self.format_tex(new_lines)

    def get_index_line(self, idx: int, line: str) -> int:
        idx_l = len(self.indent) + idx
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), _ in parenthesis:
            if start is None:
                assert end is not None
                idx_l = end + idx
                continue
            if start >= idx_l:
                break
            assert end is not None, line
            idx_l += end - start - 1
        return idx_l

    def _format_text(self, line: str) -> list[str]:
        skeleton, parenthesis = self.get_skeleton(line,
                                                  self.multline_parenthesis)
        if self.commentafter < len(line):
            new_lines = [line[:self.commentafter], line[self.commentafter:]]
            return self.format_tex(new_lines)
        # Split newlines into a new line
        pattern = re.compile(r'(\\\\|\\newline)(?=.)')
        if pattern.search(skeleton):
            return self.line_split(line, pattern, keep='first')
        # Split sentences (separated by . or ?) into multiple lines
        pattern = re.compile(r'.[\.\?](?=\s[A-Z{])')
        if pattern.search(skeleton[:-1]):
            return self.line_split(line, pattern, keep='first')
        # Split the line by ':'
        elif re.search(r':\W', skeleton[:-1]):
            return self.line_split(line, ':', keep='first')
        # Split the line by ','
        elif ',' in skeleton[:-1]:
            return self.line_split(line, ',', keep='first')
        # Split the line by ' and '
        elif ' and ' in skeleton[:-1]:
            return self.line_split(line, r'(?<=\s)and(?=\s)', keep='second')
        # Split the formulas into a new line.
        new_lines = []
        if skeleton == '$$':
            start = self.get_index_line(0, line)
            end = self.get_index_line(1, line)
            new_lines.append(line[:start+1].rstrip() + '\n')
            self._context.append('equation')
            indent = self.indent
            new_lines.extend(self.format_tex(
                [indent + 4 * ' ' + line[start+1:end].lstrip()]
            ))
            self._context.pop()
            # TODO?: Add % after last $.
            new_lines.append(indent + line[end:].strip() + '\n')
            return [line for line in new_lines if line]
        if ' $$' in skeleton:
            return self.line_split(line, r'\s\$\$', keep=True)
        # Split {} into multiple lines
        for (_start, _end), char in parenthesis:
            start = _start+1 if _start is not None else 0
            end = _end if _end is not None else len(line)
            if end - start > 40 and char == '{':
                pass
            elif end - start > 75 and char == '(':
                pass
            else:
                continue
            # Decide wether to put comment at the end of the line or not.
            EOL1 = EOL2 = '%'
            if (line[start] == ' '
                    and (char != '{' or start > 2 and line[start-2] != '\\')):
                EOL1 = ''
            if _end is None and line.strip()[-1] != '%':
                EOL2 = ''
            new_lines.append(self.indent + line[:start].lstrip(' %') + EOL1)
            new_lines.append(
                self.indent + 4 * ' ' + line[start:end].lstrip() + EOL2
            )
            new_lines.append(self.indent + line[end:].lstrip())
            new_lines = [line for line in new_lines if line.strip(' %')]
        if new_lines:
            return self.format_tex(new_lines)
        if ' ' in skeleton:
            return self.line_split(line, ' ', keep=False)
        return [line]

    def _format_equation(self, line: str) -> list[str]:
        skeleton, _ = self.get_skeleton(line, self.multline_parenthesis)
        # If equation separator (quad) or equality is present, split line.
        if pattern_separate.search(skeleton):
            return self.line_split(line, pattern_separate, keep=True)
        # Split line in implication
        if pattern_arrow.search(skeleton):
            return self.line_split(line, pattern_arrow, keep=True)
        # Split line in equality
        if pattern_equal.search(skeleton[1:]):
            return self.line_split(line, pattern_equal, keep='second')
        # If unmatched parenthesis, split right after/before.
        ret = self.check_unmatched_parenthesis(line)
        if ret is not None:
            return ret
        # Split sums and subtractions into multiple lines
        ret = self.split_sums(line)
        if ret is not None:
            return ret
        # Split parenthesis into multiple lines if they are big enough.
        ret = self.split_large_parenthesis(line)
        if ret is not None:
            return ret
        # Split the spaces of skeleton
        if ' ' in skeleton:
            return self.line_split(line, r'(?<!\\)\s')
        # If the parenthesis are not big enough, split the line right
        # after a parenthesis
        # TODO: There is no test for this line!
        ret = self.split_after_parenthesis(line)
        if ret is not None:
            return ret
        raise NotImplementedError(f'line "{line}" not splitted.')

    def split_sums(self, line: str) -> Optional[list[str]]:
        skeleton, _ = self.get_skeleton(line, self.multline_parenthesis)
        new_lines = []
        prev_idx = 0
        # TODO: Handle cases like \cong - 3, etc.
        # This should be done easier with new python 3.11 re functions.
        for match in re.finditer(r'[^=\s&]\s*(\+|\-)', skeleton):
            idx_s = match.start(1)
            idx_l = self.get_index_line(idx_s, line)
            new_lines.append(self.indent + line[prev_idx:idx_l].lstrip(' %'))
            prev_idx = idx_l
        if new_lines:
            new_lines.append(self.indent + line[idx_l:].lstrip())
            return self.format_tex(new_lines)

    def check_unmatched_parenthesis(self, line: str) -> Optional[list[str]]:
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), _ in parenthesis:
            if start is None and end > len(self.indent):
                if (match := re.search(r'\\((right|[bB]igg?)\s?\\?)?$',
                                       line[:end])):
                    end = match.start()
                    if end == len(self.indent):
                        continue
                new_lines = [
                    self.indent + 4*' ' + line[:end].strip() + '\n',
                    self.indent + line[end:].rstrip() + '\n'
                ]
            elif end is None and start + 2 < len(line.rstrip()):
                new_lines = [
                    line[:start+1].rstrip() + '\n',
                    self.indent + 4*' ' + line[start+1:].strip() + '\n'
                ]
            else:
                continue
            return self.format_tex(new_lines)

    def split_large_parenthesis(self, line: str) -> Optional[list[str]]:
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), _ in parenthesis:
            if end is None or start is None or end - start < 30:
                continue
            if pattern_context.search(line[:start]):
                indent = self.indent
                new_lines = [line[:start + 1] + '%\n']
                self._context.append('text')
                new_lines += self.format_tex(
                    [self.indent + 4*' ' + line[start+1:end].lstrip() + '%\n']
                )
                self._context.pop()
                new_lines += [indent + line[end:] + '\n']
                return self.format_tex(new_lines)
            if (match := re.search(r'\\(right|[bB]igg?)\s?\\?$', line[:end])):
                end = match.start()
            new_lines = [
                line[:start + 1] + '\n',
                self.indent + 4*' ' + line[start+1:end].lstrip() + '\n',
                self.indent + line[end:] + '\n'
            ]
            return self.format_tex(new_lines)

    def split_after_parenthesis(self, line: str) -> Optional[list[str]]:
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), char in reversed(parenthesis):
            if end > 80:
                continue
            elif char == '{' and end + 1 < len(line) and line[end+1] == '{':
                continue
            new_lines = [line[:end + 1] + '\n',
                         self.indent + line[end+1:].strip() + '\n']
            return self.format_tex(new_lines)

    def update_multiline_parenthesis(self, line: str) -> None:
        if 'phantom' in line:
            return
        open_p = '([{'
        close_p = ')]}'
        parenthesis = self._find_parenthesis(line,
                                             self.multline_parenthesis).copy()
        new_parenthesis = []
        while parenthesis:
            ((start, end), char), child = parenthesis.popitem()
            if start is None:
                new_parenthesis.append((end, Parenthesis.get_match(char)))
            elif end is None:
                new_parenthesis.append((start, char))
            else:
                continue
            parenthesis |= child
        sent = True
        for _, char in sorted(new_parenthesis):
            if char in close_p:
                assert sent, line
                assert self.multline_parenthesis, line
                _char = self.multline_parenthesis[-1]
                assert open_p.index(_char) == close_p.index(char), line
                self.multline_parenthesis = self.multline_parenthesis[:-1]
                continue
            elif char == '$':
                if '$' not in self.multline_parenthesis:
                    assert line.rstrip()[-1] == '$'
                    self.multline_parenthesis += '$'
                    self._context.append('equation')
                else:
                    assert self.multline_parenthesis[-1] == '$'
                    self.multline_parenthesis = self.multline_parenthesis[:-1]
                continue
            sent = False
            if char in open_p:
                self.multline_parenthesis += char

    @classmethod
    @functools.cache
    def get_skeleton(cls, line: str, unmatched_parenthesis: str = ''
                     ) -> tuple[str, ParenthesisType]:
        parenthesis = cls._find_parenthesis(line, unmatched_parenthesis)
        skeleton = line.lstrip(' %')
        offset = len(line) - len(skeleton)
        for (start, end), _ in parenthesis:
            if end is None:
                skeleton = skeleton[:start-offset+1]
                break
            elif start is None:
                skeleton = skeleton[end-offset:]
                offset = end
            else:
                skeleton = skeleton[:start-offset+1] + skeleton[end-offset:]
                offset += end - start - 1
        return skeleton.strip(' %'), parenthesis

    @staticmethod
    @functools.cache
    def _find_parenthesis(line: str, unmatched_parenthesis: str = ''
                          ) -> ParenthesisType:
        return Parenthesis().parse(line, unmatched_parenthesis)

    def __repr__(self) -> str:
        if self.formatted_lines == self.init_string:
            return 'String not modified'
        return ''.join(self.formatted_lines)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return repr(self) == other
        return NotImplemented

def run_from_command() -> None:
    pathdir = pathlib.Path.cwd()
    filename = pathdir / pathlib.Path(sys.argv[1])
    print(filename)
    dest = pathdir / pathlib.Path(sys.argv[1])
    with filename.open() as file:
        content = file.readlines()
    new_content = TeXFormatter(content).formatted_lines
    with dest.open('w') as file:
        file.writelines(new_content)


def run_from_editor() -> None:
    s = r'''
'''
    r = TeXFormatter(s)
    print(r)


if __name__ == '__main__':
    try:
        __IPYTHON__
        run_from_editor()
    except NameError:
        run_from_command()
