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
    # import pdb; pdb.set_trace()
    new_content = TeXFormatter(content).formatted_lines
    with dest.open('w') as file:
        file.writelines(new_content)


def run_from_editor() -> None:
    s = r'''
distribution functions and $d \hat \si^{\rm(res)}_{ij\tosv \tth}/ dQ^2 \left. \right|_{\scriptscriptstyle({\rm NNLO})}$ represents the perturbative expansion of the resummed cross section truncated at NNLO. The inverse Mellin transform (\ref{invmel}) is evaluated numerically using according to the ``Minimal Prescription'' ~\cite{Catani:1996yz}
'''
    r = TeXFormatter(s)
    print(r)


if __name__ == '__main__':
    try:
        __IPYTHON__
        run_from_editor()
    except NameError:
        run_from_command()

### TESTS ###

# 5
s = r'''
\cite{CMS:2014wdm,CMS:2017odg,ATLAS:2017ztq,CMS:2018fdh,ATLAS:2018mme,ATLAS:2018kot,ATLAS:2018ynr},
'''
assert TeXFormatter(s) == '\n\\cite{%\n    CMS:2014wdm, CMS:2017odg, ATLAS:2017ztq, CMS:2018fdh,\n    ATLAS:2018mme, ATLAS:2018kot, ATLAS:2018ynr%\n},\n'

# 5
s = r'''
\title{State-of-the-art cross sections for $\boldsymbol{t\bar t H}$:\\ NNLO predictions matched with NNLL resummation and EW corrections}
'''
assert TeXFormatter(s) == '\n\\title{%\n    State-of-the-art cross sections for $\\boldsymbol{t \\bar t H}$:\\\\\n    NNLO predictions matched with NNLL resummation and EW corrections%\n}\n'

# 6
s = r'''
\begin{itemize}
\item $\muF =\muR = m_t+m_H/2$
\item $\muF =\muR =H_T/2$
\item $\muF =\muR   \equiv Q/2$ ($Q\equiv M_{ttH}$)
\end{itemize}
'''
assert TeXFormatter(s) == '\n\\begin{itemize}\n\\item $\\muF = \\muR = m_t + m_H / 2$\n\\item $\\muF = \\muR = H_T / 2$\n\\item $\\muF = \\muR \\equiv Q / 2$ ($Q \\equiv M_{ttH}$)\n\\end{itemize}\n'

# 6
s = r'''
In particular, we are interested in evolving $S$ from the scale where renormalization takes place $\mu_0=\mu_R$ to the soft scale $\mu = Q\bar{N}^{-1}$. Where $\bar{N}=Ne^{\gamma_E}$ as defined in \cite{Kulesza17}. Then
'''
assert TeXFormatter(s) == '\nIn particular, we are interested in evolving $S$\nfrom the scale where renormalization takes place\n$\\mu_0 = \\mu_R$ to the soft scale $\\mu = Q \\bar{N}^{-1}$.\nWhere $\\bar{N} = Ne^{\\gamma_E}$ as defined in \\cite{Kulesza17}.\nThen\n'

# 6
s = r'''
\def\cCode#1{\begin{lstlisting}[mathescape,basicstyle=\small
\ttfamily,frame=leftline,aboveskip=4mm,belowskip=4mm,xleftmargin=20pt,framexleftmargin=10pt,
numbers=none,framerule=2pt,abovecaptionskip=0.0mm,belowcaptionskip=3.5mm #1]}
'''
assert TeXFormatter(s) == '\n\\def\\cCode#1{\\begin{lstlisting}[mathescape, basicstyle=\\small\n\\ttfamily, frame=leftline, aboveskip=4mm, belowskip=4mm,\nxleftmargin=20pt, framexleftmargin=10pt,\nnumbers=none, framerule=2pt, abovecaptionskip=0.0mm, belowcaptionskip=3.5mm #1]}\n'

#6
s = r'''
where $\left.\sigma_{\rm NNLL}^{\rm SCET}\right|_{\alphas^2}$ ($\left.\sigma_{\rm NNLL}^{\rm dQCD}\right|_{\alphas^2}$) is
the expansion of $ \sigma_{\rm NNLL}^{\rm SCET}$ ($ \sigma_{\rm NNLL}^{\rm dQCD}$) up to order $\alphas^2$.\\
The two resummed predictions are combined by simply considering their average:
'''
assert TeXFormatter(s) == '\nwhere $\\left. \\sigma_{\\rm NNLL}^{\\rm SCET}\\right|_{\\alphas^2}$\n($\\left. \\sigma_{\\rm NNLL}^{\\rm dQCD}\\right|_{\\alphas^2}$) is\nthe expansion of $ \\sigma_{\\rm NNLL}^{\\rm SCET}$\n($ \\sigma_{\\rm NNLL}^{\\rm dQCD}$) up to order $\\alphas^2$.\\\\\nThe two resummed predictions are combined by simply considering their average:\n'

# 8
s = r'''
Now, equating both derivatives and using equation \eqref{eq:Kmatrix} for $U$ we arrive at the following equation:
\begin{equation*}
    \frac{\Gamma(\alpha_S)}{\beta(\alpha_S)}K(\alpha_S)
    = \dv{K(\alpha_S)}{\alpha_S} - K(\alpha_S)\frac{\Gamma^{(1)}}{2\pi\alpha_Sb_0}
\end{equation*}
'''
assert TeXFormatter(s) == '\nNow, equating both derivatives and using equation \\eqref{eq:Kmatrix} for\n$U$ we arrive at the following equation:\n\\begin{equation*}\n    \\frac{\\Gamma(\\alpha_S)}{\\beta(\\alpha_S)} K(\\alpha_S)\n    = \\dv{K(\\alpha_S)}{\\alpha_S}\n    - K(\\alpha_S) \\frac{\\Gamma^{(1)}}{2 \\pi \\alpha_S b_0}\n\\end{equation*}\n'

# 10
s = r'''
\begin{align}
\label{eq:SCET_11}
	(\mu_F,\mu_R,\mu_h) \in \{&(S,S,S),(2S,S,2S),(2S,S,S),(S/2,S,S/2),(S/2,S,S),(S,2 S,S),\,\nonumber\\
	& (S,2 S,2 S),(S,S/2,S/2),(S,S/2,S),(2 S,2 S,2 S),(S/2,S/2,S/2)\} \, .
\end{align}
'''
assert TeXFormatter(s) == '\n\\begin{align}\n\\label{eq:SCET_11}\n    (\\mu_F, \\mu_R, \\mu_h) \\in \\{\n        &(S, S, S), (2 S, S, 2 S), (2 S, S, S), (S / 2, S, S / 2),\n        (S / 2, S, S), (S, 2 S, S), \\, \\nonumber \\\\\n        & (S, 2 S, 2 S), (S, S / 2, S / 2),\n        (S, S / 2, S), (2 S, 2 S, 2 S), (S / 2, S / 2, S / 2)\n    \\} \\, .\n\\end{align}\n'

# 13
s = r'''
\begin{equation*}
    -b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2}) = \alpha_S(\mu_0^2)\frac{b_1}{b_0}\log(\frac{\frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)}+ \alpha_S(\mu_0^2)\frac{b_1}{b_0}}{1+\alpha_S(\mu_0^2) \frac{b_1}{b_0}})+1-\frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)} + \order{\alpha_S^2}
\end{equation*}
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    -b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2})\n    = \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0} \\log(\n        \\frac{\n            \\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)}\n            + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0}\n        }{1 + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0}}\n    )\n    + 1\n    - \\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)}\n    + \\order{\\alpha_S^2}\n\\end{equation*}\n'

# 13
s = r'''
where $f_{i / h}(x, \muF^2)$ are moments of the parton
distribution functions and $ d \hat \si^{\rm
(res)}_{ij\tosv \tth}/ dQ^2 \left.
\right|_{\scriptscriptstyle({\rm NNLO})}$ represents the perturbative expansion of the resummed cross section truncated at NNLO. The inverse Mellin transform (\ref{invmel}) is evaluated numerically using according to the ``Minimal Prescription'' ~\cite{Catani:1996 yz}
along a contour ${\sf C}$ in the complex-$N$ space.
'''
assert TeXFormatter(s) == "\nwhere $f_{i / h}(x, \\muF^2)$ are moments of the parton\ndistribution functions and $\n    d \\hat \\si^{\n        \\rm\n(res)}_{ij \\tosv \\tth} / dQ^2 \\left.\n\\right|_{\\scriptscriptstyle({\\rm NNLO})}\n$\nrepresents the perturbative expansion\nof the resummed cross section truncated at NNLO.\nThe inverse Mellin transform (\\ref{invmel}) is evaluated numerically\nusing according to the ``Minimal Prescription'' ~\\cite{Catani:1996 yz}\nalong a contour ${\\sf C}$ in the complex-$N$ space.\n"

# 14
s = '''
However, the consequence is that the non-radiative amplitude in the r.h.s. of eq. \\eqref{eq:NLP} is evaluated using the momenta $p$, which are unphysical for this process, because $\\sum \\eta_i p_i \\neq 0$. This might seem problematic because an amplitude is intrinsically defined for physical momenta, and it is not uniquely defined for unphysical momenta. Therefore, the value of $\\mathcal{H}(p)$ is ambiguous, which translates into an ambiguity on $\\mathcal{A}(p, k)$ and thus seems to invalidate eq. \\eqref{eq:NLP}. The argument, however, is not entirely correct, as shown in \\cite{Balsach:2023ema}. Indeed, although an ambiguity is present, it only affects the NNLP terms.
'''
assert TeXFormatter(s) == '\nHowever,\nthe consequence is that the non-radiative amplitude in the r.h.s. of eq.\n\\eqref{eq:NLP} is evaluated using the momenta $p$,\nwhich are unphysical for this process, because $\\sum \\eta_i p_i \\neq 0$.\nThis might seem problematic because an amplitude is\nintrinsically defined for physical momenta,\nand it is not uniquely defined for unphysical momenta.\nTherefore, the value of $\\mathcal{H}(p)$ is ambiguous,\nwhich translates into an ambiguity on $\\mathcal{A}(p, k)$\nand thus seems to invalidate eq. \\eqref{eq:NLP}.\nThe argument, however, is not entirely correct,\nas shown in \\cite{Balsach:2023ema}.\nIndeed, although an ambiguity is present, it only affects the NNLP terms.\n'

# 15
s = r'''
\begin{eqnarray}
g_s (N)
= \frac{1}{2 \pi b_0} \left\{
    \log(1 - 2 \lambda) + \alphas(\muR^2) \left[
        \frac{b_1}{b_0} \frac{ \log(1 - 2 \lambda)}{ 1 - 2 \lambda}
        - 2 \gamma_{\rm E} b_0 \frac{2 \lambda}{1 - 2 \lambda} \right.
        \right.
        \nonumber \\
\left. \left.
        + \, b_0 \log \left( \frac{Q^2}{\muR^2} \right)
        \frac{2 \lambda}{1 - 2 \lambda}
    \right]
\right\}
\end{eqnarray}
'''
assert TeXFormatter(s) == 'String not modified'

# 17
s = r'''
\begin{figure}[ht!]
    \includegraphics[width=0.49\textwidth]{tth_figxsec.pdf}
    \includegraphics[width=0.49\textwidth]{tth_figerr.pdf}
    \caption{\label{fig:xsecerr}{\bf Left panel:} the total cross section for $t\bar t H$, $\sigma_{\rm NNLO+NNLL+EW}$, and the relative impact of the different
    contributions with respect to $\sigma_{\rm NLO}$. {\bf Right panel:} scale uncertainties computed for the cross section computed at different
accuracies. Solid lines display the total width of the scale-uncertainty band, while dashed line the maximum variation with respect to the central prediction.}
\end{figure}
'''
assert TeXFormatter(s) == '\n\\begin{figure}[ht!]\n    \\includegraphics[width=0.49\\textwidth]{tth_figxsec.pdf}\n    \\includegraphics[width=0.49\\textwidth]{tth_figerr.pdf}\n    \\caption{%\n        \\label{fig:xsecerr}{\\bf Left panel:} the total cross section for\n        $t \\bar t H$, $\\sigma_{\\rm NNLO + NNLL + EW}$,\n        and the relative impact of the different\n    contributions with respect to $\\sigma_{\\rm NLO}$.\n    {\\bf Right panel:} scale uncertainties\n    computed for the cross section computed at different\n    accuracies.\n    Solid lines display the total width of the scale-uncertainty band,\n    while dashed line the maximum\n    variation with respect to the central prediction.%\n}\n\\end{figure}\n'

#18
s = r'''
%%%%%%%%%%%%%%%%%%%%%%%%%
 \begin{figure}
     \centering
\includegraphics[width=.49\textwidth]{tth_nnll_comp_qcd_scet.pdf}
\includegraphics[width=.49\textwidth]{tth_nnllnnlo_lhc136_asym.pdf}
     \caption{Left: comparison between NNLO+NNLL results in  dQCD and SCET for three parametrically different choices of the default
     scales.  Right: comparison of the combined NNLO+NNLL results with NNLO, for the same three sets of scales. No EW corrections are included. See the text for
     additional explanations on the estimation of the uncertainties. }
     \label{fig:tth_comparisons}
 \end{figure}
%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
assert TeXFormatter(s) == '\n%%%%%%%%%%%%%%%%%%%%%%%%%\n\\begin{figure}\n    \\centering\n\\includegraphics[width=.49\\textwidth]{tth_nnll_comp_qcd_scet.pdf}\n\\includegraphics[width=.49\\textwidth]{tth_nnllnnlo_lhc136_asym.pdf}\n    \\caption{%\n        Left: comparison between NNLO+NNLL results in dQCD\n        and SCET for three parametrically different choices of the default\n    scales.\n    Right: comparison of the combined NNLO+NNLL results with NNLO,\n    for the same three sets of scales.\n    No EW corrections are included.\n    See the text for\n    additional explanations on the estimation of the uncertainties. }\n    \\label{fig:tth_comparisons}\n\\end{figure}\n%%%%%%%%%%%%%%%%%%%%%%%%%%\n'

#19
s = r'''
LO and NLO contributions different from $\Sigma_{\rm LO,1}$, $\Sigma_{\rm NLO,1}$, involve partonic processes with at least one photon in the initial state
and therefore depend on the photon PDF. The dominant contribution originates from the process is
$g \gamma  \to t\bar t H$,\footnote{See Ref.~\cite{Pagani:2016caq} for a analogous and more detailed discussion for the case of the $t \bar t$ production.}
which enters both at LO and NLO. However, also $ q\gamma$ and $\gamma\gamma$ initial states are possible. The quantities $\Sigma_{\rm NLO~EW}$,  $\Sigma_{\rm NLO,3}$ and $ \Sigma_{\rm NLO,4}$ receive contributions from the $q\gamma\to t\bar t H q$ processes, while the $\gamma \gamma$ initial state contributes to $\Sigma_{\rm LO,3}$, via $\gamma\gamma\to t\bar t H $, to $\Sigma_{\rm NLO,3}$, via $\gamma\gamma\to t\bar t H g$, and to $ \Sigma_{\rm NLO,4}$, via $\gamma\gamma\to t\bar t H \gamma$.
'''
assert TeXFormatter(s) == '\nLO and NLO contributions different from $\\Sigma_{\\rm LO, 1}$,\n$\\Sigma_{\\rm NLO, 1}$,\ninvolve partonic processes with at least one photon in the initial state\nand therefore depend on the photon PDF.\nThe dominant contribution originates from the process is\n$g \\gamma \\to t \\bar t H$,\\footnote{%\n    See Ref.~\\cite{Pagani:2016caq} for a analogous\n    and more detailed discussion for the case of the $t \\bar t$ production.%\n}\nwhich enters both at LO and NLO.\nHowever, also $ q \\gamma$ and $\\gamma \\gamma$ initial states are possible.\nThe quantities $\\Sigma_{\\rm NLO~EW}$,\n$\\Sigma_{\\rm NLO, 3}$ and $ \\Sigma_{\\rm NLO, 4}$ receive contributions from the\n$q \\gamma \\to t \\bar t H q$ processes,\nwhile the $\\gamma \\gamma$ initial state contributes to $\\Sigma_{\\rm LO, 3}$,\nvia $\\gamma \\gamma \\to t \\bar t H $, to $\\Sigma_{\\rm NLO, 3}$,\nvia $\\gamma \\gamma \\to t \\bar t H g$, and to $ \\Sigma_{\\rm NLO, 4}$,\nvia $\\gamma \\gamma \\to t \\bar t H \\gamma$.\n'

# 22
s = r'''
with
\begin{align*}
    \tilde{g}_S(\mu, \mu_0) & = -\frac{1}{2\pi b_0}\log(\frac{\alpha_S(\mu^2)}{\alpha_S(\mu_0^2)})
    = \frac{1}{2\pi b_0}\log(\frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)})\\&
    = \frac{1}{2\pi b_0}\log(1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2}) + \alpha_S(\mu_0^2)\frac{b_1}{b_0}\log(1+b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu^2_0}))) + \order{\alpha_S^2}\\&
    = \frac{1}{2\pi b_0}\left[\log(1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2})) + \alpha_S(\mu_0^2)\frac{b_1}{b_0}\frac{\log(1+b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu^2_0}))}{1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2})}\right] + \order{\alpha_S^2}
\end{align*}
'''
assert TeXFormatter(s) == '\nwith\n\\begin{align*}\n    \\tilde{g}_S(\\mu, \\mu_0) &\n    = - \\frac{1}{2 \\pi b_0} \\log(\\frac{\\alpha_S(\\mu^2)}{\\alpha_S(\\mu_0^2)})\n    = \\frac{1}{2 \\pi b_0} \\log(\\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)}) \\\\&\n    = \\frac{1}{2 \\pi b_0} \\log(\n        1\n        + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2})\n        + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0} \\log(\n            1 + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu^2_0})\n        )\n    )\n    + \\order{\\alpha_S^2} \\\\&\n    = \\frac{1}{2 \\pi b_0} \\left[\n        \\log(1 + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2}))\n        + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0} \\frac{\n            \\log(1 + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu^2_0}))\n        }{1 + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2})}\n    \\right]\n    + \\order{\\alpha_S^2}\n\\end{align*}\n'

# 22
s = r'''
which we can now solve iteratively by substituting the formula onto itself (and neglecting $\order{\alpha_S^2}$ terms). Note that $\mu \ll \mu_0$, so in general $\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu^2_0})$ is not necessarily a small quantity and thus we cannot neglect those terms
\begin{align*}
    \frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)} &= 1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2}) +  \alpha_S(\mu_0^2)\frac{b_1}{b_0}\log(1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu^2_0}) + \order{\alpha_S}) + \order{\alpha_S^2}\\&
    = 1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2}) +  \alpha_S(\mu_0^2)\frac{b_1}{b_0}\log(1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu^2_0})) + \order{\alpha_S^2}
\end{align*}
Putting everything together in equation \eqref{eq:Kmatrix} we have
'''
assert TeXFormatter(s) == '\nwhich we can now solve iteratively by substituting\nthe formula onto itself (and neglecting $\\order{\\alpha_S^2}$ terms).\nNote that $\\mu \\ll \\mu_0$,\nso in general $\\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu^2_0})$\nis not necessarily a small quantity and thus we cannot neglect those terms\n\\begin{align*}\n    \\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)} &\n    = 1\n    + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2})\n    + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0} \\log(\n        1 + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu^2_0}) + \\order{\\alpha_S}\n    )\n    + \\order{\\alpha_S^2} \\\\&\n    = 1\n    + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2})\n    + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0} \\log(\n        1 + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu^2_0})\n    )\n    + \\order{\\alpha_S^2}\n\\end{align*}\nPutting everything together in equation \\eqref{eq:Kmatrix} we have\n'

# 26
s = r'''
We have checked analytically that with this choice of scales in the SCET expressions, they agree with the dQCD expressions up to the higher order logarithmic terms in dQCD, as explained below. These terms originate from various sources. To start with, in SCET any ratio of scales of the form $\ln \mu_i/\mu_j$ is counted as large, where $\mu_i \in \{\muF,\muR,\mu_s,\mu_h\}$. This implies, for instance, that in the SCET formulas in the Mellin-space which are used here corrections logarithmic in $ \bar N = N e^{\gamma_E}$ are resummed. Instead, the particular formulation of the dQCD expressions presented insection~\ref{sec:theo-nnllk}, resums $\ln N$ terms, with additional terms involving $\gamma_E$ included up to the NNLL accuracy. Another difference stems from the treatment of the  $\exp (\alphas g_3)$ term: while in dQCD it is fully included, in SCET its expansion up to $\O (\alphas)$ is considered. Although technically this term is of the NNLL accuracy, we have checked that the numerical impact of the difference in its implementation is negligible.
Moreover, the set of non-logarithmic (in $N$ or $\bar N$) terms collected in the perturbative hard and soft functions in Eqs.~\ref{b:kernelmellin},~\ref{eq:res:fact:dqcd} begin to differ at $\O (\alphas^2)$, what corresponds to differences of order N$^3$LL and higher.
'''
assert TeXFormatter(s) == '\nWe have checked analytically that with this\nchoice of scales in the SCET expressions,\nthey agree with the dQCD expressions up to the\nhigher order logarithmic terms in dQCD, as explained below.\nThese terms originate from various sources.\nTo start with,\nin SCET any ratio of scales of the form $\\ln \\mu_i / \\mu_j$ is counted as large,\nwhere $\\mu_i \\in \\{\\muF, \\muR, \\mu_s, \\mu_h\\}$.\nThis implies, for instance, that in the SCET formulas in the Mellin-space which\nare used here corrections logarithmic in\n$ \\bar N = N e^{\\gamma_E}$ are resummed.\nInstead, the particular formulation of the dQCD\nexpressions presented insection~\\ref{sec:theo-nnllk}, resums $\\ln N$ terms,\nwith additional terms involving $\\gamma_E$ included up to the NNLL accuracy.\nAnother difference stems from the treatment of the $\\exp (\\alphas g_3)$ term:\nwhile in dQCD it is fully included,\nin SCET its expansion up to $\\O (\\alphas)$ is considered.\nAlthough technically this term is of the NNLL accuracy,\nwe have checked that the numerical\nimpact of the difference in its implementation is negligible.\nMoreover, the set of non-logarithmic (in $N$ or $\\bar N$)\nterms collected in the perturbative hard\nand soft functions in Eqs.~\\ref{b:kernelmellin},%\n~\\ref{eq:res:fact:dqcd} begin to differ at $\\O (\\alphas^2)$,\nwhat corresponds to differences of order N$^3$LL and higher.\n'

# 31
s = r'''
The jet functions $\Delta^i$ account for (soft-)collinear logarithmic contributions from the initial state partons and are well known at NNLL~\cite{Catani:1996yz,Catani:2003zt}.  The term $\mathbf{\bar{U}}_R\,\mathbf{\tilde S}_R \, \mathbf{{U}}_R$ originates from a solution of the renormalization group equation for the soft function and consists of the evolution matrices $\mathbf{\bar{U}}_R$, $\mathbf{{U}}_R$, as well as the function $\mathbf{\tilde S}_R$ which plays the role of a boundary condition of the renormalization group equation. The evolution matrices are given by (reverse in the case of $\mathbf{\bar{U}}_R$) path-ordered exponentials of the soft anomalous dimension matrix {$\mathbf{\bar \Gamma}_{ij\tosv \tth}(\alphas)= \left(\frac{\alphas}{\pi}\right) \mathbf{\bar \Gamma}^{(1)}_{ij\tosv \tth} +\left(\frac{\alphas}{\pi}\right)^2 \mathbf{\bar \Gamma}^{(2)}_{ij\tosv \tth}+\ldots$} which is obtained by subtracting the contributions already taken into account in $\Delta^i \Delta^j$ from the full soft anomalous dimension for the process $ij \to \tth$.   At NLL, the path-ordered exponentials collapse to standard exponential factors in the colour space $\mathbf R$ where $\mathbf \Gamma^{(1)}_R$  is diagonal.  At NNLL, the  path-ordered exponentials are eliminated by treating $\mathbf{U}_R$ and  $\mathbf{\bar{U}}_R$  perturbatively
'''
assert TeXFormatter(s) == '\nThe jet functions $\\Delta^i$ account for (soft-)collinear\nlogarithmic contributions from the initial state partons\nand are well known at NNLL~\\cite{Catani:1996yz, Catani:2003zt}.\nThe term $\\mathbf{\\bar{U}}_R \\, \\mathbf{\\tilde S}_R \\, \\mathbf{{U}}_R$\noriginates from a solution of the\nrenormalization group equation for the soft function\nand consists of the evolution matrices $\\mathbf{\\bar{U}}_R$,\n$\\mathbf{{U}}_R$, as well as the function $\\mathbf{\\tilde S}_R$\nwhich plays the role of a boundary\ncondition of the renormalization group equation.\nThe evolution matrices are given by\n(reverse in the case of $\\mathbf{\\bar{U}}_R$)\npath-ordered exponentials of the soft anomalous dimension matrix {%\n    $\n        \\mathbf{\\bar \\Gamma}_{ij \\tosv \\tth}(\\alphas)\n        = \\left(\\frac{\\alphas}{\\pi}\\right)\n        \\mathbf{\\bar \\Gamma}^{(1)}_{ij \\tosv \\tth}\n        + \\left(\\frac{\\alphas}{\\pi}\\right)^2\n        \\mathbf{\\bar \\Gamma}^{(2)}_{ij \\tosv \\tth}\n        + \\ldots\n    $%\n} which is obtained by subtracting the contributions\nalready taken into account in $\\Delta^i \\Delta^j$\nfrom the full soft anomalous dimension for the process $ij \\to \\tth$.\nAt NLL, the path-ordered exponentials collapse to standard\nexponential factors in the colour space\n$\\mathbf R$ where $\\mathbf \\Gamma^{(1)}_R$ is diagonal.\nAt NNLL,\nthe path-ordered exponentials are eliminated by treating $\\mathbf{U}_R$\nand $\\mathbf{\\bar{U}}_R$ perturbatively\n'

# 34
s = r'''
\begin{align*}
    S^{(1)}&
    = S^{(0)}\left(\red{\frac{1}{2}}\sum C^{ij}S^{ij} + (C_1 + C_2)S^{12}\right)\\&
    = S^{(0)}\left(
        2C^{12}S^{12}
        + 2C^{13}S^{13}
        + 2C^{14}S^{14}
        + 2C^{23}S^{23}
        + 2C^{24}S^{24}
        + C^{33}S^{33}
        + 2C^{34}S^{34}
        + C^{44}S^{44}
        + (C_1+C_2)S^{12}
    \right)\\&
    = S^{(0)}\bigg(
        C^{34}\left(2S^{34}-S^{33}-S^{44}\right)
        + C^{13}\left(2S^{13}-2S^{14}-S^{33}+S^{44}\right)
        \\&\phantom{= S^{(0)}\bigg(}
        + C^{23}\left(2S^{23}-2S^{24}-S^{33}+S^{44}\right)
        + N_cC_8\left(-S^{14}-S^{24}+S^{44}\right)
        + (2C^{12}+C_1+C_2)S^{12}
    \bigg)\\&
    = S^{(0)}\bigg(
        C^{34}\left(2S^{34}-S^{33}-S^{44}\right)
        + 2C^{13}\left(S^{13}-S^{14}-S^{23}+S^{24}\right)
        \\&\phantom{= S^{(0)}\bigg(}
        + \frac{N_c}{2}C_8\left(2S^{12}-2S^{14}-2S^{23}+S^{33}+S^{44}\right)
    \bigg)
\end{align*}
'''
assert TeXFormatter(s) == '\n\\begin{align*}\n    S^{(1)} &\n    = S^{(0)} \\left(\n        \\red{\\frac{1}{2}} \\sum C^{ij} S^{ij} + (C_1 + C_2) S^{12}\n    \\right) \\\\&\n    = S^{(0)} \\left(\n        2 C^{12} S^{12}\n        + 2 C^{13} S^{13}\n        + 2 C^{14} S^{14}\n        + 2 C^{23} S^{23}\n        + 2 C^{24} S^{24}\n        + C^{33} S^{33}\n        + 2 C^{34} S^{34}\n        + C^{44} S^{44}\n        + (C_1 + C_2) S^{12}\n    \\right) \\\\&\n    = S^{(0)} \\bigg(\n        C^{34} \\left(2 S^{34} - S^{33} - S^{44}\\right)\n        + C^{13} \\left(2 S^{13} - 2 S^{14} - S^{33} + S^{44}\\right)\n        \\\\& \\phantom{= S^{(0)} \\bigg(}\n        + C^{23} \\left(2 S^{23} - 2 S^{24} - S^{33} + S^{44}\\right)\n        + N_c C_8 \\left(-S^{14} - S^{24} + S^{44}\\right)\n        + (2 C^{12} + C_1 + C_2) S^{12}\n    \\bigg) \\\\&\n    = S^{(0)} \\bigg(\n        C^{34} \\left(2 S^{34} - S^{33} - S^{44}\\right)\n        + 2 C^{13} \\left(S^{13} - S^{14} - S^{23} + S^{24}\\right)\n        \\\\& \\phantom{= S^{(0)} \\bigg(}\n        + \\frac{N_c}{2} C_8 \\left(\n            2 S^{12} - 2 S^{14} - 2 S^{23} + S^{33} + S^{44}\n        \\right)\n    \\bigg)\n\\end{align*}\n'

# 49
s = r'''
\begin{equation*}
    \left(2\pi b_0 + \gamma_i - \gamma_j\right)K_{ij}^{(1)}
    = \frac{\pi b_1}{b_0}\gamma_i\delta_{ij} - \Gamma_{ij}^{(2)}
\end{equation*}
\begin{equation*}
    K_{ij}^{(1)}
    = \frac{b_1}{2b^2_0}\gamma_i\delta_{ij} - \frac{\Gamma_{ij}^{(2)}}{\left(2\pi b_0 + \gamma_i - \gamma_j\right)}
\end{equation*}
This can be continued to obtain higher order expansions, for example the $\order{\alpha_S}$ coefficient gives us an equation to find $K^{(2)}$
\begin{equation*}
    4\pi b_0 K^{(2)} - \comm{K^{(2)}}{\Gamma^{(1)}}
    = \frac{\pi b_1}{b_0}\Gamma^{(1)}K^{(1)} - \Gamma^{(2)}K^{(1)} + \frac{\pi^2b_2}{b_0}\Gamma^{(1)} - \frac{\pi^2b_1^2}{b_0^2}\Gamma^{(1)} + \frac{\pi b_1}{b_0}\Gamma^{(2)} - \Gamma^{(3)}
\end{equation*}
But $K^{(1)}$ is enough for our purposes.

Now that we know $K$, the only thing we need to compute to calculate $U$ is the ratio between $\alpha_S(\mu_0^2)$ and $\alpha_S(\mu^2)$, to do it we can use the beta function:
\begin{align*}
    -b_0\log(\frac{\mu^2}{\mu_0^2})& = \int_{\alpha_S(\mu^2_0)}^{\alpha_S(\mu^2)}\frac{-2b_0}{\beta(\alpha_S)}\dd{\alpha_S}
    = \int_{\alpha_S(\mu^2_0)}^{\alpha_S(\mu^2)}\frac{b_0}{\alpha^2_S\left(b_0 + \alpha_S b_1\right)}\dd{\alpha_S} + \order{\alpha_S}\\&
    = \left(\frac{b_1}{b_0}\log(\frac{1}{\alpha_S(\mu^2)}+ \frac{b_1}{b_0})-\frac{1}{\alpha_S(\mu^2)}\right)-\left(\frac{b_1}{b_0}\log(\frac{1}{\alpha_S(\mu_0^2)}+ \frac{b_1}{b_0})-\frac{1}{\alpha_S(\mu_0^2)}\right) + \order{\alpha_S}
\end{align*}
One can solve for $\alpha_S(\mu^2)$ in terms of the Lambert W function \cite{Brodsky16, Karliner98}, but we are only interested in the ratio up to $\order{\alpha_S^2}$ corrections, so we can solve it as
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    \\left(2 \\pi b_0 + \\gamma_i - \\gamma_j\\right) K_{ij}^{(1)}\n    = \\frac{\\pi b_1}{b_0} \\gamma_i \\delta_{ij} - \\Gamma_{ij}^{(2)}\n\\end{equation*}\n\\begin{equation*}\n    K_{ij}^{(1)}\n    = \\frac{b_1}{2 b^2_0} \\gamma_i \\delta_{ij}\n    - \\frac{\\Gamma_{ij}^{(2)}}{\\left(2 \\pi b_0 + \\gamma_i - \\gamma_j\\right)}\n\\end{equation*}\nThis can be continued to obtain higher order expansions,\nfor example the $\\order{\\alpha_S}$\ncoefficient gives us an equation to find $K^{(2)}$\n\\begin{equation*}\n    4 \\pi b_0 K^{(2)} - \\comm{K^{(2)}}{\\Gamma^{(1)}}\n    = \\frac{\\pi b_1}{b_0} \\Gamma^{(1)} K^{(1)}\n    - \\Gamma^{(2)} K^{(1)}\n    + \\frac{\\pi^2 b_2}{b_0} \\Gamma^{(1)}\n    - \\frac{\\pi^2 b_1^2}{b_0^2} \\Gamma^{(1)}\n    + \\frac{\\pi b_1}{b_0} \\Gamma^{(2)}\n    - \\Gamma^{(3)}\n\\end{equation*}\nBut $K^{(1)}$ is enough for our purposes.\n\nNow that we know $K$, the only thing we need to compute to calculate\n$U$ is the ratio between $\\alpha_S(\\mu_0^2)$ and $\\alpha_S(\\mu^2)$,\nto do it we can use the beta function:\n\\begin{align*}\n    -b_0 \\log(\\frac{\\mu^2}{\\mu_0^2}) &\n    = \\int_{\\alpha_S(\\mu^2_0)}^{\\alpha_S(\\mu^2)}\n    \\frac{-2 b_0}{\\beta(\\alpha_S)} \\dd{\\alpha_S}\n    = \\int_{\\alpha_S(\\mu^2_0)}^{\\alpha_S(\\mu^2)} \\frac{b_0}{\n        \\alpha^2_S \\left(b_0 + \\alpha_S b_1\\right)\n    } \\dd{\\alpha_S}\n    + \\order{\\alpha_S} \\\\&\n    = \\left(\n        \\frac{b_1}{b_0} \\log(\\frac{1}{\\alpha_S(\\mu^2)} + \\frac{b_1}{b_0})\n        - \\frac{1}{\\alpha_S(\\mu^2)}\n    \\right)\n    - \\left(\n        \\frac{b_1}{b_0} \\log(\\frac{1}{\\alpha_S(\\mu_0^2)} + \\frac{b_1}{b_0})\n        - \\frac{1}{\\alpha_S(\\mu_0^2)}\n    \\right)\n    + \\order{\\alpha_S}\n\\end{align*}\nOne can solve for $\\alpha_S(\\mu^2)$\nin terms of the Lambert W function \\cite{Brodsky16, Karliner98},\nbut we are only interested in the ratio up to $\\order{\\alpha_S^2}$ corrections,\nso we can solve it as\n'

s = r'''
% This makes it interesting because the high mass causes a particularly strong interaction with the Higgs boson, one of the mediators in the Standard Model. Therefore it is interesting to study the interaction of the top quark with the Higgs boson with the aim of verifying the predictions of the Standard Model.
'''
assert TeXFormatter(s) == '\n%This makes it interesting because the high mass causes a particularly\n%strong interaction with the Higgs boson,\n%one of the mediators in the Standard Model.\n%Therefore it is interesting to study the\n%interaction of the top quark with the Higgs boson with the aim of\n%verifying the predictions of the Standard Model.\n'

s = r'''
\begin{equation*}
    \frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)} = 1 + b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2}) +  \alpha_S(\mu_0^2)\frac{b_1}{b_0}\log(\frac{\frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)}+ \alpha_S(\mu_0^2)\frac{b_1}{b_0}}{1+\alpha_S(\mu_0^2) \frac{b_1}{b_0}}) + \order{\alpha_S^2}
\end{equation*}
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    \\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)}\n    = 1\n    + b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2})\n    + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0} \\log(\n        \\frac{\n            \\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)}\n            + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0}\n        }{1 + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0}}\n    )\n    + \\order{\\alpha_S^2}\n\\end{equation*}\n'

s = r'''
Doing the same on equation \eqref{eq:Kmatrix}
\begin{align*}
    \dv{U(\mu, \mu_0)}{\mu} = & \dv{\alpha_S(\mu^2)}{\mu}\left(\dv{K(\alpha_S(\mu^2))}{\alpha_S}
    -K(\alpha_S(\mu^2))\frac{\Gamma^{(1)}}{2\pi\alpha_S(\mu^2) b_0}\right)\exp(-\frac{\Gamma^{(1)}}{2\pi b_0}\log(\frac{\alpha_S(\mu^2)}{\alpha_S(\mu_0^2)}))K^{-1}(\alpha_S(\mu_0^2))
\end{align*}
'''
assert TeXFormatter(s) == '\nDoing the same on equation \\eqref{eq:Kmatrix}\n\\begin{align*}\n    \\dv{U(\\mu, \\mu_0)}{\\mu}\n    = & \\dv{\\alpha_S(\\mu^2)}{\\mu} \\left(\n        \\dv{K(\\alpha_S(\\mu^2))}{\\alpha_S}\n        -K(\\alpha_S(\\mu^2)) \\frac{\\Gamma^{(1)}}{2 \\pi \\alpha_S(\\mu^2) b_0}\n    \\right) \\exp(\n        -\\frac{\\Gamma^{(1)}}{2 \\pi b_0} \\log(\n            \\frac{\\alpha_S(\\mu^2)}{\\alpha_S(\\mu_0^2)}\n        )\n    ) K^{-1}(\\alpha_S(\\mu_0^2))\n\\end{align*}\n'

s = r'''
\begin{align*}
    \tilde{g}_S\left(\frac{Q}{\bar{N}}, \mu_R\right)&
    = \frac{1}{2\pi b_0}\left[\log(1 - 2\lambda + b_0\alpha_S(\mu_R^2)\left(\log(\frac{Q^2}{\mu_R^2}) - 2\gamma_E\right)) + \alpha_S(\mu_R^2)\frac{b_1}{b_0}\frac{\log(1-2\lambda)}{1 - 2\lambda}\right] + \order{\alpha_S^2}\\&
    = \frac{1}{2\pi b_0}\left[\log(1-2\lambda) + \frac{b_0\alpha_S(\mu_R^2)}{1-2\lambda}\left(\log(\frac{Q^2}{\mu_R^2}) - 2\gamma_E\right) + \alpha_S(\mu_R^2)\frac{b_1}{b_0}\frac{\log(1-2\lambda)}{1 - 2\lambda}\right] + \order{\alpha_S^2}
\end{align*}
'''
assert TeXFormatter(s) == '\n\\begin{align*}\n    \\tilde{g}_S \\left(\\frac{Q}{\\bar{N}}, \\mu_R\\right) &\n    = \\frac{1}{2 \\pi b_0} \\left[\n        \\log(\n            1\n            - 2 \\lambda\n            + b_0 \\alpha_S(\\mu_R^2) \\left(\n                \\log(\\frac{Q^2}{\\mu_R^2}) - 2 \\gamma_E\n            \\right)\n        )\n        + \\alpha_S(\\mu_R^2) \\frac{b_1}{b_0}\n        \\frac{\\log(1 - 2 \\lambda)}{1 - 2 \\lambda}\n    \\right]\n    + \\order{\\alpha_S^2} \\\\&\n    = \\frac{1}{2 \\pi b_0} \\left[\n        \\log(1 - 2 \\lambda)\n        + \\frac{b_0 \\alpha_S(\\mu_R^2)}{1 - 2 \\lambda} \\left(\n            \\log(\\frac{Q^2}{\\mu_R^2}) - 2 \\gamma_E\n        \\right)\n        + \\alpha_S(\\mu_R^2) \\frac{b_1}{b_0}\n        \\frac{\\log(1 - 2 \\lambda)}{1 - 2 \\lambda}\n    \\right]\n    + \\order{\\alpha_S^2}\n\\end{align*}\n'

s = r'''
\begin{equation*}
    U\left(\frac{Q}{\bar{N}}, \mu_R\right)
    = \left(1 + \frac{\alpha_S(\mu_R^2)}{\pi}\frac{K^{(1)}}{1-2\lambda}\right)e^{\tilde{g}_S\left(\frac{Q}{\bar{N}}, \mu_R\right)\Gamma^{(1)}}\left(1 - \frac{\alpha_S(\mu_R^2)}{\pi}K^{(1)}\right) + \order{\alpha_S^2}
\end{equation*}
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    U \\left(\\frac{Q}{\\bar{N}}, \\mu_R\\right)\n    = \\left(\n        1 + \\frac{\\alpha_S(\\mu_R^2)}{\\pi} \\frac{K^{(1)}}{1 - 2 \\lambda}\n    \\right) e^{\n        \\tilde{g}_S \\left(\\frac{Q}{\\bar{N}}, \\mu_R\\right) \\Gamma^{(1)}\n    } \\left(1 - \\frac{\\alpha_S(\\mu_R^2)}{\\pi} K^{(1)}\\right)\n    + \\order{\\alpha_S^2}\n\\end{equation*}\n'

s = r'''
\begin{equation*}
    K_{ij}^{(1)} - \frac{1}{2\pi b_0}(K_{ij}^{(1)}\gamma_j - \gamma_iK_{ij}^{(1)})
    = \frac{b_1}{2 b^2_0}\gamma_i\delta_{ij} - \frac{1}{2\pi b_0}\Gamma_{ij}^{(2)}
\end{equation*}
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    K_{ij}^{(1)}\n    - \\frac{1}{2 \\pi b_0}(K_{ij}^{(1)} \\gamma_j - \\gamma_i K_{ij}^{(1)})\n    = \\frac{b_1}{2 b^2_0} \\gamma_i \\delta_{ij}\n    - \\frac{1}{2 \\pi b_0} \\Gamma_{ij}^{(2)}\n\\end{equation*}\n'

s = r'''
\begin{align*}
    \frac{\Gamma(\alpha_S)}{\beta(\alpha_S)} &
    = \frac{-1}{2\pi\alpha_Sb_0}\left(\Gamma^{(1)} + \left(\frac{\alpha_S}{\pi}\right)\Gamma^{(2)} + \order{\alpha_S^2}\right)\left(1 + \alpha_S\frac{b_1}{b_0} + \order{\alpha_S^2}\right)^{-1}\\&
    = \frac{-1}{2\pi \alpha_S b_0}\left(\Gamma^{(1)} + \left(\frac{\alpha_S}{\pi}\right)\Gamma^{(2)}\right)\left(1 - \alpha_S\frac{b_1}{b_0}\right) + \order{\alpha_S}\\&
    = \frac{-1}{2\pi \alpha_S b_0}\left(\Gamma^{(1)} - \alpha_S\frac{b_1}{b_0}\Gamma^{(1)} + \left(\frac{\alpha_S}{\pi}\right)\Gamma^{(2)}\right) + \order{\alpha_S}
\end{align*}
thus, comparing the $\order{1}$ coefficients in \eqref{eq:diffeqn} we obtain
\begin{equation*}
    K^{(1)} - \frac{1}{2\pi b_0}\comm{K^{(1)}}{\Gamma^{(1)}}
    = \frac{b_1}{2 b^2_0}\Gamma^{(1)} - \frac{1}{2\pi b_0}\Gamma^{(2)}
\end{equation*}
In the eigenbasis of $\Gamma^{(1)}$, we can write
'''
assert TeXFormatter(s) == '\n\\begin{align*}\n    \\frac{\\Gamma(\\alpha_S)}{\\beta(\\alpha_S)} &\n    = \\frac{-1}{2 \\pi \\alpha_S b_0} \\left(\n        \\Gamma^{(1)}\n        + \\left(\\frac{\\alpha_S}{\\pi}\\right) \\Gamma^{(2)}\n        + \\order{\\alpha_S^2}\n    \\right) \\left(\n        1 + \\alpha_S \\frac{b_1}{b_0} + \\order{\\alpha_S^2}\n    \\right)^{-1} \\\\&\n    = \\frac{-1}{2 \\pi \\alpha_S b_0} \\left(\n        \\Gamma^{(1)} + \\left(\\frac{\\alpha_S}{\\pi}\\right) \\Gamma^{(2)}\n    \\right) \\left(1 - \\alpha_S \\frac{b_1}{b_0}\\right)\n    + \\order{\\alpha_S} \\\\&\n    = \\frac{-1}{2 \\pi \\alpha_S b_0} \\left(\n        \\Gamma^{(1)}\n        - \\alpha_S \\frac{b_1}{b_0} \\Gamma^{(1)}\n        + \\left(\\frac{\\alpha_S}{\\pi}\\right) \\Gamma^{(2)}\n    \\right)\n    + \\order{\\alpha_S}\n\\end{align*}\nthus, comparing the $\\order{1}$ coefficients in \\eqref{eq:diffeqn} we obtain\n\\begin{equation*}\n    K^{(1)} - \\frac{1}{2 \\pi b_0} \\comm{K^{(1)}}{\\Gamma^{(1)}}\n    = \\frac{b_1}{2 b^2_0} \\Gamma^{(1)} - \\frac{1}{2 \\pi b_0} \\Gamma^{(2)}\n\\end{equation*}\nIn the eigenbasis of $\\Gamma^{(1)}$, we can write\n'

s = r'''
Expanding $K$ in powers of $\alpha_S$
\begin{equation*}
    K(\alpha_S) = 1 + \left(\frac{\alpha_S}{\pi}\right)K^{(1)} + \left(\frac{\alpha_S}{\pi}\right)^2K^{(2)} + \order{\alpha_S^3}
\end{equation*}
'''
assert TeXFormatter(s) == '\nExpanding $K$ in powers of $\\alpha_S$\n\\begin{equation*}\n    K(\\alpha_S)\n    = 1\n    + \\left(\\frac{\\alpha_S}{\\pi}\\right) K^{(1)}\n    + \\left(\\frac{\\alpha_S}{\\pi}\\right)^2 K^{(2)}\n    + \\order{\\alpha_S^3}\n\\end{equation*}\n'

s = r'''
\begin{equation}\label{eq:diffeqn}
    \dv{K(\alpha_S)}{\alpha_S} - \frac{1}{2\pi\alpha_S b_0}\comm{K(\alpha_S)}{\Gamma^{(1)}}
    = \left(\frac{\Gamma(\alpha_S)}{\beta(\alpha_S)} + \frac{\Gamma^{(1)}}{2\pi\alpha_Sb_0}\right)K(\alpha_S)
\end{equation}
'''
assert TeXFormatter(s) == '\n\\begin{equation} \\label{eq:diffeqn}\n    \\dv{K(\\alpha_S)}{\\alpha_S}\n    - \\frac{1}{2 \\pi \\alpha_S b_0} \\comm{K(\\alpha_S)}{\\Gamma^{(1)}}\n    = \\left(\n        \\frac{\\Gamma(\\alpha_S)}{\\beta(\\alpha_S)}\n        + \\frac{\\Gamma^{(1)}}{2 \\pi \\alpha_S b_0}\n    \\right) K(\\alpha_S)\n\\end{equation}\n'

s = r'''
\begin{equation*}
    \dv{U(\mu, \mu_0)}{\mu} = \dv{\alpha_S(\mu^2)}{\mu}\frac{\Gamma(\alpha_S(\mu^2))}{\beta(\alpha_S(\mu^2))}\Pexp{\int_{\alpha_S(\mu^2_0)}^{\alpha_S(\mu^2)} \frac{\Gamma(\alpha_S)}{\beta(\alpha_S)}\dd{\alpha_S}}
    = \dv{\alpha_S(\mu^2)}{\mu}\frac{\Gamma(\alpha_S(\mu^2))}{\beta(\alpha_S(\mu^2))}U(\mu, \mu_0)
\end{equation*}
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    \\dv{U(\\mu, \\mu_0)}{\\mu}\n    = \\dv{\\alpha_S(\\mu^2)}{\\mu}\n    \\frac{\\Gamma(\\alpha_S(\\mu^2))}{\\beta(\\alpha_S(\\mu^2))} \\Pexp{\n        \\int_{\\alpha_S(\\mu^2_0)}^{\\alpha_S(\\mu^2)}\n        \\frac{\\Gamma(\\alpha_S)}{\\beta(\\alpha_S)} \\dd{\\alpha_S}\n    }\n    = \\dv{\\alpha_S(\\mu^2)}{\\mu}\n    \\frac{\\Gamma(\\alpha_S(\\mu^2))}{\\beta(\\alpha_S(\\mu^2))} U(\\mu, \\mu_0)\n\\end{equation*}\n'

s = r'''
In order to evolve the soft matrix $S$ from a scale $\mu_0$ to a soft scale
$\mu$ where we can use the Eikonal approximation we need to use the RGE,
whose solution depends on a matrix $U$ defined in \cite{Daniel} as
\begin{equation} \label{eq:pathorder}
    U(\mu, \mu_0)
    = \Pexp{
        \int_{\mu_0}^\mu \Gamma(\alpha_S(\tilde{\mu}^2))
        \frac{\dd{\tilde{\mu}}}{\tilde{\mu}}
    }
    = \Pexp{
        \int_{\alpha_S(\mu^2_0)}^{\alpha_S(\mu^2)}
        \frac{\Gamma(\alpha_S)}{\beta(\alpha_S)} \dd{\alpha_S}
    }
\end{equation}
where we define $\beta(\alpha_S)$ by the convention
\begin{equation*}
    \beta(\alpha_S)
    = \dv{\alpha_S(\mu^2)}{\log(\mu)}
    = \mu \dv{\alpha_S(\mu^2)}{\mu}
\end{equation*}
We only need the matrix $U$ up to
$\order{\alpha_S^2}$ and we know that $\Gamma$ and $\beta$ can be expanded as
\begin{equation} \label{eq:alphaexpI}
    \Gamma(\alpha_S)
    = \left(\frac{\alpha_S}{\pi}\right) \Gamma^{(1)}
    + \left(\frac{\alpha_S}{\pi}\right)^2 \Gamma^{(2)}
    + \order{\alpha_S^3},
    \qquad
    \beta(\alpha_S)
    = - 2 \alpha^2_S \left(b_0 + \alpha_S b_1 + \order{\alpha_S^2}\right)
\end{equation}
This means a straightforward expansion of $U$ must be done carefully,
since the leading terms behaves like $1 / \alpha_S$.
To simplify the expansion is thus advisable to ``factor'' this first term,
writing the matrix $U$ as:
\begin{equation} \label{eq:Kmatrix}
    U(\mu, \mu_0)
    = K(\alpha_S(\mu^2)) \exp(
        - \frac{\Gamma^{(1)}}{2 \pi b_0} \log(
            \frac{\alpha_S(\mu^2)}{\alpha_S(\mu_0^2)}
        )
    ) K^{-1}(\alpha_S(\mu_0^2))
\end{equation}
where $K$ is a matrix to be determined.
'''
assert TeXFormatter(s) == "String not modified"

s = r'''
Following the results from \cite[Section VIII]{Buras79}, \cite[Appendix A]{Buras91}, \cite[Section III.F.1]{Buras95}, \cite[Appendix A]{Neubert11}, we can start by differentiating $U$ with respect to $\mu$.
'''
assert TeXFormatter(s) == '\nFollowing the results from \\cite[Section VIII]{Buras79},\n\\cite[Appendix A]{Buras91}, \\cite[Section III.F.1]{Buras95},\n\\cite[Appendix A]{Neubert11},\nwe can start by differentiating $U$ with respect to $\\mu$.\n'

s = r'''
\begin{equation}\label{eq:LBKshifts}
    \savg{\mathcal{A}(p, k)} = -\left(\sum_{ij=1}^{n}\frac{(\eta_i Q_i p_i) \cdot (\eta_j Q_j p_j)}{(p_i \cdot k) (p_j \cdot k)}\right)\savg{\mathcal{H}(p+\delta p)}~,
\end{equation}
'''
assert TeXFormatter(s) == '\n\\begin{equation} \\label{eq:LBKshifts}\n    \\savg{\\mathcal{A}(p, k)}\n    = - \\left(\n        \\sum_{ij = 1}^{n} \\frac{\n            (\\eta_i Q_i p_i) \\cdot (\\eta_j Q_j p_j)\n        }{(p_i \\cdot k) (p_j \\cdot k)}\n    \\right) \\savg{\\mathcal{H}(p + \\delta p)}~,\n\\end{equation}\n'
