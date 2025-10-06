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

# TODO:
#   - Fix phantoms
#   - Fix aligned blocks
#   - Format also missing {} after _ and ^?
#   - Change [b|B]ig? to the l/r versions (e.g big -> bigl or bigr)
#   - Change the =& align to ={}& for correct space.
#   - Have a limit for non-comments 1 character smaller than commented ones.

LINE_LENGTH = 80

separate_list = [r'\\quad', r'\\qquad']
arrow_list = [r'\\to', r'\\xrightarrow(\[.*?\])?\{.*?\}', r'\\Longrightarrow']
equal_list = ['=', r'\\equiv', r'\\cong', r'\\neq', r'\\geq', r'\\leq']
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
    tuple[tuple[int | None, int] | tuple[int, int | None], str],
    'ParenthesisType'
]


class Parenthesis():
    OPEN_PARENTHESIS = '([{'
    CLOSE_PARENTHESIS = ')]}'
    unmatched_parenthesis: str
    levels: list[tuple[int, str]]

    def __init__(self) -> None:
        self.current_struct: ParenthesisType = {}
        self.parenthesis_structure: ParenthesisType = {}

    def add_open_brace(self, idx: int, char: str) -> None:
        self.levels.append((idx, char))
        self.current_struct[((idx, None), char)] = {}
        self.current_struct = self.current_struct[((idx, None), char)]

    def close_parenthesis(self, idx: int, char: str) -> None:
        if self.levels:
            # Check if this closes the previous open parenthesis
            start, schar = self.levels.pop()
            if schar == '$':
                valid = self.process_end_equation(char, idx, start)
                if not valid:
                    self.levels.append((start, '$'))
                    return
            elif not self.match(char, schar):
                # PARENTHESIS NOT MATCHING!
                # Add previous level back untouched and process this
                # parenthesis in a special way.
                self.levels.append((start, schar))
                start, schar = self.process_not_match(idx, char, start)
                if not schar:
                    return
            elif char == '}':
                if self.is_escaped(idx):
                    idx -= 1
        else:
            if self.unmatched_parenthesis:
                escaped = self.process_unmatched(char)
                if escaped:
                    return
                self.unmatched_parenthesis = self.unmatched_parenthesis[:-1]
            # Why we need the escape ")"?
            elif char == ')' or char == 'END':
                return
            elif char == '}':
                if self.is_escaped(idx):
                    idx -= 1
            schar = self.get_match(char)
            self.parenthesis_structure = {
                ((None, idx), schar): self.parenthesis_structure}
            self.current_struct = self.parenthesis_structure
            return
        self.current_struct = self.update_structure(start, schar, idx)

    @classmethod
    def get_match(cls, char: str) -> str:
        if char in cls.OPEN_PARENTHESIS:
            return cls.CLOSE_PARENTHESIS[cls.OPEN_PARENTHESIS.index(char)]
        elif char in cls.CLOSE_PARENTHESIS:
            return cls.OPEN_PARENTHESIS[cls.CLOSE_PARENTHESIS.index(char)]
        elif char == '$':
            return char
        elif char == 'BEGIN':
            return 'END'
        raise ValueError(f'char not valid: {char}')

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
            # Assume that ) is not part of a parenthesis. Just ignore this and
            # continue parsing the parenthesis
            return start, ''
        # Check whether we are in a phantom context.
        if 'p' in self.unmatched_parenthesis:
            # We are in a phantom context open in a previous line.
            if char != '}':
                # Ignore error
                return start, ''
            # We are closing the phantom from the previous line.
            # TODO: The closing parentheis need not be closing the phantom.
            # I should add a check and decide what to do in that case.
            start, schar = self.levels[0]
            self.parenthesis_structure.pop(((start, None), schar))
            self.current_struct = self.parenthesis_structure
            self.parenthesis_structure = {((None, idx), '{'):
                                          self.current_struct}
            self.levels = []
            return start, ''
        while self.levels:
            # Ignore all levels that are not "{" type.
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

    def process_unmatched(self, char: str) -> bool:
        schar = self.unmatched_parenthesis[-1]
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
        self.unmatched_parenthesis = unmatched_parenthesis
        self.in_equation = '$' in unmatched_parenthesis
        self.parenthesis_structure = {}
        self.levels: list[tuple[int, str]] = []
        self.current_struct = self.parenthesis_structure

        valid_str = '\\$' + self.OPEN_PARENTHESIS + self.CLOSE_PARENTHESIS
        for idx, char in ((i, c) for i, c in enumerate(line)
                          if c in valid_str):
            if char == '\\':
                if line[idx+1:idx+7] == 'begin{':
                    self.add_open_brace(idx + 6, 'BEGIN')
                elif line[idx+1:idx+5] == 'end{':
                    self.close_parenthesis(idx, 'END')
            elif char in self.OPEN_PARENTHESIS:
                self.add_open_brace(idx, char)

            elif char == '$' and not self.in_equation:
                if self.is_escaped(idx):
                    continue
                self.add_open_brace(idx, '$')
                self.in_equation = True

            elif char in self.CLOSE_PARENTHESIS + '$':
                self.close_parenthesis(idx, char)
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
    eq_envs = ('equation', 'align', 'eqnarray', 'gather')
    txt_envs = ('document', 'figure', 'itemize')

    def __init__(self, content: Union[str, list[str]], *, reset: bool = False
                 ) -> None:
        if isinstance(content, str):
            content = content.splitlines(keepends=True)
        self.init_string = content.copy()
        if reset:
            content = [' '.join([line.strip() for line in content]).strip()]
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
        if r'\begin' in line:
            if any(word in line for word in self.eq_envs):
                self._context.append('equation')
            elif any(word in line for word in self.txt_envs):
                self._context.append('text')
            elif 'verbatim' in line:
                self._context.append('verbatim')
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
        #TODO: Format also missing {} after _ and ^?
        for i, line in enumerate(lines):
            # Replate all tabs by spaces
            line = line.replace('\t', '    ')
            # Calculate the indent of the line, remove spaces in the beginning
            # of the line.
            if line.lstrip().startswith('%%% '):
                # Emacs local variable definition.
                indent = 0
                cmt = '%%% '
            elif line.lstrip().startswith('%%'):
                # Long comment. Keep it as is
                continue
            elif line.lstrip().startswith('%'):
                # Line is commented code
                indent = (len(line[1:]) - len(line.lstrip(' %'))) // 4 * 4
                cmt = '%'
            else:
                indent = (len(line) - len(line.lstrip())) // 4 * 4
                cmt = ''
            self.indent = cmt + ' ' * indent
            self.update_context(line)
            if self.context == 'verbatim':
                continue
            line = self.indent + line.lstrip(' %')
            # Remove double spaces (except for the indent)
            while '  ' in line[indent:]:
                line = self.indent + line.lstrip(' %').replace('  ', ' ')
            # Make sure all the commas are followed by a space, except for ,~
            # and footnotes
            line = re.sub(r',(?!\s|~|\\footnote)', r', ', line)
            # Move "begin" commands to a new line.
            # TODO: The check for "def" commands should be better.
            # if (r'\begin' in line.strip(' %')[6:]
            #         and all(x not in line for x in def_list)):
            #     idx = line.index(r'\begin')
            #     if not ((match := re.search(r'(?<!\\)%', line))
            #             and match.start() < idx):
            #         new_line = self.indent + line[idx:]
            #         line = line[:idx]
            #         lines.insert(i+1, new_line)
            offset = len(self.indent)
            self.indent = ''
            if self.context == 'equation':
                add_space = self._equation_addspace(line.lstrip(' %'), offset)
            elif self.context == 'text':
                add_space = self._text_addspace(line.lstrip(' %'), offset)
            elif self.context == 'phantom':
                # Ignore indent inside phantom
                line = line.strip()
                add_space = self._phantom_addspace(line.lstrip(' %'), offset)
            # Add all the spaces found previously
            for space_pos in sorted(set(add_space), reverse=True):
                line = line[:space_pos] + ' ' + line[space_pos:]
            lines[i] = line.rstrip() + '\n'
        self.reset_context()
        return lines

    def _equation_addspace(self, line: str, offset: int = 0) -> list[int]:
        # TODO: Add space in between parenthesis ")("?
        # Find position that need space
        add_space = []
        self.indent = ' ' * (len(line) - len(line.lstrip(' %')))
        skeleton, parenthesis = self.get_skeleton(line,
                                                  self.multline_parenthesis)
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
            for match in re.finditer(r'&[^\s]', skeleton):
                add_space.append(self.get_index_line(match.start(), line) + 1)
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
            if start is None and self.multline_parenthesis:
                assert end is not None
                start = -1
                if char == '$':
                    assert skeleton[0] == '$', line
                    # Equation was opened in a previous line.
                    # 1. Close equation context
                    # 2. Remove spaces, they were determined assuming an
                    # equation enviroment.
                    self._context.pop()
                    # Ensure that there is a space after "$"
                    add_space = []
                    end2 = end + 2
                    if line[end+1] != ' ':
                        end2 = end + 1
                        add_space = [end2]
                add_space.extend(self._equation_addspace(line[:end]))
                self.multline_parenthesis = self.multline_parenthesis[:-1]
                if char == '$':
                    # 3. Compute the correct parenthesis for a text
                    # environment. This needs to be done after
                    # multline_parenthesis has been updated.
                    # 4. We can ignore further parenthesis, they should be
                    # handled in the _text_addspace call.
                    add_space.extend(self._text_addspace(line[end2:], end2))
                    break
                continue
            assert start is not None
            if end is None:
                if char in ('BEGIN'):
                    self.multline_parenthesis += '{'
                else:
                    self.multline_parenthesis += char[0]
            if start < 1:
                self.indent = ''
            if char == '{':
                if pattern_context.search(line[:start]):
                    add_space.extend(self._text_addspace(line[start+1:end],
                                                         start+1))
                    continue
                elif line[start-8:start] == r'\phantom':
                    # We are entering a phantom environment. If it is contained
                    # within this line, just ignore the inside. If is spans
                    # multiple lines, modify the context and multline
                    # accordingly. The call to _phantom_addspace is to add
                    # other opening multiline parenthesis. This is needed to
                    # know when the phantom environment is being closed.
                    if end is None:
                        self._context.append('phantom')
                        self.multline_parenthesis = (
                            self.multline_parenthesis[:-1] + 'p{'
                        )
                        add_space.extend(
                            self._phantom_addspace(line[start+1:end], start+1)
                        )
                    continue
            add_space.extend(self._equation_addspace(line[start+1:end],
                                                     start+1))
        return [offset + n for n in add_space]

    def _text_addspace(self, line: str, offset: int = 0) -> list[int]:
        add_space = []
        parenthesis_stack = [
            self._find_parenthesis(line, self.multline_parenthesis)
        ].copy()
        while parenthesis_stack:
            parenthesis = parenthesis_stack.pop()
            for ((start, end), char), par in parenthesis.items():
                if char == '$':
                    assert start is not None, line
                    if end is None:
                        self.multline_parenthesis += char
                        self._context.append('equation')
                    add_space.extend(
                        self._equation_addspace(line[start+1:end],
                                                offset=start+1)
                    )
                elif char == 'BEGIN':
                    key = list(par)[0]
                    # par.pop(key)
                    senv, eenv = key[0]
                    assert senv is not None
                    assert eenv is not None
                    env = line[senv+1:eenv].strip('*')
                    if env in self.eq_envs:
                        add_space.extend(
                            self._equation_addspace(line[eenv+1:end],
                                                    offset=eenv+1)
                        )
                        if end is None:
                            self._context.append('equation')
                    else:
                        parenthesis_stack.append(par)
                else:
                    parenthesis_stack.append(par)
        return [offset + n for n in add_space]

    def _phantom_addspace(self, line: str, offset: int) -> list[int]:
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for p_key in parenthesis:
            (start, end), char = p_key
            if start is None:
                assert end is not None
                p = parenthesis[p_key]
                chars = [char]
                while p:
                    for (start, end2), char in p:
                        if start is None:
                            break
                    assert end2 is not None
                    chars.append(char)
                    p = p[(start, end2), char]
                for char in reversed(chars):
                    if char == self.multline_parenthesis[-1]:
                        self.multline_parenthesis =\
                            self.multline_parenthesis[:-1]
                    if self.multline_parenthesis[-1] == 'p':
                        self.multline_parenthesis =\
                            self.multline_parenthesis[:-1]
                        self._context.pop()
                        return self._equation_addspace(line[end+1:],
                                                       offset+end+1)
            if end is None:
                assert start is not None
                if char in ('BEGIN'):
                    self.multline_parenthesis += '{'
                else:
                    self.multline_parenthesis += char[0]
                return self._phantom_addspace(line[start+1:], offset+start+1)
        return []

    def _format_spaces_operation(self, line: str, offset: int = 0
                                 ) -> list[int]:
        add_space = []
        # Add a space before an operation, unless preceeded
        # by a parenthesis or exponent (^) or underscore (_)
        for match in re.finditer(r'([^\s\(\[\{\^_])[\+\-/=\<\>]', line):
            if not match.group(1) == ' ':
                add_space.append(offset + match.start(1) + 1)
            else:
                assert False
        # Add a space after an operation if not preceded by parenthesis
        # or followed by EOL.
        for match in re.finditer(r'[^\{\(\[\^_](\+|/|=|\\neq|\-|\<)(?!\s|$)',
                                 line):
            # Do not add space between underscore and superscore.
            if {line[match.start()], line[match.end()]} == {'^', '_'}:
                continue
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
                new_content.append(line[:LINE_LENGTH] + '\n')
                continue
            # TODO: Compute indent from scratch from previous lines.
            if line.strip().startswith('%'):
                level = len(line) - len(line.lstrip(' %'))
                self.indent = '%' + ' ' * (level - 1)
            elif (match := re.search(r'(?<!\\)%(?!$)', line)):
                self.commentafter = match.start()
            # Handle equation in the middle of the text
            if line.replace('$$', '$').count('$') % 2 == 1:
                if self.context == 'text':
                    line_wo_comment = line.partition('%')[0].strip()
                    if not line_wo_comment:
                        # Line is a comment
                        line_wo_comment = line.partition('%')[-1].strip()
                    if (('$' not in self.multline_parenthesis
                         or line_wo_comment.strip()[0] != '$')
                            and line_wo_comment[-1] != '$'):
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
                    if idx > len(self.indent):
                        new_content.extend(self.format_tex([line[:idx]]))
                    self._context.pop()
                    # if '$' in self.multline_parenthesis:
                    #     assert self.multline_parenthesis[-1] == '$'
                    #     self.multline_parenthesis = self.multline_parenthesis[:-1]
                    new_content.extend(self.format_tex([self.indent
                                                        + line[idx:]]))
                    continue
            # If line is short enough, leave it as it is
            if len(line) <= LINE_LENGTH or self.context == 'verbatim':
                if not first and self.context == 'equation':
                    # TODO: If previous lines were splitted, check that all
                    # operations here have lower priority. For example, if the
                    # previous line splitted sums, and this line contains some
                    # sums, we should split them also.

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
                else:
                    raise NotImplementedError()
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
            valid_comb = comb_len[(comb_len <= LINE_LENGTH) & index_mask]
            if not valid_comb.size:
                break
            idx = np.where((comb_len == min(valid_comb)) & index_mask)[0][0]
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
                and first.strip()[-1] != ','
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
        if self.context == 'phantom':
            return True
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
        if r'\begin{' in first or r'\begin' in second:
            return False
        if r'\end{' in first or r'\end{' in second:
            return False
        _first = first.endswith
        _second = second.startswith
        if _first('&') or _first('\\'):
            return False
        if _second('+') or _second('-') or _second(r'\pm') or _second(r'\mp'):
            return False
        if _first('(') or _first('[') or _first('{'):
            return False
        if re.match(r'(\\(left|right|[bB]igg?))?(\)|\]|}|\\})', second):
            return False
        # if _first('\\left(') or _first('\\left[') or _first('\\left\\{'):
        #     return False
        # if _second('\\right)') or _second('\\right]') or _second('\\right\\}'):
        #     return False
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
            if end == len(line) or line[end] == ' ' or line[end-1] == ' ':
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
        new_lines = [self.indent + s.lstrip(' %').rstrip() + '\n'
                     for s in lines]
        return self.format_tex(new_lines)

    def split_phantom(self, line: str) -> list[str]:
        pattern = re.compile(r' ')
        lines = []
        prev_idx = 0
        for match in pattern.finditer(line):
            lines.append(line[prev_idx:match.start()])
            prev_idx = match.end()
        lines[0] = r'\phantom{' + lines[0]
        lines.append(line[prev_idx:] + '}')
        lines = [line for line in lines if line.strip(' %')]
        new_lines = [self.indent + s.lstrip(' %').rstrip() + '\n'
                     for s in lines]
        self._context.append('phantom')
        ret = self.format_tex(new_lines)
        self._context.pop()
        return ret

    def get_index_line(self, idx: int, line: str) -> int:
        idx_l = len(self.indent) + idx
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), char in parenthesis:
            if start is None:
                assert end is not None
                idx_l = end + idx
                continue
            if start >= idx_l:
                break
            if char == 'BEGIN':
                if idx_l == start + 1:
                    (start, end), _ = list(parenthesis[(start, end), char])[0]
                    assert start is not None
                    assert end is not None
                    idx_l += end - start - 1
                    break
                idx_l -= 1
            assert end is not None, line
            idx_l += end - start - 1
        return idx_l

    def _format_text(self, line: str) -> list[str]:
        skeleton, parenthesis = self.get_skeleton(line,
                                                  self.multline_parenthesis)
        if self.commentafter < len(line):
            new_lines = [line[:self.commentafter+1], line[self.commentafter:]]
            return self.format_tex(new_lines)
        if (idx := skeleton.find(r'\begin{}\end{}')) >= 0:
            if skeleton == r'\begin{}\end{}':
                sidx = self.get_index_line(7, line) + 1
                eidx = self.get_index_line(8, line)
                new_lines = [line[:sidx],
                             self.indent + 4*' ' + line[sidx:eidx].strip(),
                             self.indent + line[eidx:].strip()]
                new_lines = [line.rstrip() + '\n'
                             for line in new_lines if line]
            else:
                sidx = self.get_index_line(idx, line)
                eidx = self.get_index_line(idx + 13, line)
                new_lines = [line[:sidx], line[sidx:eidx+1], line[eidx+1:]]
                new_lines = [self.indent + line.strip() + '\n'
                             for line in new_lines if line]
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
        pattern = re.compile(r'\$\$[\.\,]?')
        if pattern.match(skeleton):
            start = self.get_index_line(0, line)
            end = self.get_index_line(1, line)
            # Add line before first '$'
            new_lines.append(line[:start].rstrip() + '\n')
            # Format the formula inside '$$'
            self._context.append('equation')
            indent = self.indent
            new_lines.extend(self.format_tex(
                [indent + 4 * ' ' + line[start+1:end].lstrip()]
            ))
            self._context.pop()
            # Add text after last '$'
            new_lines.append(indent + line[end+1:].strip() + '\n')
            # Add '$$':
            if len(new_lines) == 3 and len(new_lines[1]) < LINE_LENGTH:
                # The equation fits in a single line
                new_lines[1] = f"{indent}${new_lines[1].strip()}$\n"
            else:
                if new_lines[0] == '\n':
                    new_lines[0] = f"{indent}$\n"
                else:
                    new_lines[0] = new_lines[0].rstrip() + '$\n'
                new_lines[-1] = f"{indent}${new_lines[-1].lstrip()}"
            # TODO?: Add % after last $.
            new_lines = [line for line in new_lines if line.strip()]
            return self.format_tex(new_lines)
        pattern = re.compile(r'\s\$\$[\s\.\,]\s?')
        if pattern.search(skeleton):
            return self.line_split(line, pattern, keep=True)
        # Split {} into multiple lines
        for (_start, _end), char in parenthesis:
            start = _start+1 if _start is not None else 0
            end = _end if _end is not None else len(line)
            if end - start > int(0.51*LINE_LENGTH) and char == '{':
                pass
            elif end - start > int(0.95*LINE_LENGTH) and char in '([':
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
        # Split label and phantom into own line
        if (r'\label' in skeleton and skeleton != r'\label{}'
                or r'\phantom' in skeleton and skeleton != r'\phantom{}'):
            return self.line_split(line, r'\\(label|phantom){.*?}', keep=True)
        if skeleton == r'\phantom{}':
            return self.split_phantom(line.strip()[9:-1])
        # If equation separator (quad), split line.
        if pattern_separate.search(skeleton):
            return self.line_split(line, pattern_separate, keep=True)
        # Split line in implication
        if pattern_arrow.search(skeleton):
            return self.line_split(line, pattern_arrow, keep=True)
        # Split line in equality
        if pattern_equal.search(skeleton[1:]):
            return self.line_split(line, pattern_equal, keep='second')
        if r'\\' in skeleton[:-2]:
            return self.line_split(line, r'\\\\\s*\&?', keep='first')
        # If unmatched parenthesis, split right after/before.
        ret = self.check_unmatched_parenthesis(line)
        if ret is not None:
            return ret
        # Split commas
        if re.search(r'[^\\];', skeleton[:-1]):
            return self.line_split(line, r'[^\\];', keep='first')
        if re.search(r'[^\\],', skeleton[:-1]):
            return self.line_split(line, r'[^\\],', keep='first')
        # Split sums and subtractions into multiple lines
        ret = self.split_sums(line)
        if ret is not None:
            return ret
        # Split parenthesis into multiple lines if they are big enough.
        ret = self.split_long_parenthesis(line)
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

    def split_sums(self, line: str) -> list[str] | None:
        skeleton, _ = self.get_skeleton(line, self.multline_parenthesis)
        new_lines = []
        prev_idx = 0
        # TODO: Handle cases like \cong - 3, etc.
        for match in re.finditer(r'[^=\s\^_]\s*(\+|\-|\\pm|\\mp)', skeleton):
            idx_s = match.start(1)
            idx_l = self.get_index_line(idx_s, line)
            new_lines.append(self.indent + line[prev_idx:idx_l].lstrip(' %'))
            prev_idx = idx_l
        if new_lines:
            new_lines.append(self.indent + line[idx_l:].lstrip())
            return self.format_tex(new_lines)
        return None

    def check_unmatched_parenthesis(self, line: str) -> list[str] | None:
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), char in parenthesis:
            if start is None and end > len(self.indent):  # type: ignore[operator]
                if (match := re.search(r'\\((right|[bB]igg?)\s?\\?)?$',
                                       line[:end])):
                    end = match.start()
                    if end == len(self.indent):
                        continue
                new_lines = [
                    self.indent + 4*' ' + line[:end].strip('% ') + '\n',
                    self.indent + line[end:].rstrip() + '\n'
                ]
            elif (end is None and char != 'BEGIN'
                  and start + 2 < len(line.rstrip(' \n&'))):  # type: ignore[operator]
                assert start is not None
                new_lines = [
                    line[:start+1].rstrip() + '\n',
                    self.indent + 4*' ' + line[start+1:].strip() + '\n'
                ]
            else:
                continue
            return self.format_tex(new_lines)
        return None

    def split_long_parenthesis(self, line: str) -> list[str] | None:
        # ToDo: What should I do with broken left and right parenthesis?
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        maxlength = int(0.38 * LINE_LENGTH)
        split_start, split_end = None, None

        # Find the longest parenthesis that exceeds the threshold
        for (start, end), _ in parenthesis:
            if end is None or start is None or end - start < maxlength:
                continue
            split_start, split_end = start, end
            maxlength = end - start
        # If no long parenthesis exists, return None
        if split_start is None:
            return None
        # Check if there is a label or text context and handle it as text.
        if pattern_context.search(line[:split_start]):
            indent = self.indent
            new_lines = [line[:split_start + 1] + '%\n']
            self._context.append('text')
            new_lines += self.format_tex(
                [self.indent + 4*' '
                 + line[split_start+1:split_end].lstrip() + '%\n']
            )
            self._context.pop()
            new_lines += [indent + line[split_end:] + '\n']
            return self.format_tex(new_lines)
        # Ajust the end for large parenthesis
        if (match := re.search(r'\\(right|[bB]igg?)\s?\\?$',
                               line[:split_end])):
            split_end = match.start()
        if (match := re.match(r'\(\s*\&', line[split_start:])):
            split_start += match.end() - 1

        new_lines = [
            line[:split_start + 1] + '\n',
            self.indent + 4*' '
            + line[split_start+1:split_end].lstrip() + '\n',
            self.indent + line[split_end:] + '\n'
        ]
        return self.format_tex(new_lines)

    def split_after_parenthesis(self, line: str) -> list[str] | None:
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), char in reversed(parenthesis):
            # Why can we be sure end is not None here?
            assert end is not None
            if end > LINE_LENGTH:
                continue
            elif char == '{' and end + 1 < len(line) and line[end+1] == '{':
                continue
            new_lines = [line[:end + 1] + '\n',
                         self.indent + line[end+1:].strip() + '\n']
            return self.format_tex(new_lines)
        return None

    def update_multiline_parenthesis(self, line: str) -> None:
        open_p = '([{'
        close_p = ')]}'
        parenthesis = self._find_parenthesis(line,
                                             self.multline_parenthesis).copy()
        new_parenthesis: list[tuple[int, str]] = []
        while parenthesis:
            ((start, end), char), child = parenthesis.popitem()
            if start is None:
                assert end is not None
                new_parenthesis.append((end, Parenthesis.get_match(char)))
            elif end is None:
                new_parenthesis.append((start, char))
            else:
                continue
            parenthesis |= child
        sent = True
        for start, char in sorted(new_parenthesis):
            if char in close_p:
                assert sent, line
                assert self.multline_parenthesis, line
                _char = self.multline_parenthesis[-1]
                assert open_p.index(_char) == close_p.index(char), line
                self.multline_parenthesis = self.multline_parenthesis[:-1]
                if (self.multline_parenthesis
                        and self.multline_parenthesis[-1] == 'p'):
                    assert self.context == 'phantom'
                    self._context.pop()
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
                if line[start-8:start] == r'\phantom':
                    self.multline_parenthesis += 'p'
                    self._context.append('phantom')
                self.multline_parenthesis += char

    @classmethod
    @functools.cache
    def get_skeleton(cls, line: str, unmatched_parenthesis: str = ''
                     ) -> tuple[str, ParenthesisType]:
        parenthesis = cls._find_parenthesis(line, unmatched_parenthesis)

        skeleton = line.lstrip(' %')
        offset = len(line) - len(skeleton)

        for (start, end), char in parenthesis:
            if end is None:
                assert start is not None
                skeleton = skeleton[:start-offset+1]
                break

            elif start is None:
                skeleton = skeleton[end-offset:]
                offset = end

            elif char == 'BEGIN':
                (sidx, eid), _ = list(parenthesis[(start, end), char])[0]
                skeleton = (
                    skeleton[:start-offset+1] + '}' + skeleton[end-offset:]
                )
                offset += end - start - 2

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


def format_files(*names: str | pathlib.PosixPath) -> None:
    pathdir = pathlib.Path.cwd()
    for name in names:
        filename = pathdir / pathlib.Path(name)
        print(filename)
        dest = pathdir / pathlib.Path(name)
        with filename.open() as file:
            content = file.readlines()
        # import pdb; pdb.set_trace()
        try:
            new_content = TeXFormatter(content).formatted_lines
        except Exception as e:
            err_msg = 'Exception encountered when formatting the file '
            err_msg += f'{filename}: {e}'
            raise type(e)(err_msg).with_traceback(e.__traceback__) from None
        with dest.open('w') as file:
            file.writelines(new_content)


def format_string() -> None:
    s = r'''
'''
    r = TeXFormatter(s, reset=False)
    print(r)


if __name__ == '__main__':
    try:
        __IPYTHON__  # type: ignore
        format_string()
    except NameError:
        _, *names = sys.argv
        format_files(*names)
