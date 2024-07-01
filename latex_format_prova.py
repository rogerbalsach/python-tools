#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:42:48 2023

@author: Roger Balsach
"""

# import pathlib
import functools
import re

import numpy as np


class TeXFormatter:
    # TODO: Manage comments
    # TODO: Manage text environments inside equations
    def __init__(self, content):
        if isinstance(content, str):
            content = content.splitlines(keepends=True)
        self.init_string = content.copy()
        self.reset_context()
        self.multline_parenthesis = ''
        format_content = self._format_spaces(content)
        self.indent = ''
        self.formatted_lines = self.format_tex(format_content, first=True)

    @property
    def context(self):
        return self._context[-1]

    def update_context(self, line):
        if '\\begin' in line:
            if 'equation' in line or 'align' in line:
                self._context.append('equation')
            elif 'document' in line or 'figure' in line:
                self._context.append('text')
            else:
                from warnings import warn
                warn(f'unknown environment: {line}')
                self._context.append(self._context[-1])
        if '\\end' in line:
            self._context.pop()

    def reset_context(self):
        self._context = ['text']

    def _format_spaces(self, lines):
        for i, line in enumerate(lines):
            self.update_context(line)
            # Replate all tabs by spaces
            line = line.expandtabs(4)
            # Calculate the indent of the line, remove spaces in the beginning
            # of the line.
            if line.startswith('%'):
                indent = (len(line[1:]) - len(line.lstrip(' %'))) // 4 * 4
                line = '%' + ' ' * indent + line[1:].lstrip()
            else:
                indent = (len(line) - len(line.lstrip())) // 4 * 4
                line = ' ' * indent + line.lstrip()
            # Remove double spaces (except for the indent)
            while '  ' in line[indent:]:
                line = ' ' * indent + line.lstrip().replace('  ', ' ')
            # Make sure all the commas are followed by a space.
            line = line.replace(', ', ',').replace(',', ', ')
            if self.context == 'equation':
                add_space = self._format_spaces_addspace(line)
            elif self.context == 'text':
                add_space = []
                list_parenthesis = [
                    self._find_parenthesis(line, self.multline_parenthesis)
                ]
                while list_parenthesis:
                    parenthesis = list_parenthesis.pop()
                    for (start, end), char in parenthesis:
                        if char != '$':
                            list_parenthesis.append(
                                parenthesis[(start, end), char]
                            )
                            continue
                        add_space.extend(
                            self._format_spaces_addspace(
                                line[start+1:end], offset=start+1
                            )
                        )
            # Add all the spaces found previously
            for space_pos in sorted(set(add_space), reverse=True):
                line = line[:space_pos] + ' ' + line[space_pos:]
            lines[i] = line.rstrip() + '\n'
        self.reset_context()
        return lines

    def _format_spaces_addspace(self, line, offset=0):
        # Find position that need space
        add_space = []
        # Add a space before '\' except when following ( [ { $ ^ \ or a space
        # or except when prefacing "right".
        backslash_match = re.compile(r'[^\(\[\{\s\$\^\\\-\|\_]\\(?!right)')
        if '\\' in line:
            for match in backslash_match.finditer(line):
                add_space.append(offset + match.start(0) + 1)
        # Add a space before '&' except when following \ or a space.
        if '&' in line:
            for match in re.finditer(r'[^\\\s]&', line):
                add_space.append(offset + match.start(0) + 1)
        # Add a space before and after the + - and / operations.
        if '+' in line or '-' in line or '/' in line or '=' in line:
            add_space.extend(self._format_spaces_operation(line, offset))
        # Add a space after ).
        if ')' in line or ']' in line or '}' in line:
            for match in re.finditer(r'[\)\]\}][A-Za-z0-9]', line):
                add_space.append(offset + match.start(0) + 1)
        # Add a space after super and underscript.
        if '_' in line or '^' in line:
            for match in re.finditer(r'[_^]\w[A-Za-z0-9]', line):
                add_space.append(offset + match.end() - 1)
        # Add a space after digits.
        for match in re.finditer(r'\d[A-Za-z]', line):
            add_space.append(offset + match.end() - 1)
        return add_space

    def _format_spaces_operation(self, line, offset=0):
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
        for match in re.finditer(r'[^\{\(\[][\+/=\-\<](?!\s|$)', line):
            add_space.append(offset + match.end())
        return add_space

    def format_tex(self, lines, first=False):
        new_content = []
        for line in map(str.rstrip, lines):
            # Detect when we are inside an environment
            self.update_context(line)
            # Compute the indent of the line
            if line.startswith('%'):
                self.indent = '%'
                level = (len(line[1:]) - len(line.lstrip(' %')))
                self.indent += ' ' * level
            else:
                self.indent = ' ' * (len(line) - len(line.lstrip()))
            # If line is shoft enough, leave it as it is
            if len(line) <= 80:
                if not first and self.context == 'equation':
                    # TODO: If previous lines were broken, check that all
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
            if self.context == 'text':
                new_content.extend(self._format_text(line))
            elif self.context == 'equation':
                new_content.extend(self._format_equation(line))
        # Combine the lines to avoid lines too short
        if not first:
            new_content = self.combine_lines(new_content)
        return new_content

    def combine_lines(self, content):
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
            if not self.allow_combine(first, second):
                index_mask[idx] = False
                continue
            content.pop(idx + 1)
            space = ' '
            # why does the first need to be alnum?
            # if not first[-1].isalnum() and second[0] in {'.', ','}:
            if second.lstrip(' %')[0] in {'.', ','}:
                space = ''
            content[idx] = first.rstrip() + space + second.lstrip(' %')
            index_mask.pop(idx)
        return content

    def allow_combine(self, _first, _second):
        if re.search(r'(?:\w{3,}|\W)\.$', _first.strip()):
            return False
        first = _first.strip()
        second = _second.strip()
        if first == '$' or second == '$':
            return False
        if (first[0]=='%') ^ (second[0]=='%'):
            return False
        first = first.strip(' %').endswith
        second = second.strip(' %').startswith
        if second('=') or second('+') or second('-'):
            return False
        if (second('\\equiv') or second('\\cong') or second('\\Longrightarrow')
                or second('\\to')):
            return False
        if second('\\qquad') or second('\\quad'):
            return False
        if first('(') or first('[') or first('{'):
            return False
        if second(')') or second(']') or second('}'):
            return False
        if first('\\left(') or first('\\left[') or first('\\left\\{'):
            return False
        if second('\\right)') or second('\\right]') or second('\\right\\}'):
            return False
        return True

    def line_split(self, line, pattern, keep=False):
        if not isinstance(pattern, re.Pattern):
            pattern = re.compile(pattern)
        skeleton, _,  = self.get_skeleton(line, self.multline_parenthesis)
        lines = []
        prev_idx = 0
        for match in pattern.finditer(skeleton):
            start = self.get_index_line(match.start(), line)
            end = self.get_index_line(match.end(), line)
            if keep == 'first':
                lines.append(line[prev_idx:end])
                prev_idx = end
            elif keep == 'second':
                lines.append(line[prev_idx:start])
                prev_idx = start
            elif keep is True:
                lines.append(line[prev_idx:start])
                lines.append(line[start:end])
                prev_idx = end
            elif keep is False:
                lines.append(line[prev_idx:start])
                prev_idx = end
        lines.append(line[prev_idx:])
        lines = [line for line in lines if line]
        new_lines = map(lambda s: self.indent + s.lstrip(' %').rstrip() + '\n',
                        lines)
        return self.format_tex(new_lines)

    def get_index_line(self, idx, line):
        idx_l = len(self.indent) + idx
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), _ in parenthesis:
            if start is None:
                idx_l = end + idx
                continue
            if start >= idx_l:
                break
            idx_l += end - start - 1
        return idx_l

    def _format_text(self, line):
        skeleton, parenthesis = self.get_skeleton(line,
                                                  self.multline_parenthesis)
        new_lines = []
        # Split phases (separated by . or ?) into multiple lines
        pattern = re.compile(r'[\.\?]\s(?=[A-Z])')
        if pattern.search(skeleton[:-1]):
            return self.line_split(line, pattern, keep='first')
        # Split the line by ':'
        elif ':' in skeleton[:-1]:
            return self.line_split(line, ':', keep='first')
        # Split the line by ','
        elif ',' in skeleton[:-1]:
            return self.line_split(line, ',', keep='first')
        # Split the formulas into a new line.
        if skeleton == '$$':
            start = self.get_index_line(0, line)
            end = self.get_index_line(1, line)
            new_lines.append(line[:start+1])
            self._context.append('equation')
            indent = self.indent
            new_lines.extend(self.format_tex(
                [indent + 4 * ' ' + line[start+1:end].lstrip()]
            ))
            self._context.pop()
            new_lines.append(indent + line[end:].lstrip())
            return self.format_tex(filter(lambda x: x, new_lines))
        if ' $$' in skeleton:
            return self.line_split(line, '\s\$\$', keep=True)
        # Split {} into multiple lines
        for (start, end), char in parenthesis:
            if end - start > 40 and char == '{':
                pass
            elif end - start > 75 and char == '(':
                pass
            else:
                continue
            new_lines.append(self.indent + line[:start+1].lstrip() + '%')
            new_lines.append(
                self.indent + 4 * ' ' + line[start+1:end].lstrip() + '%'
            )
            new_lines.append(self.indent + line[end:].lstrip())
        if new_lines:
            return self.format_tex(new_lines)
        else:
            return self.line_split(line, ' ', keep=False)

    def _format_equation(self, line):
        skeleton, _ = self.get_skeleton(line, self.multline_parenthesis)
        # If equation separator (quad) or equality is present, split line.
        pattern = re.compile(r'\\quad|\\qquad')
        if pattern.search(skeleton[1:]):
            return self.line_split(line, pattern, keep=True)
        pattern = re.compile(r'=|\\equiv|\\cong|\\to|\\Longrightarrow')
        if pattern.search(skeleton[1:]):
            return self.line_split(line, pattern, keep='second')
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
            return self.line_split(line, ' ')
        # If the parenthesis are not big enough, split the line right
        # after a parenthesis
        ret = self.split_after_parenthesis(line)
        if ret is not None:
            return ret
        raise NotImplementedError(f'line "{line}" not splitted.')

    def split_sums(self, line):
        skeleton, _ = self.get_skeleton(line, self.multline_parenthesis)
        new_lines = []
        prev_idx = 0
        # TODO: Handle cases like \cong - 3, etc.
        # This should be done easier with new python 3.11 re functions.
        for match in re.finditer(r'[^=\s]\s*(\+|\-)', skeleton):
            idx_s = match.start(1)
            idx_l = self.get_index_line(idx_s, line)
            new_lines.append(self.indent + line[prev_idx:idx_l].lstrip())
            prev_idx = idx_l
        if new_lines:
            new_lines.append(self.indent + line[idx_l:].lstrip())
            return self.format_tex(new_lines)

    def check_unmatched_parenthesis(self, line):
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), _ in parenthesis:
            if start is None and end > len(self.indent):
                if end > 5 and line[end-6:end] == '\\right':
                    end -= 6
                    if end == len(self.indent):
                        continue
                new_lines = [
                    self.indent + 4*' ' + line[:end].strip() + '\n',
                    self.indent + line[end:].rstrip() + '\n'
                ]
            elif end is None and start + 1 < len(line.rstrip()):
                new_lines = [
                    line[:start+1].rstrip() + '\n',
                    self.indent + 4*' ' + line[start+1:].strip() + '\n'
                ]
            else:
                continue
            return self.format_tex(new_lines)

    def split_large_parenthesis(self, line):
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), _ in parenthesis:
            if end is None or start is None or end - start < 30:
                continue
            if line[end-6:end] == '\\right':
                end -= 6
            new_lines = [
                line[:start + 1] + '\n',
                self.indent + 4*' ' + line[start+1:end].lstrip() + '\n',
                self.indent + line[end:] + '\n'
            ]
            return self.format_tex(new_lines)

    def split_after_parenthesis(self, line):
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), char in reversed(parenthesis):
            if end > 80:
                continue
            elif char == '{' and line[end+1] == '{':
                continue
            new_lines = [line[:end + 1] + '\n',
                         self.indent + line[end+1:].strip() + '\n']
            return self.format_tex(new_lines)

    def update_multiline_parenthesis(self, line):
        if 'phantom' in line:
            return
        open_p = '([{$'
        close_p = ')]}$'
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for ((start, end), char) in parenthesis:
            if start is None:
                _char = self.multline_parenthesis[-1]
                assert close_p.index(char) == open_p.index(_char), (char, _char)
                self.multline_parenthesis = self.multline_parenthesis[:-1]
            elif end is None:
                self.multline_parenthesis += char

    @classmethod
    @functools.cache
    def get_skeleton(cls, line, unmatched_parenthesis=''):
        parenthesis = cls._find_parenthesis(line, unmatched_parenthesis)
        skeleton = line.strip()
        offset = len(line) - len(skeleton)
        for (start, end), _ in parenthesis:
            try:
                skeleton = skeleton[:start-offset+1] + skeleton[end-offset:]
                offset += end - start - 1
            except TypeError:
                if start:
                    skeleton = skeleton[:start-offset+1]
                    break
                elif end:
                    skeleton = skeleton[end-offset:]
                    offset = end
        return skeleton, parenthesis

    @staticmethod
    @functools.cache
    def _find_parenthesis(line, unmatched_parenthesis=''):
        # IDEA: unatched parenthesis is a string containing multiline parenthesis
        # This function must treat this locally, a higher function should keep track of everything
        # There should be an instance attribute containing the multiline parenthesis
        # When a line is splitted, make sure to remove the parenthesis from multiline to avoid
        # double counting.
        open_p = '([{$'
        close_p = ')]}'
        parenthesis_structure = {}
        current_struct = parenthesis_structure
        levels = []
        for idx, char in enumerate(line):
            if char in open_p:
                if char == '$':
                    if idx > 0 and line[idx-1] == '\\':
                        continue
                    open_p = open_p.replace('$', '')
                    close_p = close_p + '$'
                key = (idx, char)
                levels.append(key)
                current_struct[key] = {}
                current_struct = current_struct[key]
            elif char in close_p:
                if levels:
                    start, schar = levels.pop()
                    if schar == '$':
                        if char != '$':
                            raise ValueError(
                                f'Parenthesis not well written: {line}'
                            )
                        elif line[idx-1] == '\\':
                            levels.append((start, schar))
                            continue
                        open_p = open_p + '$'
                        close_p = close_p.replace('$', '')
                    elif open_p.index(schar) != close_p.index(char):
                        if char == ')':
                            # Assume that ) is not part of a parenthesis
                            levels.append((start, schar))
                            continue
                        raise Exception(
                            f'Parenthesis not well written: {line}'
                        )
                else:
                    if unmatched_parenthesis:
                        schar = unmatched_parenthesis[-1]
                        if open_p.index(schar) != close_p.index(char):
                            if char == ')':
                                continue
                            raise Exception(
                                f'Parenthesis not well written: {line}'
                            )
                        unmatched_parenthesis = unmatched_parenthesis[:-1]
                    elif char == ')':
                        continue
                    start = None
                    schar = char
                    current_struct = {}
                parent_structure = parenthesis_structure
                for level in levels:
                    parent_structure = parent_structure[level]
                if start is not None:
                    parent_structure.pop((start, schar))
                parent_structure[((start, idx), schar)] = current_struct
                current_struct = parent_structure
        while levels:
            start, schar = levels.pop()
            parent_structure = parenthesis_structure
            for level in levels:
                parent_structure = parent_structure[level]
            parent_structure.pop((start, schar))
            parent_structure[((start, None), schar)] = current_struct
            current_struct = parent_structure
        return parenthesis_structure

    def __repr__(self):
        if self.formatted_lines == self.init_string:
            return 'String not modified'
        return ''.join(self.formatted_lines)

    def __eq__(self, other):
        if isinstance(other, str):
            return repr(self) == other
        return NotImplemented

s = r'''
'''
r = TeXFormatter(s)
print(r)
