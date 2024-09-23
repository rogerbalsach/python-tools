#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:42:48 2023

@author: Roger Balsach
"""

import pathlib
import functools
import re
import sys

import numpy as np

separate_list = [r'\\quad', r'\\qquad']
arrow_list = [r'\\to', r'\\xrightarrow(\[.*?\])?\{.*?\}', r'\\Longrightarrow']
equal_list = ['=', r'\\equiv', r'\\cong', r'\\neq']
# A backslash is allowed after these characters without space
backslash_prefactor = [r'\(', r'\[', r'\{', r'\s', r'\$', r'\^', r'\\', r'\-',
                       r'\|', '_', '%']
context_list = [r'\\label', r'\\text']

### Define re patterns:
pattern_separate = re.compile('|'.join(separate_list))
pattern_arrow = re.compile('|'.join(arrow_list))
pattern_equal = re.compile('|'.join(equal_list))
pattern_backslash = re.compile(
    rf'[^{"".join(backslash_prefactor)}]\\(?!right)'
)
pattern_context = re.compile(rf'({"|".join(context_list)})$')

# TODO: Add CLI interface
# TODO: Implement read from file properly


class TeXFormatter:
    # TODO: Manage comments
    # TODO: Manage text environments inside equations
    def __init__(self, content):
        if isinstance(content, str):
            content = content.splitlines(keepends=True)
        self.init_string = content.copy()
        self.reset_context()
        self.multline_parenthesis = ''
        self.indent = ''
        format_content = self._format_spaces(content)
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
            # Replate all tabs by spaces
            line = line.expandtabs(4)
            # Calculate the indent of the line, remove spaces in the beginning
            # of the line.
            if line.lstrip().startswith('%%% '):
                # Emacs local variable definition.
                indent = 0
                cmt = '%%% '
            elif line.lstrip().startswith('%'):
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
            # Make sure all the commas are followed by a space.
            line = line.replace(', ', ',').replace(',', ', ')
            if r'\begin' in line.strip(' %')[6:]:
                idx = line.index(r'\begin')
                if not((match := re.search(r'(?<!\\)%', line))
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

    def _equation_addspace(self, line, offset=0):
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
            end = end or len(line)
            if not start:
                self.indent = ''
            if char == '{':
                if pattern_context.search(line[:start]):
                    add_space.extend(self._text_addspace(line[start+1:end],
                                                         start+1))
                    continue
            add_space.extend(self._equation_addspace(line[start+1:end],
                                                          start+1))
        return [offset + n for n in add_space]

    def _text_addspace(self, line, offset=0):
        add_space = []
        list_parenthesis = [
            self._find_parenthesis(line, self.multline_parenthesis)
        ].copy()
        while list_parenthesis:
            parenthesis = list_parenthesis.pop()
            for (start, end), char in parenthesis:
                if char != '$':
                    list_parenthesis.append(parenthesis[(start, end), char])
                    continue
                add_space.extend(
                    self._equation_addspace(line[start+1:end], offset=start+1)
                )
        return [offset + n for n in add_space]

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
        for match in re.finditer(r'[^\{\(\[](\+|/|=|\\neq|\-|\<)(?!\s|$)',
                                 line):
            add_space.append(offset + match.end())
        return add_space

    def format_tex(self, lines, first=False):
        new_content = []
        for line in map(str.rstrip, lines):
            # print(line)
            # Detect when we are inside an environment
            self.update_context(line)
            # Compute the indent of the line
            self.indent = ' ' * (len(line) - len(line.lstrip()))
            self.commentafter = len(line)
            if line.strip().startswith('%'):
                level = len(line) - len(line.lstrip(' %'))
                self.indent = '%' + ' ' * (level - 1)
            elif (match := re.search(r'(?<!\\)%(?!$)', line)):
                # TODO: Go to line splittings and add a comment if the split
                # occurs after commentafter.
                self.commentafter = match.start()
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
            first = re.search(r'^(.*?)(\s|(?<!\\)%)*$', first).group(1)
            content[idx] = first + space + second.lstrip(' %')
            index_mask.pop(idx)
        return content

    def allow_combine(self, _first, _second):
        if re.search(r'(?:\w{3,}|\W)\.$', _first.strip(' %')):
            return False
        first = _first.strip()
        second = _second.strip()
        if (first[0]=='%') ^ (second[0]=='%'):
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
        first = first.endswith
        second = second.startswith
        if second('+') or second('-'):
            return False
        if first('(') or first('[') or first('{'):
            return False
        if second(')') or second(']') or second('}'):
            return False
        if first('\\left(') or first('\\left[') or first('\\left\\{'):
            return False
        if second('\\right)') or second('\\right]') or second('\\right\\}'):
            return False
        # if _first.strip()[-1] == '%' and _first.strip()[-2] != '\\':
        #     return True
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
        if self.commentafter < len(line):
            new_lines = [line[:self.commentafter], line[self.commentafter:]]
            return self.format_tex(new_lines)
        # Split phases (separated by . or ?) into multiple lines
        pattern = re.compile(r'.[\.\?](?=\s[A-Z])')
        if pattern.search(skeleton[:-1]):
            return self.line_split(line, pattern, keep='first')
        # Split the line by ':'
        elif ':' in skeleton[:-1]:
            return self.line_split(line, ':', keep='first')
        # Split the line by ','
        elif ',' in skeleton[:-1]:
            return self.line_split(line, ',', keep='first')
        # Split the line by ','
        elif ' and ' in skeleton[:-1]:
            return self.line_split(line, r'(?<=\s)and(?=\s)', keep='first')
        # Split the formulas into a new line.
        new_lines = []
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
            # TODO: Add % after last $.
            new_lines.append(indent + line[end:].lstrip())
            return self.format_tex(filter(lambda x: x, new_lines))
        if ' $$' in skeleton:
            return self.line_split(line, r'\s\$\$', keep=True)
        # Split {} into multiple lines
        for (start, end), char in parenthesis:
            start = start or 0
            end = end or len(line)
            if end - start > 40 and char == '{':
                pass
            elif end - start > 75 and char == '(':
                pass
            else:
                continue
            new_lines.append(self.indent + line[:start+1].lstrip(' %') + '%')
            new_lines.append(
                self.indent + 4 * ' ' + line[start+1:end].lstrip() + '%'
            )
            new_lines.append(self.indent + line[end:].lstrip())
            new_lines = [line for line in new_lines if line.strip(' %')]
        if new_lines:
            return self.format_tex(new_lines)
        if ' ' in skeleton:
            return self.line_split(line, ' ', keep=False)
        return [line]

    def _format_equation(self, line):
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
        for match in re.finditer(r'[^=\s]\s*(\+|\-)', skeleton.strip(' &')):
            idx_s = match.start(1)
            idx_l = self.get_index_line(idx_s, line)
            new_lines.append(self.indent + line[prev_idx:idx_l].lstrip(' %'))
            prev_idx = idx_l
        if new_lines:
            new_lines.append(self.indent + line[idx_l:].lstrip())
            return self.format_tex(new_lines)

    def check_unmatched_parenthesis(self, line):
        parenthesis = self._find_parenthesis(line, self.multline_parenthesis)
        for (start, end), _ in parenthesis:
            if start is None and end > len(self.indent):
                if (match := re.search(r'\\(right|[bB]igg?)\s?\\?$',
                                       line[:end])):
                    end = match.start()
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
            if (match := re.search(r'\\(right|[bB]igg?)\s?\\?$', line[:end])):
                end = match.start()
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
            elif char == '{' and end + 1 < len(line) and line[end+1] == '{':
                continue
            new_lines = [line[:end + 1] + '\n',
                         self.indent + line[end+1:].strip() + '\n']
            return self.format_tex(new_lines)

    def update_multiline_parenthesis(self, line):
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
                new_parenthesis.append((end, char))
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
                continue
            sent = False
            if char in open_p:
                self.multline_parenthesis += char

    @classmethod
    @functools.cache
    def get_skeleton(cls, line, unmatched_parenthesis=''):
        parenthesis = cls._find_parenthesis(line, unmatched_parenthesis)
        skeleton = line.lstrip(' %')
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
        return skeleton.strip(' %'), parenthesis

    @staticmethod
    @functools.cache
    def _find_parenthesis(line, unmatched_parenthesis=''):
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
                        fail = True
                        if char == ')':
                            # Assume that ) is not part of a parenthesis
                            levels.append((start, schar))
                            continue
                        if char == '}':
                            # Check its not a phantom context
                            while levels:
                                start, schar = levels.pop()
                                if schar != '{':
                                    continue
                                if not re.search(r'\\phantom$', line[:start]):
                                    continue
                                fail = False
                                break
                        if fail:
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
                    parenthesis_structure = {
                        ((None, idx), char): parenthesis_structure
                    }
                    current_struct = parenthesis_structure
                    continue
                parent_structure = parenthesis_structure
                for level in levels:
                    parent_structure = parent_structure[level]
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
\begin{equation*}
    L_{1}
    = \frac{2 \alpha_{S}^{2}}{b_{0} \pi^2} \left\{
        \pi b_{0}^{2} \left(- C_{i} \log(\frac{Q^{2}}{\mu^{2}_{F}})^{2} H^{(0)} S^{(0)} + 2 C_{i} \log(\frac{Q^{2}}{\mu^{2}_{F}}) \log(\frac{Q^{2}}{\mu^{2}_{R}}) H^{(0)} S^{(0)} - 4 \gamma_{E} C_{i} \log(\frac{Q^{2}}{\mu^{2}_{R}}) H^{(0)} S^{(0)} + 4 C_{i} \zeta_{2} H^{(0)} S^{(0)} + 4 \gamma_{E}^{2} C_{i} H^{(0)} S^{(0)} + \log(\frac{Q^{2}}{\mu^{2}_{R}}) H^{(0)} S^{(0)} \Gamma^{(1)} + \log(\frac{Q^{2}}{\mu^{2}_{R}}) H^{(0)} \overline{\Gamma^{(1)}} S^{(0)} + 2 H^{(0)} S^{(0)} K - 2 \gamma_{E} H^{(0)} S^{(0)} \Gamma^{(1)} + 2 H^{(0)} S^{(1)} + 2 H^{(0)} \overline{K} S^{(0)} - 2 \gamma_{E} H^{(0)} \overline{\Gamma^{(1)}} S^{(0)}\right) + b_{0} \left(- 2 A^{(2)} C_{i} \log(\frac{Q^{2}}{\mu^{2}_{F}}) H^{(0)} S^{(0)} + 4 \gamma_{E} A^{(2)} C_{i} H^{(0)} S^{(0)} - C_{i} D^{(2)} H^{(0)} S^{(0)} - 2 C_{i} \log(\frac{Q^{2}}{\mu^{2}_{F}}) H^{(0)} S^{(1)} - 2 C_{i} \log(\frac{Q^{2}}{\mu^{2}_{F}}) H^{(1)} S^{(0)} + 4 \gamma_{E} C_{i} H^{(0)} S^{(1)} + 4 \gamma_{E} C_{i} H^{(1)} S^{(0)} - H^{(0)} S^{(0)} K \Gamma^{(1)} + H^{(0)} S^{(0)} \Gamma^{(1)} K - H^{(0)} S^{(1)} \Gamma^{(1)} + H^{(0)} \overline{K} \overline{\Gamma^{(1)}} S^{(0)} - H^{(0)} \overline{\Gamma^{(1)}} S^{(1)} - H^{(0)} \overline{\Gamma^{(1)}} \overline{K} S^{(0)} - H^{(1)} S^{(0)} \Gamma^{(1)} - H^{(1)} \overline{\Gamma^{(1)}} S^{(0)}\right) - \pi b_{1} \left(H^{(0)} S^{(0)} \Gamma^{(1)} + H^{(0)} \overline{\Gamma^{(1)}} S^{(0)}\right)
    \right\}
\end{equation*}
'''
r = TeXFormatter(s)
print(r)

# if __name__ == '__main__':
#     pathdir = pathlib.Path.cwd()
#     filename = pathdir / pathlib.Path(sys.argv[1])
#     print(filename)
#     dest = pathdir / pathlib.Path(sys.argv[1])
#     with filename.open() as file:
#         content = file.readlines()
#     new_content = TeXFormatter(content).formatted_lines
#     with dest.open('w') as file:
#         file.writelines(new_content)

### TESTS ###

# 6
s = r'''
In particular, we are interested in evolving $S$ from the scale where renormalization takes place $\mu_0=\mu_R$ to the soft scale $\mu = Q\bar{N}^{-1}$. Where $\bar{N}=Ne^{\gamma_E}$ as defined in \cite{Kulesza17}. Then
'''
assert TeXFormatter(s) == '\nIn particular, we are interested in evolving $S$\nfrom the scale where renormalization takes place\n$\\mu_0 = \\mu_R$ to the soft scale $\\mu = Q \\bar{N}^{-1}$.\nWhere $\\bar{N} = Ne^{\\gamma_E}$ as defined in \\cite{Kulesza17}.\nThen\n'

# 13
s = r'''
\begin{equation*}
    -b_0\alpha_S(\mu_0^2)\log(\frac{\mu^2}{\mu_0^2}) = \alpha_S(\mu_0^2)\frac{b_1}{b_0}\log(\frac{\frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)}+ \alpha_S(\mu_0^2)\frac{b_1}{b_0}}{1+\alpha_S(\mu_0^2) \frac{b_1}{b_0}})+1-\frac{\alpha_S(\mu_0^2)}{\alpha_S(\mu^2)} + \order{\alpha_S^2}
\end{equation*}
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    -b_0 \\alpha_S(\\mu_0^2) \\log(\\frac{\\mu^2}{\\mu_0^2})\n    = \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0} \\log(\n        \\frac{\n            \\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)}\n            + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0}\n        }{1 + \\alpha_S(\\mu_0^2) \\frac{b_1}{b_0}}\n    )\n    + 1\n    - \\frac{\\alpha_S(\\mu_0^2)}{\\alpha_S(\\mu^2)}\n    + \\order{\\alpha_S^2}\n\\end{equation*}\n'

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
Now, equating both derivatives and using equation \eqref{eq:Kmatrix} for $U$ we arrive at the following equation:
\begin{equation*}
    \frac{\Gamma(\alpha_S)}{\beta(\alpha_S)}K(\alpha_S)
    = \dv{K(\alpha_S)}{\alpha_S} - K(\alpha_S)\frac{\Gamma^{(1)}}{2\pi\alpha_Sb_0}
\end{equation*}
'''
assert TeXFormatter(s) == '\nNow, equating both derivatives and\nusing equation \\eqref{eq:Kmatrix} for $U$ we arrive at the following equation:\n\\begin{equation*}\n    \\frac{\\Gamma(\\alpha_S)}{\\beta(\\alpha_S)} K(\\alpha_S)\n    = \\dv{K(\\alpha_S)}{\\alpha_S}\n    - K(\\alpha_S) \\frac{\\Gamma^{(1)}}{2 \\pi \\alpha_S b_0}\n\\end{equation*}\n'

s = r'''
\begin{equation*}
    \dv{U(\mu, \mu_0)}{\mu} = \dv{\alpha_S(\mu^2)}{\mu}\frac{\Gamma(\alpha_S(\mu^2))}{\beta(\alpha_S(\mu^2))}\Pexp{\int_{\alpha_S(\mu^2_0)}^{\alpha_S(\mu^2)} \frac{\Gamma(\alpha_S)}{\beta(\alpha_S)}\dd{\alpha_S}}
    = \dv{\alpha_S(\mu^2)}{\mu}\frac{\Gamma(\alpha_S(\mu^2))}{\beta(\alpha_S(\mu^2))}U(\mu, \mu_0)
\end{equation*}
'''
assert TeXFormatter(s) == '\n\\begin{equation*}\n    \\dv{U(\\mu, \\mu_0)}{\\mu}\n    = \\dv{\\alpha_S(\\mu^2)}{\\mu}\n    \\frac{\\Gamma(\\alpha_S(\\mu^2))}{\\beta(\\alpha_S(\\mu^2))} \\Pexp{\n        \\int_{\\alpha_S(\\mu^2_0)}^{\\alpha_S(\\mu^2)}\n        \\frac{\\Gamma(\\alpha_S)}{\\beta(\\alpha_S)} \\dd{\\alpha_S}\n    }\n    = \\dv{\\alpha_S(\\mu^2)}{\\mu}\n    \\frac{\\Gamma(\\alpha_S(\\mu^2))}{\\beta(\\alpha_S(\\mu^2))} U(\\mu, \\mu_0)\n\\end{equation*}\n'

s = '''
However, the consequence is that the non-radiative amplitude in the r.h.s. of eq. \\eqref{eq:NLP} is evaluated using the momenta $p$, which are unphysical for this process, because $\\sum \\eta_i p_i \\neq 0$. This might seem problematic because an amplitude is intrinsically defined for physical momenta, and it is not uniquely defined for unphysical momenta. Therefore, the value of $\\mathcal{H}(p)$ is ambiguous, which translates into an ambiguity on $\\mathcal{A}(p, k)$ and thus seems to invalidate eq. \\eqref{eq:NLP}. The argument, however, is not entirely correct, as shown in \\cite{Balsach:2023ema}. Indeed, although an ambiguity is present, it only affects the NNLP terms.
'''
assert TeXFormatter(s) == '\nHowever,\nthe consequence is that the non-radiative amplitude in the r.h.s. of eq.\n\\eqref{eq:NLP} is evaluated using the momenta $p$,\nwhich are unphysical for this process, because $\\sum \\eta_i p_i \\neq 0$.\nThis might seem problematic because an amplitude is\nintrinsically defined for physical momenta,\nand it is not uniquely defined for unphysical momenta.\nTherefore, the value of $\\mathcal{H}(p)$ is ambiguous,\nwhich translates into an ambiguity on $\\mathcal{A}(p, k)$ and\nthus seems to invalidate eq. \\eqref{eq:NLP}.\nThe argument, however, is not entirely correct,\nas shown in \\cite{Balsach:2023ema}.\nIndeed, although an ambiguity is present, it only affects the NNLP terms.\n'

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
