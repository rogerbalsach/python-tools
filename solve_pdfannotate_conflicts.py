#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:24:58 2024

@author: Roger Balsach
"""

import pathlib
import re
import sys

pat = re.compile(r'\\pdfmarkupcomment\[.*?\]{.*?}{(.*?)}')

def get_conflict_list(lines):
    conflict = None
    conflict_list = []
    for i, line in enumerate(lines):
        if line.startswith('<<<<<<<'):
            conflict = i
            conflict_str = ''
            continue
        if line.startswith('======='):
            HEAD_str = conflict_str
            conflict_str = ''
            continue
        if line.startswith('>>>>>>>'):
            conflict_list.append(
                {'lines': (conflict, i),
                 'HEAD': HEAD_str,
                 'comment': conflict_str.replace('\n', ' ').replace('  ', ' ')}
            )
            conflict = None
            continue
        if conflict is not None:
            conflict_str += line
    return conflict_list

def solve_conflicts(conflict_list):
    for conflict in conflict_list:
        first = conflict['HEAD'].replace('\n', ' ')
        second = conflict['comment']
        rec = ''
        idx = 0
        for match in pat.finditer(second):
            rec += second[idx:match.start(0)] + match.group(1)
            idx = match.end(0)
        rec += second[idx:]
        while '  ' in first or '  ' in rec:
            first = first.replace('  ', ' ')
            rec = rec.replace('  ', ' ')
        if first == rec:
            conflict['solved'] = True
        else:
            conflict['solved'] = False
    return conflict_list

def rewrite_solved(lines):
    # import pdb;pdb.set_trace()
    cl = get_conflict_list(lines)
    cl = solve_conflicts(cl)
    new_lines = []
    idx = 0
    for conflict in filter(lambda x: x['solved'], cl):
        start, end = conflict['lines']
        new_lines += lines[idx:start] + [conflict['HEAD']]
        idx = end + 1
    new_lines += lines[idx:]
    return new_lines


s = r'''
<<<<<<< ours
\chapter{Cross sections for $e^+ e^-$ annihilations}
\section{$e^+ e^- \xrightarrow{} \gamma^* \xrightarrow{} \mu^+ \mu^-$}
In this section a simple $e^+ e^-$ annihilation is studied.
The goal of this section is to understand how to apply the Feynman rules and
how to ascertain a cross section out of the amplitude.
The reaction studied in this section is
$e^+ e^- \xrightarrow{} \mu^+ \mu^-$ via exchange of a photon.
=======
\chapter{Cross sections for e$^+$e$^-$ Annihilations}
\section{\pdfmarkupcomment[markup=Highlight, color=yellow]{e$^+$e$^- \xrightarrow{} \gamma^* \xrightarrow{} \mu^+ \mu^-$}{why electrons in roman font?}}
In this section a simple e$^+$e$^-$-annihilation is studied.
The goal of this section is to understand how to apply the Feynman rules and
how to ascertain a cross section out of the amplitude.
The reaction studied in this section is
\pdfmarkupcomment[markup=Highlight, color=yellow]{e$^+$e$^- \xrightarrow{} \mu^+ \mu^-$}{why electrons in roman font?} via exchange of a photon.
>>>>>>> theirs
The Feynman diagram for this reaction is given by \\
\begin{figure}[H]
    \centering
\begin{tikzpicture}
    \begin{feynman}
        \vertex (a);
        \vertex [above left=of a] (e1) {$e^-$};
        \vertex [below left=of a] (e2) {$e^+$};
        \vertex [right=of a] (b);
        \vertex [above right=of b] (mu1) {$\mu^-$};
        \vertex [below right=of b] (mu2) {$\mu^+$};
        \diagram{
        (e1)
        -- [fermion, momentum=$p_1$] (a)
        -- [fermion, reversed momentum=$p_2$] (e2);
        (a) -- [boson, edge label=$\gamma$, momentum'=$q_1$] (b);
        (mu2)
        -- [fermion, reversed momentum=$p_4$] (b)
        -- [fermion, momentum=$p_3$] (mu1)
        };
    \end{feynman}
\end{tikzpicture}
\caption{%
    Feynman diagram for the electron-positron annihilation process
    $e^+ e^- \to \mu^+ \mu^-$%
}
\end{figure}
'''
try:
    assert __IPYTHON__
    text = s.splitlines(keepends=True)
    print(rewrite_solved(text))
except NameError:
    if __name__ == '__main__':
        pathdir = pathlib.Path.cwd()
        filename = pathdir / pathlib.Path(sys.argv[1])
        print(filename)
        dest = pathdir / pathlib.Path(sys.argv[1])
        with filename.open() as file:
            content = file.readlines()
        new_content = rewrite_solved(content)
        with dest.open('w') as file:
            file.writelines(new_content)
