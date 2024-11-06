#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:15:54 2024

@author: Roger Balsach
"""
import argparse as ap
from datetime import datetime, timedelta
from os import chdir
from pathlib import Path
from random import random
from re import sub
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Union, Optional, Dict, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

YEAR: int = 2024
COLS: str = 'GHIJKLMN'

min_in_day: int = 1440
sec_in_hour: int = 3600
sec_in_day: int = 86400
refday: datetime = datetime.strptime('', '')


def format_date(date: Optional[str] = None) -> datetime:
    if date is None:
        date = input('Day (dd/mm): ')
    return datetime.strptime(f'{date}/{YEAR}', '%d/%m/%Y')


def main(filename: Union[Path, str]) -> None:
    args = get_cml_arguments()
    if not isinstance(filename, Path):
        filename = Path(filename)
    if not args.v:
        create_backup_copies(filename)
    date: datetime
    if args.d:
        date = args.d
    elif args.is_interactive:
        date = format_date()
    else:
        date = datetime.today()
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sheet = access_internal_data(filename, tmp_path)
        rn, date_info = get_date_information(sheet, date)
        if args.v:
            print(show_date_info(date, date_info))
        if args.is_interactive:
            date_info = ask_working_hours(date_info, date)
        elif args.n:
            index: int = len(date_info)
            col = COLS[index]
            r = 1 if index % 2 else -1
            status = ('starting' if index == 0 else
                      'finsih/pause' if index % 2 else 'restart')
            value = datetime.today() + r * timedelta(minutes=5)
            date_info[col] = value
            print(f'Set {value:%H:%M} as {status} time on {date:%d/%m/%y}')
        elif args.r:
            if args.r < 0:
                args.r = len(date_info)
            col = COLS[args.r - 1]
            status = 'start' if args.r % 2 else 'stop'
            status, value = ask_reset_time(date, status, date_info[col])
            if status == '_DELETE':
                date_info.pop(col)
            elif status == '_OK':
                date_info[col] = value
        if not args.v:
            save_data(filename, date_info, rn, tmp_path)
            create_xlsm_file(filename, tmp_path)


def get_cml_arguments() -> ap.Namespace:
    parser: ap.ArgumentParser = ap.ArgumentParser()
    parser.add_argument('-d', metavar='date', type=format_date,
                        help="Set date")
    group: ap._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    group.add_argument('-v', action='store_true',
                       help="Get information about the date")
    group.add_argument('-n', action='store_true',
                       help="Add current time to date as start/stop")
    group.add_argument('-r', metavar='n', type=int, nargs='?', const=-1,
                       help="Modify n-th entry of date")

    args: ap.Namespace = parser.parse_args('-v'.split())

    args.is_interactive = False
    if args.n is False and args.r is None and args.v is False:
        args.is_interactive = True

    if args.r is not None and args.r > 8:
        raise Exception('Only 8 entries are allowed per day.')

    return args


def create_backup_copies(filename: Path, num_copies: int = 7) -> None:
    copy_all: bool = False
    for i in range(num_copies, 1, -1):
        backup_file: Path = (filename.parent
                             / f".{filename.stem}_{i - 1}{filename.suffix}")
        new_backup: Path = (filename.parent
                            / f".{filename.stem}_{i}{filename.suffix}")
        r: float = 1/2 * 50**((1 - i)/(num_copies - 1))
        copy_all = copy_all or not new_backup.exists()
        if not (backup_file.exists() and (copy_all or random() < r)):
            continue
        copy_all = True
        copy2(backup_file, new_backup)
    copy2(filename, filename.parent / f".{filename.stem}_1{filename.suffix}")


def access_internal_data(filename: Path, directory: Path) -> ET.ElementTree:
    with ZipFile(filename) as zf:
        zf.extractall(directory)
    return ET.parse(directory / 'xl' / 'worksheets' / 'sheet4.xml')


def get_date_information(sheet: ET.ElementTree, date: datetime
                         ) -> Tuple[int, Dict[str, datetime]]:
    row_number: int = (date - datetime(YEAR, 1, 1)).days + 7
    info: Dict[str, datetime] = {}
    error: bool = False
    for col, cell in get_tree_elements(sheet, row_number).items():
        if len(cell) == 0 or cell[0].text is None:
            error = True
            continue
        assert not error
        value: float = float(cell[0].text)
        info[col] = refday + timedelta(minutes=value * min_in_day)
    return row_number, info


def get_tree_elements(tree: ET.ElementTree, row: int) -> Dict[str, ET.Element]:
    node: ET.Element = tree.getroot()
    prefix: str = node.tag.split('}')[0] + '}'
    for tag in ['sheetData', f"row[@r='{row}']"]:
        next_node = node.find(prefix + tag)
        assert next_node is not None
        node = next_node
    return {col: cell for col in COLS
            if (cell := node.find(prefix + f"c[@r='{col}{row}']")) is not None}


def show_date_info(date: datetime, info: Dict[str, datetime]) -> str:
    info_list: Tuple[datetime, ...] = tuple(info.values())
    if not info_list:
        return f'No information registered for day {date:%A %d-%b}'
    prev_time = info_list[0]
    total: timedelta = timedelta()
    s = f'Working hours from {date:%A %d-%b}:\n  Started at {prev_time:%H:%M}'
    if len(info_list) < 2:
        return s
    for i, time in enumerate(info_list[1:]):
        diff: timedelta = time - prev_time
        if not i % 2:
            total += diff
        status = 'Rested' if i % 2 else 'Worked'
        s += f'\n   {status} between {prev_time:%H:%M} and {time:%H:%M} '
        s += f'({diff.seconds / sec_in_hour:.2f} hours)'
        prev_time = time
    if i % 2 == 0:
        s += f'\n  Finished at {prev_time:%H:%M}. '
        s += f'Total work: {total.seconds / sec_in_hour:.2f} hours'
    return s


def ask_working_hours(date_info: Dict[str, datetime], date: datetime
                      ) -> Dict[str, datetime]:
    for i, col in enumerate(COLS):
        value = date_info.get(col)
        status: str = 'stop' if i % 2 else 'start'
        if value:
            status, time = ask_reset_time(date, status, value)
        else:
            status, time = check_format_time(
                input(f'Enter {status} time (HH:MM) for {date:%d/%m/%y} '
                      + '(leave black to exit): ')
            )
            if status == '_EMTPY':
                break
        if status == '_EMPTY':
            continue
        elif status == '_QUIT':
            break
        elif status == '_DELETE':
            date_info.pop(col)
        else:
            date_info[col] = time
    return date_info


def check_format_time(time: str) -> Tuple[str, datetime]:
    if time.lower() in ('q', 'quit'):
        return '_QUIT', refday
    if time.lower() in ('d', 'delete', 'del'):
        return '_DELETE', refday
    if time.lower() in ('n', 'now'):
        return '_OK', datetime.today()
    count: int = time.count(':')
    if count == 0:
        return '_EMPTY', refday
    elif count == 1:
        return '_OK', datetime.strptime(time, '%H:%M')
    else:
        return '_OK', datetime.strptime(time, '%H:%M:%S')


def ask_reset_time(date: datetime, status: str, value: datetime
                   ) -> Tuple[str, datetime]:
    status, res = check_format_time(
        input(f'On {date:%d/%m/%y} you {status}ed working at {value:%H:%M}.\n'
              + 'Enter new value to overwrite, '
              + 'leave black to continue or type q to quit: ')
    )
    if status == '_EMTPY':
        return status, value
    return status, res


def save_data(filename: Path, date_info: Dict[str, datetime],
              rn: int, directory: Path) -> None:
    cell_content: Dict[str, float] = {col: -1 for col in COLS}
    for cell, time in date_info.items():
        value = (time - refday).seconds / sec_in_day
        assert 0 < value < 1
        cell_content[cell] = value
    xml_file = directory / 'xl' / 'worksheets' / 'sheet4.xml'
    with xml_file.open() as file:
        content = file.read()
    for col, value in cell_content.items():
        if value == -1:
            sub_str = rf'<c r="{col}{rn}" s="\1"/>'
        else:
            sub_str = rf'<c r="{col}{rn}" s="\1" t="n"><v>{value}</v></c>'
        content = sub(
            rf'<c r="{col}{rn}" s="(\d+)"( t="n"|/)>(<v>[\w\.]*</v></c>)?',
            sub_str, content
        )
    with xml_file.open('w') as file:
        file.write(content)


def create_xlsm_file(filename: Path, directory: Path) -> None:
    with ZipFile(filename, 'w', ZIP_DEFLATED, compresslevel=8) as zf:
        chdir(directory)
        for file in directory.rglob('*'):
            if file.is_dir():
                continue
            file.chmod(0o600)
            zf.write(file.relative_to(directory))


if __name__ == '__main__':
    filename = '/home/roger/Documents/DESY/Time_Record.xlsx'
    # print(get_cml_arguments())
    main(filename)
