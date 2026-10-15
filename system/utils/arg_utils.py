'''Command-line argument parsers shared by experiment entry points.'''
from __future__ import annotations

import argparse


TRUE_VALUES = frozenset({'1', 'true', 't', 'yes', 'y', 'on'})
FALSE_VALUES = frozenset({'0', 'false', 'f', 'no', 'n', 'off'})


def parse_bool(value: str | bool) -> bool:
    '''Parse common CLI boolean spellings without treating every string as true.'''
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise argparse.ArgumentTypeError(
            f'expected a boolean value, got {value!r}'
        )

    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise argparse.ArgumentTypeError(
        f'expected one of {sorted(TRUE_VALUES | FALSE_VALUES)}, got {value!r}'
    )
