'''Command-line argument parsers shared by experiment entry points.'''
from __future__ import annotations

import argparse
import math


TRUE_VALUES = frozenset({'1', 'true', 't', 'yes', 'y', 'on'})
FALSE_VALUES = frozenset({'0', 'false', 'f', 'no', 'n', 'off'})

POSITIVE_INT_FIELDS = (
    'num_classes',
    'batch_size',
    'global_rounds',
    'local_epochs',
    'num_clients',
    'times',
    'eval_gap',
    'dlg_gap',
    'batch_num_per_client',
    'feature_dim',
    'vocab_size',
    'max_len',
    'K',
    'M',
    'itk',
    'plocal_epochs',
    'fine_tuning_epochs',
    'noise_dim',
    'hidden_dim',
    'server_epochs',
    'layer_idx',
)

NONNEGATIVE_INT_FIELDS = (
    'top_cnt',
    'prev',
    'num_new_clients',
    'fine_tuning_epoch_new',
    'rand_percent',
)

POSITIVE_FLOAT_FIELDS = (
    'local_learning_rate',
    'learning_rate_decay_gamma',
    'time_threthold',
    'p_learning_rate',
    'alphaK',
    'sigma',
    'lamda',
    'alpha',
    'tau',
    'L',
    'generator_learning_rate',
    'server_learning_rate',
    'eta',
    'mentee_learning_rate',
    'T_start',
    'T_end',
)

NONNEGATIVE_FLOAT_FIELDS = (
    'beta',
    'mu',
    'dr_learning_rate',
    'momentum',
    'kl_weight',
)

RATIO_FIELDS = (
    'join_ratio',
    'client_drop_rate',
    'train_slow_rate',
    'send_slow_rate',
)


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


def validate_experiment_args(args, parser: argparse.ArgumentParser | None = None):
    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    for field_name in POSITIVE_INT_FIELDS:
        if not hasattr(args, field_name):
            continue
        value = getattr(args, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            fail(f'{field_name} must be an integer')
        if value < 1:
            fail(f'{field_name} must be at least 1')

    for field_name in NONNEGATIVE_INT_FIELDS:
        if not hasattr(args, field_name):
            continue
        value = getattr(args, field_name)
        if isinstance(value, bool) or not isinstance(value, int):
            fail(f'{field_name} must be an integer')
        if value < 0:
            fail(f'{field_name} must be at least 0')
    if hasattr(args, 'rand_percent') and args.rand_percent > 100:
        fail('rand_percent must be at most 100')

    for field_name in POSITIVE_FLOAT_FIELDS:
        if not hasattr(args, field_name):
            continue
        value = getattr(args, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(f'{field_name} must be a number')
        if not math.isfinite(value) or value <= 0:
            fail(f'{field_name} must be finite and greater than 0')

    for field_name in NONNEGATIVE_FLOAT_FIELDS:
        if not hasattr(args, field_name):
            continue
        value = getattr(args, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(f'{field_name} must be a number')
        if not math.isfinite(value) or value < 0:
            fail(f'{field_name} must be finite and non-negative')

    for field_name in RATIO_FIELDS:
        if not hasattr(args, field_name):
            continue
        value = getattr(args, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail(f'{field_name} must be a number')
        if not math.isfinite(value) or not 0 <= value <= 1:
            fail(f'{field_name} must be in [0, 1]')

    return args
