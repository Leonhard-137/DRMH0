"""Checkpoint validation and file preparation for the project's MICA launcher.

The native checkpoint format is the x86-64 CDNest format shipped with MICA 2.2.
It does not encode RNG state; this resumes a saved sampler, not a bitwise replay.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time

STATE_FILE = 'run_mica_state.json'
OPTION_KEYS = {'NumberParticles', 'TotalParticles', 'ThreadStepsFactor',
               'NewLevelIntervalFactor', 'SaveIntervalFactor', 'MaxNumberLevels',
               'BacktrackingLength', 'StrengthEqualPush', 'PTol'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def acquire_run_lock(rd):
    # Keep the lock outside rd so explicitly archiving a run cannot drop it.
    rd = Path(rd)
    rd.parent.mkdir(parents=True, exist_ok=True)
    handle = (rd.parent / ('.' + rd.name + '.lock')).open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'An existing launcher is using {rd}; stop that job before resuming.')
    return handle


def read_settings(path, commented_options=False):
    result = {}
    for line in Path(path).read_text().splitlines():
        if commented_options and line.lstrip().startswith('#'):
            candidate = line.lstrip()[1:].lstrip()
            if candidate.split() and candidate.split()[0] in OPTION_KEYS:
                line = candidate
        words = line.split('#', 1)[0].split(maxsplit=1)
        if len(words) == 2:
            result[words[0]] = words[1].strip()
    return result


def _value(value):
    try:
        if value.startswith('[') and value.endswith(']'):
            return [float(x) for x in value[1:-1].split(':')]
        return float(value)
    except ValueError:
        return value


def compare_saved_setup(rd, preview_dir):
    """Compare actual pymica serialization before touching the saved run."""
    rd, preview_dir = Path(rd), Path(preview_dir)
    old = read_settings(rd / 'param/param_input', True)
    new = read_settings(preview_dir / 'param/param_input', True)
    for values in (old, new):
        values.setdefault('TotalParticles', '0')
    if int(new['MaxNumberSaves']) < int(old['MaxNumberSaves']):
        raise ValueError('max_num_saves is a cumulative target and cannot decrease on resume.')
    ignored = {'FileDir', 'DataFile', 'MaxNumberSaves'}
    changed = [key for key in old.keys() | new.keys() if key not in ignored and
               _value(old.get(key, '')) != _value(new.get(key, ''))]
    if changed:
        raise ValueError('Saved MICA configuration differs: ' + ', '.join(sorted(changed)) +
                         '. Restore the original settings, or use fresh for a separate run.')
    original_data = rd / old.get('DataFile', 'data/data_input.txt')
    proposed_data = preview_dir / new.get('DataFile', 'data/data_input.txt')
    if original_data.read_bytes() != proposed_data.read_bytes():
        raise ValueError('Serialized observations differ from the saved MICA input; refusing resume.')
    for prior in ['param/ordered_gaussian_lag_priors.txt', 'param/joint_parameter_priors.txt']:
        if (rd / prior).exists() != (preview_dir / prior).exists() or (
            (rd / prior).exists() and (rd / prior).read_bytes() != (preview_dir / prior).read_bytes()
        ):
            raise ValueError(f'The saved parameter prior differs from the requested prior: {prior}')


def _rows(path, columns, required=None):
    """Validate complete numeric rows; only a torn tail after required is ignored."""
    rows, end, headers = 0, 0, 0
    with Path(path).open('rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith(b'#'):
                if rows == 0:
                    end = f.tell()
                    headers += 1
                continue
            if required is not None and rows >= required:
                break
            try:
                values = [float(x) for x in stripped.split()]
            except ValueError as exc:
                raise ValueError(f'Invalid saved numeric row {rows + 1}: {path}') from exc
            if not line.endswith(b'\n') or len(values) != columns or not all(map(math.isfinite, values)):
                raise ValueError(f'Incomplete or invalid saved row {rows + 1}: {path}')
            rows += 1
            end = f.tell()
    if required is not None and rows < required:
        raise ValueError(f'{path} has {rows} complete rows, but checkpoint needs {required}.')
    if not rows and required != 0:
        raise ValueError(f'No saved numeric rows: {path}')
    return {'rows': rows, 'prefix_bytes': end}


def read_checkpoint(path, *, ranks, particles, dimension, max_levels=0):
    if sys.byteorder != 'little' or struct.calcsize('P') != 8:
        raise ValueError('This checkpoint reader requires the verified x86-64 MICA layout.')
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError('truncated checkpoint header')
    saves, steps, levels = struct.unpack_from('<IIi', data)
    if saves < 1 or not 1 <= levels <= 1000 or (max_levels and levels > max_levels):
        raise ValueError('invalid checkpoint count or incompatible level limit')
    total = ranks * particles
    expected = 12 + 56 * levels + 20 * total + 16 * levels * dimension + 8 * total * dimension
    if len(data) != expected:
        raise ValueError(f'checkpoint has {len(data)} bytes; expected {expected} for the saved MPI/model layout')
    for i in range(levels):
        values = struct.unpack_from('<3d4Q', data, 12 + 56 * i)
        if not all(map(math.isfinite, values[:3])):
            raise ValueError('nonfinite checkpoint level')
    offset = 12 + 56 * levels
    for i in range(total):
        level, likelihood, tie = struct.unpack_from('<Idd', data, offset + 20 * i)
        if level >= levels or not math.isfinite(likelihood) or not math.isfinite(tie):
            raise ValueError('invalid particle assignment or likelihood')
    offset += 20 * total
    for (value,) in struct.iter_unpack('<d', data[offset:]):
        if not math.isfinite(value):
            raise ValueError('nonfinite checkpoint limits or parameters')
    return {'path': str(Path(path)), 'saves': saves, 'global_steps_low32': steps,
            'levels': levels, 'dimension': dimension, 'particles_per_rank': particles,
            'ranks': ranks, 'bytes': len(data)}


def response_dimension(rd, ncomp):
    """One observed driver and one or more responses in the same MICA dataset."""
    with (Path(rd) / 'data/data_input.txt').open() as stream:
        first, second = stream.readline().split(), stream.readline().split()
    if first != ['#', '1'] or len(second) != 2 or second[0] != '#':
        raise ValueError('This launcher expects exactly one MICA dataset.')
    counts = [int(value) for value in second[1].split(':')]
    if len(counts) < 2 or any(count < 1 for count in counts):
        raise ValueError('MICA input must contain a nonempty driver and response curves.')
    return 3 + (len(counts) - 1) * (1 + 3*ncomp)


def _layout(rd, phase, ranks, ncomp, mixed):
    if phase == 'response':
        suffix, dimension = f'1d.txt_{ncomp}', response_dimension(rd, ncomp)
        alias = 'restart_dnest1d.txt' if mixed else f'restart_dnest1d.txt_{ncomp}'
        options = read_settings(rd / 'param/OPTIONS1D')
    else:
        suffix, dimension, alias = '.txt', 3, 'restart_dnest.txt'
        options = read_settings(rd / 'param/OPTIONSCON')
    particles = int(options.get('NumberParticles', '1'))
    total = int(options.get('TotalParticles', '0'))
    if total:
        if total < ranks or total % ranks:
            raise ValueError('Saved TotalParticles is incompatible with the MPI process count.')
        particles = total // ranks
    if particles < 1:
        raise ValueError('Invalid saved particle count.')
    batch = int(dimension * float(options.get('ThreadStepsFactor', '1')) * particles) * ranks
    interval = int(batch * float(options.get('SaveIntervalFactor', options.get('NewLevelIntervalFactor', '1'))))
    return {'alias': alias, 'dimension': dimension, 'particles': particles,
            'max_levels': int(options.get('MaxNumberLevels', '0')),
            'batch': batch, 'interval': interval,
            'sample': rd / 'data' / ('sample' + suffix),
            'info': rd / 'data' / ('sample_info' + suffix)}


def _choose(rd, phase, ranks, ncomp, mixed, target):
    layout = _layout(rd, phase, ranks, ncomp, mixed)
    pattern = re.compile(re.escape(layout['alias']) + r'_(\d+)$')
    candidates = [(int(match[1]), path) for path in (rd / 'data').glob(layout['alias'] + '_*')
                  if (match := pattern.fullmatch(path.name))]
    rejected = []
    for count, path in sorted(candidates, reverse=True):
        try:
            cp = read_checkpoint(path, ranks=ranks, particles=layout['particles'],
                                 dimension=layout['dimension'], max_levels=layout['max_levels'])
            if count != cp['saves']:
                raise ValueError('filename count disagrees with the checkpoint header')
            if count > target:
                raise ValueError('checkpoint exceeds the requested cumulative target')
            # Native CDNest writes and reads only 32 bits of its 64-bit counter.
            expected_steps = max(count, math.ceil(count * layout['interval'] / layout['batch'])) * layout['batch']
            if expected_steps >= 2**32 or cp['global_steps_low32'] != expected_steps:
                raise ValueError('native checkpoint step counter is inconsistent or exceeds its 32-bit format')
            cp['sample'] = str(layout['sample'])
            cp['info'] = str(layout['info'])
            cp['sample_prefix'] = _rows(layout['sample'], layout['dimension'], count)['prefix_bytes']
            cp['info_prefix'] = _rows(layout['info'], 4, count)['prefix_bytes']
            cp['alias'] = str(rd / 'data' / layout['alias'])
            cp['rejected_newer'] = rejected
            return cp
        except (OSError, ValueError, ZeroDivisionError) as exc:
            rejected.append({'path': str(path), 'reason': str(exc)})
    raise ValueError('No complete, compatible checkpoint with matching sample rows. ' +
                     '; '.join(f"{Path(x['path']).name}: {x['reason']}" for x in rejected[:3]))


def _validate_prior_file(rd):
    path = rd / 'data/posterior_sample.txt'
    lines = path.read_text().splitlines()
    header = lines[0].split() if lines else []
    if len(header) != 2 or header[0] != '#' or int(header[1]) < 1:
        raise ValueError('Continuum posterior has an invalid sample-count header.')
    rows = [line.split() for line in lines[1:] if line.strip()]
    if len(rows) != int(header[1]) or any(len(row) != 3 or not all(math.isfinite(float(x)) for x in row) for row in rows):
        raise ValueError('Continuum posterior has incomplete or invalid rows.')
    commitment = rd / 'data/continuum_prior.json'
    if commitment.is_file():
        prior = json.loads(commitment.read_text())
        values = prior['mean'] + prior['std']
        valid = (prior.get('schema_version') == 1 and prior.get('parameter_count') == 3 and
                 prior.get('sample_count') == len(rows) and prior.get('posterior_sha256') == digest(path) and
                 len(values) == 6 and all(map(math.isfinite, values)) and
                 hashlib.sha256(struct.pack('<6d', *values)).hexdigest() == prior.get('state_sha256'))
        if not valid:
            raise ValueError('Continuum posterior/prior no longer matches its saved commitment.')


def _signature(settings):
    return json.loads(json.dumps({k: v for k, v in settings.items()
                                 if k not in {'data', 'data_file', 'max_num_saves'}}, sort_keys=True))


def plan_run(rd, *, action, mpi_size, mixed, ncomp, target_saves,
             input_files, settings, native_extension):
    rd = Path(rd)
    inputs = {name: digest(path) for name, path in input_files.items()}
    state = {'schema': 1, 'settings': _signature(settings), 'mpi_size': mpi_size,
             'mixed': mixed, 'ncomp': ncomp, 'target_saves': target_saves,
             'inputs': inputs, 'native_sha256': digest(native_extension)}
    plan = {'mode': 'fresh', 'state': state, 'action': action}
    old = json.loads((rd / STATE_FILE).read_text()) if (rd / STATE_FILE).is_file() else None
    populated = (rd / 'param/param_input').exists() or any((rd / 'data').glob('sample*.txt*'))
    if action == 'fresh':
        plan['archive_existing'] = rd.exists() and any(rd.iterdir())
        return plan
    if not populated:
        if action != 'run':
            raise ValueError('No existing MICA run to resume or postprocess.')
        return plan
    for name, value in inputs.items():
        if not (rd / 'input' / name).is_file() or digest(rd / 'input' / name) != value:
            raise ValueError(f'Prepared input {name} differs from the saved run; refusing to overwrite it.')
    if old:
        identity_keys = ['settings', 'mixed', 'ncomp', 'inputs']
        if action not in {'post', 'plot'}:
            identity_keys += ['mpi_size', 'native_sha256']
        for key in identity_keys:
            if old.get(key) != state[key]:
                raise ValueError(f'Saved run {key} differs; restore the original environment/configuration for resume.')
        if target_saves < old['target_saves']:
            raise ValueError('max_num_saves is cumulative and cannot decrease.')
        for name, value in old.get('frozen_files', {}).items():
            if not (rd / name).is_file() or digest(rd / name) != value:
                raise ValueError(f'Saved run file was changed externally: {name}')
        state['frozen_files'] = old.get('frozen_files', {})
        if action in {'post', 'plot'}:
            state['mpi_size'] = old['mpi_size']
            state['native_sha256'] = old['native_sha256']
        if action == 'run' and old.get('status') == 'complete' and target_saves == old['target_saves']:
            _validate_prior_file(rd)
            return {**plan, 'mode': 'skip'}
    else:
        runtime_path = rd / 'data/DNEST_OPTIONS'
        if not runtime_path.is_file():
            raise ValueError('Legacy run has no DNEST_OPTIONS; its original MPI layout cannot be verified.')
        runtime = read_settings(runtime_path)
        if action not in {'post', 'plot'} and int(runtime.get('NumberCores', '0')) != mpi_size:
            raise ValueError('Resume requires the original MPI process count (NumberCores in data/DNEST_OPTIONS).')
        state['adopted_legacy'] = True
        if action in {'post', 'plot'}:
            state['mpi_size'] = int(runtime['NumberCores'])
    previous = read_settings(rd / 'param/param_input')
    if target_saves < int(previous['MaxNumberSaves']):
        raise ValueError('max_num_saves is cumulative and cannot decrease.')
    if (rd / 'data/posterior_sample.txt').is_file():
        try:
            _validate_prior_file(rd)
        except (ValueError, UnicodeError):
            # The continuum checkpoint is saved before its posterior is written.
            # Recover a torn posterior only before an RM prior was committed or
            # any response artifacts appeared; an established prior is immutable.
            response_started = (rd / 'data/continuum_prior.json').exists() or any((rd / 'data').glob('*1d*'))
            if action not in {'run', 'resume'} or response_started:
                raise
            options = read_settings(rd / 'param/OPTIONSCON')
            cp = _choose(rd, 'continuum', mpi_size, ncomp, mixed, int(options.get('MaxNumberSaves', '2000')))
            return {**plan, 'mode': 'resume', 'phase': 'continuum', 'checkpoint': cp,
                    'backup_torn_continuum': True,
                    'notice': 'Continuum posterior write was interrupted; restore its checkpoint and regenerate it before starting the response.'}
    if action in {'post', 'plot'}:
        return {**plan, 'mode': action}
    con_posterior = rd / 'data/posterior_sample.txt'
    rm_sample = rd / 'data' / f'sample1d.txt_{ncomp}'
    if rm_sample.is_file():
        _validate_prior_file(rd)
        alias = 'restart_dnest1d.txt' if mixed else f'restart_dnest1d.txt_{ncomp}'
        numbered = re.compile(re.escape(alias) + r'_\d+$')
        if not any(numbered.fullmatch(path.name) for path in (rd / 'data').glob(alias + '_*')):
            state.setdefault('frozen_files', {})['data/posterior_sample.txt'] = digest(con_posterior)
            return {**plan, 'mode': 'resume', 'phase': 'response_start',
                    'notice': 'No response checkpoint yet; preserve the completed continuum and restart only the response stage.'}
        cp = _choose(rd, 'response', mpi_size, ncomp, mixed, target_saves)
        state.setdefault('frozen_files', {})['data/posterior_sample.txt'] = digest(con_posterior)
        return {**plan, 'mode': 'resume',
                'phase': 'response', 'checkpoint': cp}
    if con_posterior.is_file():
        _validate_prior_file(rd)
        state.setdefault('frozen_files', {})['data/posterior_sample.txt'] = digest(con_posterior)
        return {**plan, 'mode': 'resume', 'phase': 'response_start'}
    if (rd / 'data/sample.txt').exists():
        options = read_settings(rd / 'param/OPTIONSCON')
        cp = _choose(rd, 'continuum', mpi_size, ncomp, mixed, int(options.get('MaxNumberSaves', '2000')))
        return {**plan, 'mode': 'resume', 'phase': 'continuum', 'checkpoint': cp}
    raise ValueError('The previous run stopped before a usable checkpoint. Use fresh to archive it and start again.')


def apply_plan(rd, plan):
    rd = Path(rd)
    stamp = time.strftime('%Y%m%dT%H%M%S') + f'-{time.time_ns() % 1000000000:09d}'
    if plan.get('archive_existing'):
        archive = rd.with_name(rd.name + '.before-fresh-' + stamp)
        rd.rename(archive)
        plan['archive'] = str(archive)
    rd.mkdir(parents=True, exist_ok=True)
    if plan['mode'] == 'fresh':
        return
    backup = rd / 'resume_history' / stamp
    backup.mkdir(parents=True)
    plan['backup'] = str(backup)
    for name in ['param/param_input', 'param/OPTIONSCON', 'param/OPTIONS1D', STATE_FILE]:
        source = rd / name
        if source.is_file():
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    cp = plan.get('checkpoint')
    if plan.get('backup_torn_continuum'):
        shutil.copy2(rd / 'data/posterior_sample.txt', backup / 'posterior_sample.txt')
    if plan.get('phase') == 'response_start':
        for path in (rd / 'data').glob('*1d*'):
            if path.is_file():
                shutil.copy2(path, backup / path.name)
    if cp:
        # Back up BOTH files before changing either. Existing prefixes are exact.
        for key in ['sample', 'info']:
            path = Path(cp[key])
            shutil.copy2(path, backup / path.name)
        for key in ['sample', 'info']:
            path = Path(cp[key])
            with path.open('rb') as source:
                prefix = source.read(cp[key + '_prefix'])
            tmp = path.with_name(path.name + '.resume-tmp')
            tmp.write_bytes(prefix)
            os.replace(tmp, path)
        alias = Path(cp['alias'])
        if alias.exists():
            shutil.copy2(alias, backup / alias.name)
        temp = alias.with_name(alias.name + '.resume-tmp')
        shutil.copy2(cp['path'], temp)
        os.replace(temp, alias)
    atomic_text(backup / 'resume.json', json.dumps(plan, indent=2) + '\n')


def record_started(rd, plan):
    rd = Path(rd)
    state = dict(plan['state'])
    state['status'] = 'running'
    state['last_mode'] = plan['mode']
    state['last_checkpoint'] = plan.get('checkpoint')
    state['attempt_started_at'] = time.time()
    frozen = state.setdefault('frozen_files', {})
    for name in [*[f'input/{name}' for name in state['inputs']], 'data/data_input.txt',
                 'param/ordered_gaussian_lag_priors.txt', 'param/joint_parameter_priors.txt']:
        if (rd / name).is_file():
            frozen[name] = digest(rd / name)
    plan['state'] = state
    atomic_text(rd / STATE_FILE, json.dumps(state, indent=2) + '\n')


def record_complete(rd, plan):
    rd = Path(rd)
    # Native code has some exit(0) failure paths. Completion needs real products.
    target = plan['state']['target_saves']
    ncomp = plan['state']['ncomp']
    dimension = response_dimension(rd, ncomp)
    for name, columns in [(f'sample1d.txt_{ncomp}', dimension), (f'sample_info1d.txt_{ncomp}', 4)]:
        result = _rows(rd / 'data' / name, columns)
        if result['rows'] != target:
            raise ValueError(f'{name} has {result["rows"]} rows, expected cumulative target {target}.')
    required = [f'posterior_sample1d.txt_{ncomp}', 'evidence.txt']
    if not plan['state']['settings'].get('_fit_only', False):
        required += [f'pall.txt_{ncomp}'] + [f'pline.txt_{ncomp}_comp{i}' for i in range(ncomp)]
    for name in required:
        if not (rd / 'data' / name).is_file() or (rd / 'data' / name).stat().st_size == 0:
            raise ValueError(f'MICA has not produced required output: {name}')
    if _rows(rd / 'data' / f'posterior_sample1d.txt_{ncomp}', dimension)['rows'] < 1:
        raise ValueError('MICA posterior contains no valid samples.')
    _validate_prior_file(rd)
    state = dict(plan['state'])
    state.setdefault('frozen_files', {})['data/posterior_sample.txt'] = digest(rd / 'data/posterior_sample.txt')
    state['status'] = 'complete'
    state['completed_at'] = time.time()
    if state.get('attempt_started_at') is not None:
        state['last_attempt_seconds'] = state['completed_at'] - state['attempt_started_at']
    state['continuum_prior'] = plan.get('continuum_prior', state.get('continuum_prior'))
    atomic_text(rd / STATE_FILE, json.dumps(state, indent=2) + '\n')
