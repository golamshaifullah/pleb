"""Training I/O helpers. All PLEB configs and stage calls live in the notebooks."""
from collections import Counter
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile

import pandas as pd
import tomlkit

sys.path.insert(0, '/timing-course')
import course_tools as simulation

PULSARS = ['J1234+5678', 'J1235+5678']


def command(args, cwd=None, log=None):
    print(shlex.join([str(x) for x in args]))
    result = subprocess.run([str(x) for x in args], cwd=cwd, text=True, capture_output=True)
    text = result.stdout + result.stderr
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        Path(log).write_text(text)
    print(text[-6000:])
    result.check_returncode()
    return result.stdout


def write_config(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomlkit.dumps(values))
    print(path.read_text())
    return path


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob('*')) if p.is_file() and '.git' not in p.parts}


def verify_hashes(root, expected):
    actual = hashes(root)
    assert actual == expected, 'Missing, changed or extra files; inspect the manifest difference'
    print('Verified', len(expected), 'files')


def snapshot(repo, branch, destination):
    """Export a named dataset branch, without changing the main worktree."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'{destination}: use a new export name')
    raw = subprocess.check_output(['git', '-C', str(repo), 'archive', branch, 'data'])
    destination.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter='data')
    return destination / 'data'


def inventory(dataset):
    from pleb.release_bundle import parse_tim
    records = []
    for path in sorted(Path(dataset).glob('J*/tims/*.tim')):
        rows = parse_tim(path.read_text(), str(path.relative_to(dataset)))
        identities = Counter(r.identity for r in rows if not r.commented)
        records.append(dict(pulsar=path.parent.parent.name, backend=path.stem,
                            physical=len(rows), active=sum(not r.commented for r in rows),
                            commented=sum(r.commented for r in rows),
                            extra_active_duplicates=sum(n-1 for n in identities.values())))
    return pd.DataFrame(records)


def prepare_raw(root):
    """Generate two artificial pulsars, two backends, duplicates and isolated bad TOAs."""
    raw = root / 'raw'
    record = root / 'raw_manifest.json'
    if record.exists():
        verify_hashes(raw, json.loads(record.read_text())['sha256'])
        return json.loads(record.read_text())
    for name in ['par', 'A', 'B', 'clockfiles']:
        (raw/name).mkdir(parents=True, exist_ok=True)
    injected = []
    original_par = simulation.TRUTH_PAR
    try:
        for number, psr in enumerate(PULSARS):
            simulation.TRUTH_PAR = original_par.replace(PULSARS[0], psr).replace(
                'RAJ 12:34:00', f'RAJ 12:{34+number}:00')
            work = simulation.project(root/'simulation'/psr)
            runtime = simulation.synthetic_runtime(work)
            simulation.create_synthetic(work, seed=simulation.SEED+number)
            shutil.copy2(work.inputs/'truth.par', raw/'par'/f'{psr}.par')
            rows = simulation.tim_metadata(work.inputs/'synthetic.tim')
            payload = [line for line in (work.inputs/'synthetic.tim').read_text().splitlines()
                       if len(line.split()) >= 5 and line.split()[4] == 'train']
            by_backend = {'A': [], 'B': []}
            for i, line in enumerate(payload):
                parts = line.split()
                backend = parts[parts.index('-sys')+1]
                if i in (40, 171):
                    parts[2] = str(Decimal(parts[2]) + Decimal('0.000050')/Decimal(86400))
                    injected.append(dict(pulsar=psr, backend=backend, mjd=parts[2], offset_us=50))
                by_backend[backend].append(' '.join(parts))
            for backend, lines in by_backend.items():
                # A duplicate is another occurrence of the exact same measurement.
                (raw/backend/f'{psr}.tim').write_text('FORMAT 1\n'+'\n'.join(lines+[lines[10]])+'\n')
            shutil.copy2(runtime/'clock/train2utc.clk', raw/'clockfiles/train2utc.clk')
    finally:
        simulation.TRUTH_PAR = original_par
    manifest = dict(synthetic=True, pulsars=PULSARS, expected_physical_rows=580,
                    duplicate_occurrences=4, injected_bad_toas=injected, sha256=hashes(raw))
    record.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def environment(root):
    root = Path(root)
    work = simulation.project(root/'simulation'/PULSARS[0])
    simulation.synthetic_runtime(work)
    for name in ['configs', 'logs', 'review', 'exports', 'results']:
        (root/name).mkdir(parents=True, exist_ok=True)
    return root
