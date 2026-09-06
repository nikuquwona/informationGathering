"""Export recorded trajectories to a self-contained, offline research replay.

Only Python's standard library is required. No checkpoints are executed and no
historical map is regenerated. Every displayed datum has a source file hash.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read_matrix(path):
    rows = [[float(v) for v in line.split()] for line in path.read_text().splitlines() if line.strip()]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f'Empty or ragged matrix: {path}')
    if any(not math.isfinite(v) for row in rows for v in row):
        raise ValueError(f'Non-finite matrix: {path}')
    return rows


def build_payload(root=ROOT):
    nav_path = root / 'Maps/example_map copy.csv'
    nav = read_matrix(nav_path)
    sources = {}

    def source(path):
        relative = path.relative_to(root).as_posix()
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return relative

    source(nav_path)
    runs = []
    for first in sorted((root / 'forth/path').glob('final_eva_path1_*_eva_.txt')):
        match = re.fullmatch(r'final_eva_path1_(\d+)_eva_\.txt', first.name)
        if not match:
            continue
        run_id = match[1]
        paths = [first.with_name(f'final_eva_path{i}_{run_id}_eva_.txt') for i in (1, 2, 3)]
        if not all(p.is_file() for p in paths):
            continue
        tracks = [read_matrix(p) for p in paths]
        if len({len(t) for t in tracks}) != 1 or any(len(row) != 2 for t in tracks for row in t):
            raise ValueError(f'Run {run_id}: expected three equal-length (N, 2) trajectories')
        if any(not (0 <= row[0] < len(nav) and 0 <= row[1] < len(nav[0])) for t in tracks for row in t):
            raise ValueError(f'Run {run_id}: trajectory lies outside the navigation grid')
        snapshots, run_sources = {}, [source(p) for p in paths]
        for kind in ('mu', 'sigma'):
            path = root / f'mu_sig_map/final_{kind}_map_{run_id}_eva.txt'
            if path.exists():
                matrix = read_matrix(path)
                if (len(matrix), len(matrix[0])) != (len(nav), len(nav[0])):
                    raise ValueError(f'Run {run_id}: snapshot and grid shapes differ')
                snapshots[kind] = [[round(v, 6) for v in row] for row in matrix]
                run_sources.append(source(path))
        runs.append(dict(id=run_id, tracks=tracks, snapshots=snapshots, sources=run_sources))
    if not runs:
        raise ValueError('No complete three-agent historical trajectories found')
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    return dict(schema_version=1, revision=revision, navigation=nav, runs=runs, sources=sources,
                provenance='Archived trajectories and final snapshots; not a new policy evaluation.')


def export(root, output):
    payload = build_payload(root)
    # Escape '<' so even a future string containing </script> cannot escape JSON.
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    template = (ROOT / 'tools/replay_template.html').read_text()
    assert template.count('__REPLAY_DATA__') == 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace('__REPLAY_DATA__', data))
    output.with_suffix('.manifest.json').write_text(json.dumps({k: v for k, v in payload.items() if k not in ('navigation', 'runs')}, indent=2))
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/replay.html')
    args = parser.parse_args()
    result = export(ROOT, args.output.resolve())
    print(f'Exported {len(result["runs"])} archived runs to {args.output.resolve()}')
