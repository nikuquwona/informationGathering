import hashlib
import json
from pathlib import Path
import re

import pytest
from tools.build_replay import ROOT, build_payload, export, read_matrix


def test_export_uses_original_trajectory_and_hashes(tmp_path):
    out = tmp_path / 'replay.html'
    payload = export(ROOT, out)
    assert payload['runs']
    for run in payload['runs']:
        for i in range(3):
            assert run['tracks'][i] == read_matrix(ROOT / run['sources'][i])
    for name, digest in payload['sources'].items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    embedded = re.search(r'<script type="application/json" id="data">(.*?)</script>', out.read_text(), re.S)
    assert json.loads(embedded[1]) == payload
    assert '__REPLAY_DATA__' not in out.read_text()
    assert out.with_suffix('.manifest.json').is_file()


@pytest.mark.parametrize('text', ['', '1 2\n3\n', 'nan 2\n', '1 inf\n'])
def test_invalid_archives_are_rejected(tmp_path, text):
    p = tmp_path / 'bad.txt'
    p.write_text(text)
    with pytest.raises(ValueError):
        read_matrix(p)
