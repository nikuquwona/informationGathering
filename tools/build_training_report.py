"""Build an offline viewer from a versioned episode and paired evaluation."""
import argparse
import json
from pathlib import Path


def build(directory):
    directory=Path(directory)
    comparison=json.loads((directory/'comparison.json').read_text())
    episodes={name:json.loads((directory/f'{name}-episode.json').read_text()) for name in ('trained','greedy')}
    payload=dict(comparison=comparison,episodes=episodes)
    encoded=json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c')
    template=Path(__file__).with_name('training_report_template.html').read_text()
    if template.count('__TRAINING_DATA__')!=1:
        raise ValueError('Expected exactly one report payload slot')
    output=directory/'report.html'
    output.write_text(template.replace('__TRAINING_DATA__',encoded))
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args=parser.parse_args()
    print(build(args.directory))
