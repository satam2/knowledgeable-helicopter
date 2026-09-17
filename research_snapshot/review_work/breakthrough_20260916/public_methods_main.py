"""Fetch specified public source documents only, outside both Git checkouts."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'models_retrieval/retrospective_research'))
import public_solution_check as fetcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', nargs=2, action='append', required=True)
    args = parser.parse_args()
    fetcher.OUT = HERE.parents[1] / 'output/breakthrough_20260916/research_gap/public_methods_main'
    fetcher.OUT.mkdir(parents=True, exist_ok=True)
    for key, url in args.source:
        if not url.startswith(('https://api.github.com/', 'https://raw.githubusercontent.com/')):
            raise ValueError('Public GitHub metadata/source endpoints only')
        if not key.replace('_', '').isalnum():
            raise ValueError('Simple artifact key required')
        fetcher.fetch(key, url)


if __name__ == '__main__':
    main()
