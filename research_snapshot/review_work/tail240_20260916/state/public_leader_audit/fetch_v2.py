"""Repeat only the four public GETs with network access after sandbox socket denial."""
import fetch

fetch.OUT = fetch.external_path(fetch.ROOT / 'output/tail240_20260916/state/public_leader_audit/v2')
if __name__ == '__main__':
    fetch.main()
