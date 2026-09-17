"""Freeze complete campaign source snapshots and declare routed direct probes."""
import shutil
from common import HERE, OUT, read_json, write_json, sha256, utc_now, external_path


def main():
    hashes={}
    for p in HERE.rglob('*.py'):
        if '__pycache__' in p.parts:
            continue
        digest=sha256(p)
        dest=external_path(OUT/'source_snapshots'/digest/p.name)
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
        hashes[str(p.relative_to(HERE))]=digest
    stamp=utc_now().replace(':','').replace('.','')
    write_json(OUT/'source_receipts'/f'{stamp}.json',dict(created_utc=utc_now(),hashes=hashes,
        note='Content-addressed immutable snapshots; XGBoost also depends on lgbm_adapter.NativeFrameEncoder.'))
    amendment=OUT/'direct_routes_declaration.json'
    if not amendment.exists():
        write_json(amendment,dict(created_utc=utc_now(),runner_sha256=sha256(HERE/'run.py'),
            variants=['candidate','blend25','missing_only','missing_blend25'],
            status='Declared in source before first direct fit; registry omission corrected before direct score inspection',
            routing='Observed missing NM clock only; all other reference predictions preserved. No target-defined routing.',
            blending='fixed 25 percent candidate; no scoring-label weight fitting'))
    print('Preserved source files',len(hashes))


if __name__=='__main__':
    main()
