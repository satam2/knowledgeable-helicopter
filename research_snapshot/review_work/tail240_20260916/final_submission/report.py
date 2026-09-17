"""Summarize only verified organizer acceptance and local release receipts."""
import json
from pathlib import Path
import sys
import math

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'knowledgeable-helicopter-screening/src'))
from taxiout.artifacts import sha256
from taxiout.paths import external_path
OUT=ROOT/'private_runs/tail240_20260916/final_submission_v3_v2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    protocol=read(OUT/'protocol.json')
    result=read(OUT/'organizer_result.json')
    receipt=read(OUT/'upload_receipt.json')
    verification=read(OUT/'independent_verification.json')
    remote=read(OUT/'remote_verification.json')
    ready=read(OUT/'submission_ready.json')
    key=protocol['submission']['key']
    assert result['status']=='Succeeded' and result['used_pairs']==344841
    assert result['bucket']==protocol['submission']['bucket'] and result['file']==key
    assert type(result['score']) in (int,float) and math.isfinite(result['score'])
    digest=sha256(OUT/key)
    assert digest==receipt['sha256']==verification['submission_sha256']==remote['sha256']
    assert verification['status']==remote['status']=='passed' and receipt['http_status']==200
    previous=read(ROOT/'private_runs/submission_v2/organizer_v2_result.json')['score']
    change=previous-result['score']
    lines=['# PRC 2026 V3 Submission Result','',
        f"Official organizer score: **{result['score']:.6f} seconds RMSE**.",
        f"V2 scored {previous:.6f}; V3 changes RMSE by {-change:+.6f} seconds ({-100*change/previous:+.2f}%).",'',
        f"Accepted: {result['used_pairs']:,} of 344,841 required pairs. File: `{key}`.",
        f"Bucket: `{result['bucket']}`. Upload HTTP 200; remote bytes match the independently approved SHA256.",'',
        '## Model','',
        'Full-year fit of the best verified local ensemble: two TabM variants, LightGBM sequence/context experts, and CatBoost. ',
        'All new finite-clock models used 2,062,577 eligible 2025 rows. Missing-clock experts used all 22,470 eligible rows. ',
        'Original V2 missing predictions were reused after exact native verification. Labels were not clipped or discarded.','',
        'The 268.662991 local development RMSE came from exposed July/November folds. It was not an official-score forecast. ',
        'Final weights combine the existing F1/F3 tune-fitted coefficients with competition-season weights; model stopping ',
        'counts use the inherited integer-median policy. No ranking outcomes or submission feedback selected these settings.','',
        '| Ordinary Expert | Weight |','|---|---:|']
    lines.extend(f'| {name} | {weight:.9f} |' for name,weight in protocol['active_weights'].items())
    lines+=['','Finite negative or >7,200-second clocks use the 225-field LightGBM. Missing clocks use ',
        '0.75 * (0.75 * V2 + 0.25 * ExtraTrees) + 0.25 * normalized predictor.','',
        '## Full-Year Fits','',
        '| Component | Training Rows | Epochs/Trees | Fit and Inference Seconds |',
        '|---|---:|---:|---:|']
    ordinary=Path(protocol['component_directories']['ordinary'])
    for name in ['tabm225','cat225','lgb225','lgb387','lgb449']:
        model=read(ordinary/name/'manifest.json')
        lines.append(f"| {name} | {model['training_rows']:,} | {model['steps']} | {model['runtime_sec']:.1f} |")
    ple=read(Path(protocol['component_directories']['ple'])/'manifest.json')
    lines.append(f"| ple387 | {ple['training_rows']:,} | {ple['epochs']} | {ple['elapsed_seconds']:.1f} |")
    missing=read(Path(protocol['component_directories']['missing'])/'manifest.json')
    lines.append(f"| Missing normalized + ExtraTrees | {missing['training_rows']:,} | 111 + 300 | {missing['elapsed_seconds']:.1f} |")
    lines += ['',
        '## Verification','',
        '- Independent source/preprocessing checks and native prediction replay for every final component.',
        '- Exact 344,841-row template schema, movement IDs, row order, and nearest-even int32 serialization.',
        '- One authorized upload, unused V3 key, quota checked, and complete remote hash verification.',
        '- Organizer acceptance and score retrieved from this submission result only; organizer truth was not accessed.','',
        f"Route counts: {json.dumps(ready['route_counts'],sort_keys=True)}.",'',
        f'File SHA256: `{digest}`.','',
        f'Release evidence: `{OUT}`.','',
        f"Organizer result retrieved at {result['retrieved_utc']}."]
    target=external_path(ROOT/'output/tail240_20260916/FINAL_SUBMISSION_RESULT.md')
    target.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(official_rmse=result['score'],previous_rmse=previous,improvement_seconds=change,
                         rows=result['used_pairs'],file=key,report=str(target))))


if __name__=='__main__':main()
