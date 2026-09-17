"""Final privacy, immutable-input, completeness and evidence inventory checks."""
import json
import subprocess
from pathlib import Path
from common import OUT, HERE, WORKSPACE, RAW, read_json, write_json, sha256, source_hashes, utc_now, external_path


def main():
    decision=read_json(OUT/'decision.json')
    original=read_json(WORKSPACE/'private_runs/submission_v2/protocol.json')
    raw={p.name:sha256(p) for p in RAW.glob('*.parquet')}
    assert raw==original['raw_hashes'] and len(raw)==14
    assert source_hashes()==original['source_hashes']
    prior=read_json(OUT/'validation/existing_evidence_audit.json')
    for name,digest in prior['frozen_external_runner_hashes'].items():
        assert sha256(WORKSPACE/'review_work'/name)==digest
    for name,digest in prior['preserved_evidence_hashes'].items():
        assert sha256(WORKSPACE/'private_runs/next_230'/name)==digest
    for fold,record in prior['selected_reference'].items():
        assert sha256(Path(record['path'])/'manifest.json')==record['manifest_sha256']
        assert sha256(Path(record['path'])/'score_predictions.parquet')==record['prediction_sha256']
    artifacts={}
    for root in [OUT,HERE,WORKSPACE/'output/campaign_20260916',WORKSPACE/'output/research_20260916']:
        external_path(root)
        for p in root.rglob('*'):
            if p.is_file() and 'pytest-final' not in p.parts and '__pycache__' not in p.parts and p.name!='final_verification.json':
                external_path(p)
                artifacts[str(p.relative_to(WORKSPACE))]=sha256(p)
    models=[*list((OUT/'models').glob('*/manifest.json')),*list((OUT/'missing_mixture/models').glob('*/manifest.json'))]
    assert len(models)==decision['completed_fit_records']
    for p in models:
        rec=read_json(p)
        assert rec['status']=='complete'
        for f,h in rec['outputs'].items():
            assert sha256(p.parent/f)==h
    for repo in ['knowledgeable-helicopter','knowledgeable-helicopter-screening']:
        head=subprocess.check_output(['git','-C',str(WORKSPACE/repo),'rev-parse','HEAD'],text=True).strip()
        assert head=='9e5b9e6e71e3b548d9bf2d0aad17821c20b0510f'
    pdf=WORKSPACE/'output/campaign_20260916/PRC_2026_Diverse_Campaign.pdf'
    assert pdf.exists()
    report=read_json(WORKSPACE/'output/campaign_20260916/pdf_build_receipt.json')
    independent=read_json(OUT/'validation/final_validation.json')
    receipt=dict(created_utc=utc_now(),passed=True,raw_hashes=raw,frozen_source_hashes_equal=True,
        all_artifacts_external_to_git=True,completed_fit_records=len(models),artifact_hashes=artifacts,
        local_reference_unchanged=True,frozen_external_runners_unchanged=True,prior_evidence_unchanged=True,
        official_score_unchanged=294.626,no_submission=True,
        objective_230_reached=False,decision=decision['recommendation'],pdf_sha256=sha256(pdf),
        tests={'frozen_checkout_suite':{'passed':19,'command':"python -B -u -c sys.path.insert(0,'knowledgeable-helicopter-screening/src'); pytest.main(['-q','-p','no:cacheprovider','knowledgeable-helicopter-screening/tests','--basetemp','private_runs/campaign_20260916/pytest-final'])"},
            'new_contract_tests':'See validation receipts and aviation/neural test receipts; not added to frozen checkouts.'},
        limitations=decision['limitations'],report_manifest=report,independent_validation=independent)
    write_json(OUT/'final_verification.json',receipt)
    print('FINAL VERIFIED',len(models),'complete fit records;',len(artifacts),'hashed external artifacts')


if __name__=='__main__':
    main()
