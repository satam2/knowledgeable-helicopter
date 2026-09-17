"""Receipt-only interpretation of the parent-scheduled exact-input CUDA probe."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audit


def main():
    root = audit.OUT
    cpu = audit.read_json(root / 'gru_cross_device_probe_cpu.json')
    gpu = audit.read_json(root / 'gru_cross_device_probe_cuda.json')
    full = audit.read_json(root / 'gru_audit_v4.json')
    queue = audit.BASE / 'gpu_queue_1302'
    job = audit.read_json(queue / 'results.json')['gru_same_tensor_probe']
    assert job['status'] == 'complete' and job['exit_code'] == 0
    assert audit.sha256(queue / 'gru_same_tensor_probe.log') == job['log_sha256']
    comparisons = {}
    for fold in ('F1', 'F3'):
        fixture = root / f'gru_{fold}_cross_device_fixture.joblib'
        assert cpu[fold]['fixture_sha256'] == gpu[fold]['fixture_sha256'] == audit.sha256(fixture)
        assert cpu[fold]['rows'] == gpu[fold]['rows'] == 144
        assert gpu[fold]['max_delta_from_original_saved_GPU_sec'] < .05
        assert full['models'][f'context_{fold}']['status'] == 'tolerance_exceeded'
        comparisons[fold] = {'exact_input_fixture_rows': 144,
            'CPU_fixture_vs_original_GPU_max_sec': cpu[fold]['max_delta_from_original_saved_GPU_sec'],
            'CUDA_fixture_vs_original_GPU_max_sec': gpu[fold]['max_delta_from_original_saved_GPU_sec'],
            'CUDA_fixture_vs_full_CPU_max_sec': gpu[fold]['max_delta_from_independent_CPU_sec'],
            'full_CPU_vs_GPU_status_unchanged': 'tolerance_exceeded',
            'fixture_sha256': gpu[fold]['fixture_sha256']}
    result = {'status': 'reviewed_device_path_discrepancy', 'source_sha256': audit.sha256(__file__),
        'CPU_probe_sha256': audit.sha256(root / 'gru_cross_device_probe_cpu.json'),
        'CUDA_probe_sha256': audit.sha256(root / 'gru_cross_device_probe_cuda.json'),
        'full_CPU_audit_sha256': audit.sha256(root / 'gru_audit_v4.json'),
        'queue_log_sha256': job['log_sha256'], 'folds': comparisons,
        'interpretation': 'Identical independently reconstructed encodedinputs andsavedweights reproduceoriginalGPU muchmorecloselyonCUDA thanCPU. Deviceexecution arithmetic isdominant explanationonprobedrows; residualsameGPU/batch difference remains.',
        'limitations': '144selectedrows/fold with16worstfullCPUdeltas+128random, notfullGPUreplay. OriginalGPU batch1024 vsprobe144; no exactGPUequivalence claim. StrictfullCPU0.05sectolerancefailure remainsvisible.',
        'promotion': False, 'new_GPU_launch_by_auditor': False}
    audit.write_json(root / 'closing_1254/gru_probe_review.json', result)
    print('GRU_PROBE_REVIEW', comparisons, flush=True)


if __name__ == '__main__':
    main()
