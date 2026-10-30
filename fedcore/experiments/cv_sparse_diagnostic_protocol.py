"""Frozen, explicitly authorized BN/SGD and TorchScript diagnostic workload."""
from .cv_sparse_protocol import digest

MAX_SECONDS = 1800
WORKER_SECONDS = 1790
PRIOR_SECONDS = 3491.18732554093
COMPLETION_REPORT_SHA = '2aa82ef99dd7a3137900c191c2fd9d1890f3cdeca6311fedbefee43156d22303'
COMPLETION_INVENTORY_SHA = 'c83086ca9f152969aadac7cc41fb750db10aaf78628d430d277d5bd809738d1f'
BASELINE_SHA = '531d02916a6a4ad7341f95a1f7e075c91c7b5350bc7009a09d5b93077c4d5510'
STAGES = (('manifest', 300), ('quality', 600), ('timing', 600), ('close', 300))
LEARNING_RATES = (.01, .001)
TIMING_ARTIFACTS = (
    ('M0', '5d6e2584b61161197937c8f1e14102a4e9ca4581d2eb5fa30b05ad207d68fd8d'),
    ('hybrid', '60391830390f675fdab67b9b65aaa31afd74957785bc3de11cf51e9a8c03f1b8'),
    ('R', '13578b48aa4f4e231ff4c8f1238c74e52e611d4d3ff22cbb502b4d13674adc09'),
    ('E', 'f15740ae470e5a1b10a447878392dd9bab0892d23328a076bd8256476d63883b'),
)


def payload():
    return {
        'schema': 'fedcore_bn_sgd_torchscript_diagnostic_v1',
        'max_seconds': MAX_SECONDS, 'worker_seconds': WORKER_SECONDS,
        'prior_seconds': PRIOR_SECONDS, 'stages': dict(STAGES),
        'learning_rates': list(LEARNING_RATES), 'SGD_trajectories': 2,
        'updates_per_trajectory': 200, 'total_updates': 400,
        'quality_variants': 6, 'quality_rows_per_variant': 5000,
        'extra_BN_rows': 1000, 'official_test_access': False,
        'BN_factor': 'extra reset/calibration; training BN still updates normally',
        'common_tape_seed': 990032, 'batch': 128,
        'optimizer': 'fresh SGD; momentum=.9; nesterov=True; weight_decay=.0005',
        'timing_artifacts': [list(row) for row in TIMING_ARTIFACTS],
        'load_policies': ['legacy_cpu_then_cuda_A_vs_direct_cuda_B', 'canonical_both_direct_cuda'],
        'condition_order': ['M0 canonical/legacy', 'hybrid legacy/canonical', 'R canonical/legacy', 'E legacy/canonical'],
        'AA_sessions': 9, 'AB_sessions_max': 3, 'pairs_per_batch': 20,
        'batches': [1, 128], 'timing_calls': 20, 'timing_warmup': 20,
        'clocks': ['cuda_event', 'synchronized_wall'],
        'same_TS_probe_atol': 1e-5, 'same_TS_probe_rtol': 1e-5,
        'final_M0_AA_before_AB': True,
        'AB_gate': 'full canonical candidate AA and both initial/final canonical M0 AA',
        'automatic_retry': False, 'M0_training': False, 'new_search': False,
        'completion_report_sha256': COMPLETION_REPORT_SHA,
        'completion_inventory_sha256': COMPLETION_INVENTORY_SHA,
    }


def sha256():
    return digest(payload())
