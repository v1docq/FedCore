"""Approved fixed recovery/extra-BN matrix; no search or official test."""
from .cv_sparse_protocol import digest

MAX_SECONDS = 1800
WORKER_SECONDS = 1790
PRIOR_SECONDS = 3648.862081296742
COMPLETION_REPORT_SHA = '2aa82ef99dd7a3137900c191c2fd9d1890f3cdeca6311fedbefee43156d22303'
COMPLETION_INVENTORY_SHA = 'c83086ca9f152969aadac7cc41fb750db10aaf78628d430d277d5bd809738d1f'
DIAGNOSTIC_REPORT_SHA = 'a73489d743b541d58e267e9df75f01557499c92f552cd44c77909466a27848a9'
DIAGNOSTIC_INVENTORY_SHA = '8143a54166160a8f3f61bc8cc76bdc5c73956e496dcb5bd0ac6bf22005798ee7'
BASELINE_SHA = '531d02916a6a4ad7341f95a1f7e075c91c7b5350bc7009a09d5b93077c4d5510'
LEARNING_RATES = (.01, .001)
STAGES = (('manifest', 300), ('quality', 1200), ('close', 300))
# R is exactly the old fixed SVD, not a sixth independent family.
FAMILIES = (
    ('M0', '4bb353479eb43b34eebaadeeb8c9ecea779a4ea270444a2ee5e224260c6fe41b', 44734408),
    ('P75', 'ed4d9d408edac3745137924481ca983c7738900c4bbef8eee66a7d884f41c603', 33741256),
    ('R', '211ec99046465bfcf6bb3046f6798d2cba3a4cac6b2c09fae408074e9d010c40', 33593288),
    ('E', 'ee8d9a0d94d06a62cdd86e3837749454b79225b79f71cb0a653a16a972a7bd94', 33316040),
    ('hybrid', '62799456f5f1877195b8691469eee3b4c43c3564b6a699b33ed8efe30e329dfd', 25733576),
)


def payload():
    return {
        'schema': 'fedcore_recovery_extra_bn_matrix_v1',
        'max_seconds': MAX_SECONDS, 'worker_seconds': WORKER_SECONDS,
        'prior_seconds': PRIOR_SECONDS, 'total_authorized_seconds': 14400,
        'stages': dict(STAGES), 'families': [list(row) for row in FAMILIES],
        'learning_rates': list(LEARNING_RATES), 'SGD_trajectories': 10,
        'updates_per_trajectory': 200, 'total_updates': 2000,
        'states': ['no_SGD', 'SGD_lr_0.01', 'SGD_lr_0.001'],
        'extra_BN_modes': ['preserve_training_statistics', 'reset_and_recalibrate'],
        'quality_variants': 30, 'quality_rows_per_variant': 5000,
        'total_validation_rows': 150000, 'extra_BN_rows': 1000,
        'BN_factor': 'extra reset/calibration; training BN still updates normally',
        'common_tape_seed': 990032, 'batch': 128,
        'optimizer': 'fresh SGD; momentum=.9; nesterov=True; weight_decay=.0005',
        'original_M0_fresh_build_for_each_family': True,
        'no_implicit_extra_BN_during_build': True,
        'dense_six_predictions_exact_replay_required': True,
        'official_test_access': False, 'new_search': False,
        'baseline_training_from_scratch': False, 'latency_sessions': 0,
        'automatic_retry': False,
        'completion_report_sha256': COMPLETION_REPORT_SHA,
        'completion_inventory_sha256': COMPLETION_INVENTORY_SHA,
        'diagnostic_report_sha256': DIAGNOSTIC_REPORT_SHA,
        'diagnostic_inventory_sha256': DIAGNOSTIC_INVENTORY_SHA,
    }


def sha256():
    return digest(payload())
