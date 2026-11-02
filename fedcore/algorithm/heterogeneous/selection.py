"""Sealed pre-report selection; missing or incomplete searches remain explicit."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SelectedFinalist:
    identity: str
    quality: float
    deployment_bytes: int
    genotype: dict


@dataclass(frozen=True)
class NoFeasibleFinalist:
    reason: str = 'No finite candidate satisfies the deployment byte constraint'


@dataclass(frozen=True)
class IncompleteSearch:
    charged: int
    expected: int
    reason: str


def select_finalist(result, *, baseline_bytes, attempt_limit=64, storage_ratio=.5):
    if (type(baseline_bytes) is not int or baseline_bytes < 1
            or type(attempt_limit) is not int or attempt_limit < 1
            or type(storage_ratio) not in (int,float) or not math.isfinite(storage_ratio)
            or not 0 < storage_ratio <= 1):
        raise ValueError('Explicit comparable baseline package and selection limits required')
    charged = result['charged_evaluations']
    if result['status'] != 'succeeded' or charged != attempt_limit:
        return IncompleteSearch(charged,attempt_limit,result.get('stop_reason','unknown'))
    finalists = []
    for row in result['finite_archive']:
        quality, cost = row['quality'], row['cost']
        if (type(quality) not in (int,float) or not math.isfinite(quality)
                or type(cost) not in (int,float) or not math.isfinite(cost) or cost < 1
                or int(cost) != cost):
            raise ValueError('Finite CE and actual integer deployment bytes required')
        identity = row.get('canonical_attempt_key',row['workflow_fingerprint'])
        if row['quality_accepted'] and cost <= baseline_bytes * storage_ratio:
            finalists.append(SelectedFinalist(identity,quality,int(cost),row['genotype']))
    return min(finalists,key=lambda x:(x.quality,x.deployment_bytes,x.identity)) if finalists else NoFeasibleFinalist()


def compare_report_pair(e0, e2, *, e0_report_ce=None, e2_report_ce=None):
    """Primary E2-E0 win with feasibility first; infrastructure gaps are not losses."""
    allowed = (SelectedFinalist,NoFeasibleFinalist,IncompleteSearch)
    if type(e0) not in allowed or type(e2) not in allowed: raise ValueError('Typed selections required')
    if type(e0) is IncompleteSearch or type(e2) is IncompleteSearch:
        return dict(status='incomplete_pair',e2_win=None,ce_difference=None)
    if type(e0) is NoFeasibleFinalist and type(e2) is NoFeasibleFinalist:
        return dict(status='both_infeasible',e2_win=.5,ce_difference=None)
    if type(e0) is NoFeasibleFinalist or type(e2) is NoFeasibleFinalist:
        return dict(status='one_feasible',e2_win=float(type(e2) is SelectedFinalist),ce_difference=None)
    if any(type(c) not in (int,float) or not math.isfinite(c) or c < 0 for c in (e0_report_ce,e2_report_ce)):
        raise ValueError('Both independently obtained Vreport CE values required')
    difference = e2_report_ce-e0_report_ce
    return dict(status='both_feasible',e2_win=1. if difference < 0 else 0. if difference > 0 else .5,
                ce_difference=difference)
