from fedcore.algorithm.heterogeneous.selection import (
    select_finalist,compare_report_pair,SelectedFinalist,NoFeasibleFinalist,IncompleteSearch)


def test_selection_uses_real_deploy_cost_and_excludes_partial_searches():
    result = dict(status='succeeded',charged_evaluations=64,finite_archive=[
        dict(quality=.1,cost=60.,canonical_attempt_key='large',workflow_fingerprint='large',quality_accepted=True,genotype={}),
        dict(quality=.2,cost=40.,canonical_attempt_key='fit',workflow_fingerprint='fit',quality_accepted=True,genotype={})])
    selected = select_finalist(result,baseline_bytes=100)
    assert type(selected) is SelectedFinalist and selected.identity == 'fit'
    assert type(select_finalist(result,baseline_bytes=50)) is NoFeasibleFinalist
    result['charged_evaluations'] = 63
    assert type(select_finalist(result,baseline_bytes=100)) is IncompleteSearch


def test_primary_contrast_preserves_absent_and_incomplete_outcomes():
    selected = SelectedFinalist('a',.1,40,{})
    assert compare_report_pair(NoFeasibleFinalist(),selected)['e2_win'] == 1
    assert compare_report_pair(selected,NoFeasibleFinalist())['e2_win'] == 0
    assert compare_report_pair(NoFeasibleFinalist(),NoFeasibleFinalist())['e2_win'] == .5
    assert compare_report_pair(selected,IncompleteSearch(60,64,'time'))['status'] == 'incomplete_pair'
    result = compare_report_pair(selected,selected,e0_report_ce=.2,e2_report_ce=.3)
    assert result['e2_win'] == 0 and result['ce_difference'] > 0
