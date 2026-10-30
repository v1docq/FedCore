"""Recount a sealed series from persisted predictions/ledgers, without models."""
from __future__ import annotations

from dataclasses import asdict
import html
import json
import math
from pathlib import Path
import statistics

from .cv_independent_protocol import MODELS, SEARCH_SEEDS, digest, primary_aggregate


def hypervolume_2d(points, reference=(1.5, .8)):
    """Minimization union area against one fixed, predeclared reference."""
    frontier = sorted((float(x), float(y)) for x, y in points
                      if math.isfinite(x) and math.isfinite(y) and x < reference[0] and y < reference[1])
    area, best_y = 0., reference[1]
    for x, y in frontier:
        if y < best_y:
            area += (reference[0]-x)*(best_y-y)
            best_y = y
    return area


def recount_predictions(root, metrics):
    import numpy as np
    from .cv_sparse_runtime import sha256_file
    root = Path(root).resolve()
    path = (root/metrics['prediction_file']).resolve()
    if not path.is_relative_to(root) or sha256_file(path) != metrics['prediction_sha256']:
        raise ValueError('Prediction evidence changed before recount')
    with np.load(path, allow_pickle=False) as rows:
        ids, labels, predictions, ce = (rows[name] for name in ('row_ids', 'labels', 'predictions', 'cross_entropy'))
        if (len(ids) != metrics['rows'] or len(set(ids.tolist())) != len(ids)
                or not all(len(values) == len(ids) for values in (labels, predictions, ce))
                or not np.isfinite(ce).all()):
            raise ValueError('Incomplete, duplicated or nonfinite prediction rows')
        correct, loss = int(np.sum(labels == predictions)), float(np.mean(ce))
        if correct != metrics['correct'] or not math.isclose(loss, metrics['cross_entropy'], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError('Independent prediction recount differs from stored metrics')
        return {'rows': len(ids), 'correct': correct, 'top1': correct/len(ids), 'cross_entropy': loss}


def primary_results(seal, quality):
    completeness = primary_aggregate(seal.outcomes)
    outcomes = {o.job: o for o in seal.outcomes}
    pairs, model_aggregates = [], {}
    for model in MODELS:
        model_rows = []
        for seed in SEARCH_SEEDS:
            random = outcomes[f'{model.name}_{seed}_random']
            evolution = outcomes[f'{model.name}_{seed}_evolution']
            row = {'model': model.name, 'seed': seed, 'random': asdict(random), 'evolution': asdict(evolution),
                   'complete_q32': random.complete_q32 and evolution.complete_q32,
                   'delta_ce': None, 'delta_top1': None}
            if random.finalist in quality and evolution.finalist in quality:
                r, e = quality[random.finalist]['eager'], quality[evolution.finalist]['eager']
                row.update(delta_ce=e['cross_entropy']-r['cross_entropy'], delta_top1=e['top1']-r['top1'])
            model_rows.append(row)
            pairs.append(row)
        complete = completeness['model_complete'][model.name] and all(r['delta_ce'] is not None for r in model_rows)
        model_aggregates[model.name] = {'complete': complete, 'pair_count': len([r for r in model_rows if r['delta_ce'] is not None]),
            'mean_delta_ce': statistics.mean(r['delta_ce'] for r in model_rows) if complete else None,
            'median_delta_ce': statistics.median(r['delta_ce'] for r in model_rows) if complete else None}
    complete = all(row['complete'] for row in model_aggregates.values())
    means = [r['mean_delta_ce'] for r in model_aggregates.values()] if complete else []
    return {'complete': complete, 'independent_trainings': 3, 'pairs': pairs, 'models': model_aggregates,
            'panel_mean_delta_ce': statistics.mean(means) if complete else None,
            'panel_median_delta_ce': statistics.median(means) if complete else None}


def write_analysis(root, seal, artifacts, quality, searches, controls, windows):
    from .cv_sparse_runtime import Output
    root = Path(root)
    # Output already belongs to the controller. Reuse its atomic writer without
    # invoking its new-directory constructor.
    output = object.__new__(Output)
    output.root, output.ledger, output.heartbeat = root, root/'ledger.jsonl', root/'heartbeats.jsonl'
    recount = {key: {name: recount_predictions(root, row[name]) for name in ('eager', 'ts', 'test') if name in row}
               for key, row in quality.items()}
    primary = primary_results(seal, quality)
    curves, points = [], []
    for job, search in searches.items():
        model, seed, strategy = job.split('_')
        counterpart = f'{model}_{seed}_{"evolution" if strategy == "random" else "random"}'
        common_cap = min(search.get('charged', 0), searches.get(counterpart, {}).get('charged', 0))
        dense_ce = controls[model]['dense_recovery']['validation']['cross_entropy']
        records = search.get('attempts', [])
        for record in records:
            if record['status'] == 'succeeded' and record['completed_within_deadline']:
                points.append({'model': model, 'seed': int(seed), 'strategy': strategy,
                    'attempt': record['attempt_id'], 'ce': record['cross_entropy'],
                    'bytes_ratio': record['cost_ratio'], 'solution': record['solution_id']})
        for k in (4, 8, 16, 24, 32):
            if common_cap < k:
                continue
            usable = [r for r in records if r['attempt_id'] <= k and r['status'] == 'succeeded'
                      and r['completed_within_deadline']]
            compact = [r for r in usable if r['storage_accepted']]
            curves.append({'model': model, 'seed': int(seed), 'strategy': strategy, 'budget': k,
                'successful': len(usable), 'best_compact_ce': min((r['cross_entropy'] for r in compact), default=None),
                'hypervolume': hypervolume_2d([(r['cross_entropy']/dense_ce, r['cost_ratio']) for r in usable]),
                'normalization_dense_recovery_ce': dense_ce})
    counters = {'baseline_updates': 0, 'recovery_updates': 0, 'training_row_presentations': 0,
                'charged_numerical_attempts': 0, 'timing_pairs': 0}
    for path in sorted((root/'workers').glob('*/ledger.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                event = json.loads(line)
                kind = event.get('event')
                if kind in ('baseline_update', 'recovery_update'):
                    counters['baseline_updates' if kind == 'baseline_update' else 'recovery_updates'] += 1
                    counters['training_row_presentations'] += event['rows']
                elif kind == 'candidate_charged':
                    counters['charged_numerical_attempts'] += 1
                elif kind == 'latency_pair':
                    counters['timing_pairs'] += 1
    # The compact quality flag is descriptive; it never selects an artifact.
    quality_flags = {}
    role_map = dict(seal.roles)
    for role, key in seal.roles:
        model = role.split('_')[0]
        dense = role_map[f'{model}_dense_recovery']
        quality_flags[role] = {name: quality[key][name]['top1'] >= quality[dense][name]['top1']-.02
                              for name in ('eager', 'test') if name in quality[key] and name in quality[dense]}
    result = {'primary': primary, 'recount': recount, 'search_curves': curves, 'search_points': points,
              'counters': counters, 'quality_flags': quality_flags,
              'outcomes': [asdict(o) for o in seal.outcomes],
              'timing': windows, 'seal_sha256': seal.sha256,
              'limits': 'CIFAR10, one architecture, three independently trained M0; test historically used; timing one A100 allocation'}
    output.json('analysis.json', result)
    lines = ['# Независимые M0: результаты фиксированного протокола', '',
        f"Основная панель: {'полная' if primary['complete'] else 'неполная'}. Независимых обучений: 3.", '',
        'Официальный test ранее использовался; это повторная описательная оценка. '
        'Поисковые повторы и временные окна не увеличивают число независимых обучений.', '',
        '| M0 | seed | R: попытки | E: попытки | Полная пара Q32 | ΔCE E−R, Vreport |',
        '|---|---:|---:|---:|---|---:|']
    for row in primary['pairs']:
        delta = 'нет обоих финалистов' if row['delta_ce'] is None else f"{row['delta_ce']:.8f}"
        lines.append(f"| {row['model']} | {row['seed']} | {row['random']['charged']} | "
                     f"{row['evolution']['charged']} | {'да' if row['complete_q32'] else 'нет'} | {delta} |")
    lines.extend(['', 'Основной агрегат публикуется только при трёх полных парах с двумя финалистами '
        'для каждой из трёх M0. Отказы и незавершённые задания сохранены в `analysis.json` и `report.json`.', '',
        f"Фактически подтверждено журналами: {counters['baseline_updates']} обновлений M0, "
        f"{counters['recovery_updates']} обновлений восстановления и {counters['timing_pairs']} временных пар.", '',
        'Ускорение допускается только после собственных A/A, обоих предварительных dense A/A, '
        'заключительного dense A/A и успешного завершения соответствующего окна. '
        'Сохранены все три назначенных окна; провал не заменялся повторным запуском.', ''])
    output.bytes('RESULTS.md', '\n'.join(lines).encode('utf-8'))
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Независимые M0</title>
<style>body{font:15px system-ui;margin:2rem;max-width:1300px;color:#182236}select{margin:.5rem;padding:.4rem}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:.5rem;border-bottom:1px solid #ddd;text-align:left}
svg{width:100%;height:320px;background:#f6f8fb}pre{white-space:pre-wrap} .note{background:#fff3da;padding:1rem}</style>
<h1>Независимые M0: Random / Evolution</h1><p class="note">Независимых обучений n=3. Test ранее использовался.
Неполные пары показываются отдельно; автоматического усреднения удачных повторов нет.</p>
<p id="status"></p><label>M0 <select id="model"><option value="">Все</option><option>M1</option><option>M2</option><option>M3</option></select></label>
<label>seed <select id="seed"><option value="">Все</option><option>140201</option><option>140202</option><option>140203</option></select></label>
<label>Бюджет <select id="budget"><option>4</option><option>8</option><option>16</option><option>24</option><option selected>32</option></select></label>
<h2>Поиск: CE и физические байты</h2><svg id="plot" viewBox="0 0 1000 320"></svg><p>Синий — Random, красный — Evolution. Точки показывают своевременные успешные попытки. Вертикаль: CE, горизонталь: байты/M0.</p>
<h2>Кривые по оплаченному бюджету</h2><div id="curves"></div><h2>Все девять пар</h2><div id="pairs"></div>
<h2>Исходы всей панели</h2><div id="outcomes"></div><h2>Счётчики по журналам</h2><pre id="counters"></pre>
<h2>Три окна времени и допуски</h2><details><summary>Показать полные записи</summary><pre id="timing"></pre></details>
<p>Источники: соседние analysis.json, global_seal.json, quality.json, workers/*/ledger.jsonl и массивы predictions/*.npz. Для независимого пересчёта модели не требуются.</p>
<script>const data=PAYLOAD;const $=id=>document.getElementById(id);
function esc(x){return String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function table(rows,keys){return '<table><tr>'+keys.map(k=>'<th>'+esc(k)+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+keys.map(k=>'<td>'+esc(typeof r[k]==='object'?JSON.stringify(r[k]):r[k])+'</td>').join('')+'</tr>').join('')+'</table>';}
function draw(){const m=$('model').value,s=$('seed').value,b=+$('budget').value,keep=r=>(!m||r.model===m)&&(!s||r.seed===+s);
const pts=data.search_points.filter(r=>keep(r)&&r.attempt<=b);const mx=Math.max(1,...pts.map(p=>p.ce));
$('plot').innerHTML='<text x="12" y="20">CE</text><text x="850" y="310">байты / M0</text>'+pts.map(p=>'<circle cx="'+(45+Math.min(1.1,p.bytes_ratio)*820)+'" cy="'+(280-p.ce/mx*245)+'" r="4" fill="'+(p.strategy==='random'?'#2763c5':'#d23f47')+'"><title>'+esc(JSON.stringify(p))+'</title></circle>').join('');
$('curves').innerHTML=table(data.search_curves.filter(r=>keep(r)&&r.budget<=b),['model','seed','strategy','budget','successful','best_compact_ce','hypervolume']);
$('pairs').innerHTML=table(data.primary.pairs.filter(keep),['model','seed','complete_q32','delta_ce','delta_top1']);}
$('status').textContent='Основная панель: '+(data.primary.complete?'полная':'неполная')+'; ΔCE = CE_E − CE_R, отрицательное значение лучше для E.';
$('outcomes').innerHTML=table(data.outcomes,['job','status','charged','complete_q32','finalist']);
$('counters').textContent=JSON.stringify(data.counters,null,2);$('timing').textContent=JSON.stringify(data.timing,null,2);
for(const id of ['model','seed','budget'])$(id).onchange=draw;draw();</script></html>'''.replace('PAYLOAD', payload)
    output.bytes('RESULTS.html', page.encode('utf-8'))
    return {'json': 'analysis.json', 'markdown': 'RESULTS.md', 'html': 'RESULTS.html',
            'primary_complete': primary['complete'], 'counters': counters}
