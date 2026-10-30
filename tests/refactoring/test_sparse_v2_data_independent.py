"""Graph and hash-order oracles independent of the v2 data planner."""
import hashlib

import pytest
from hypothesis import given, settings, strategies as st

pytest.importorskip('transformers', reason='BERT experiments require the llm extra')
from fedcore.experiments.sparse_pretrained import Article
from fedcore.experiments.sparse_pretrained_v2 import plan_article_groups, select_role_windows


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def graph_components(articles):
    remaining = set(range(len(articles)))
    result = []
    while remaining:
        reached = {min(remaining)}
        while True:
            extended = reached | {j for j in remaining if any(
                articles[j].title == articles[i].title or articles[j].text_hash == articles[i].text_hash
                for i in reached)}
            if extended == reached:
                break
            reached = extended
        result.append(reached)
        remaining -= reached
    return result


@given(st.lists(st.tuples(st.integers(0, 4), st.integers(0, 4),
                         st.sampled_from(['train', 'validation', 'test'])), min_size=1, max_size=12))
@settings(max_examples=45, deadline=None)
def test_component_ownership_matches_graph_reachability(rows):
    articles = [Article(sha(f'title-{title}'), f'title-{title}', f'body-{text}', sha(f'body-{text}'), split)
                for title, text, split in rows]
    actual = plan_article_groups(articles)
    priority = {'test': 0, 'validation': 1, 'train': 2}
    for component in graph_components(articles):
        canonical = min(articles[i].article_id for i in component)
        owner = min((articles[i].split for i in component), key=priority.get)
        role = owner if owner != 'train' else ('recovery' if int(canonical, 16) % 2 else 'calibration')
        assert sum(actual[i]['eligible'] for i in component) == 1
        for i in component:
            record = actual[i]
            assert (record['canonical_id'], record['owner_split'], record['role']) == (canonical, owner, role)
            assert set(record['group_titles']) == {articles[j].title for j in component}
            assert set(record['group_text_hashes']) == {articles[j].text_hash for j in component}
            if record['eligible']:
                assert articles[i].split == owner
    def signatures(values, records):
        return sorted((r['canonical_id'], r['role'], a.article_id, a.text_hash, a.split)
                      for a, r in zip(values, records) if r['eligible'])
    reversed_articles = list(reversed(articles))
    assert signatures(articles, actual) == signatures(reversed_articles, plan_article_groups(reversed_articles))


def test_selection_hashes_round_robin_and_offsets_are_explicit():
    articles = [Article(sha(f'paper-{i}'), f'paper-{i}', f'body-{i}', sha(f'body-{i}'), 'validation')
                for i in range(6)]
    token_table = {a.text: list(range(1000 + 400*i, 1000 + 400*i + 62*5)) for i, a in enumerate(articles)}
    records = plan_article_groups(articles)
    result = select_role_windows(articles, records, 'validation', token_table.__getitem__, 101, 102, 4, 2)
    windows, ids, units, descriptions = result
    chosen = sorted(articles, key=lambda a: (sha(f'local-bert-mlm-v2|article|validation|{a.article_id}'), a.article_id))[:4]
    expected_units = [a.article_id for a in chosen] * 2
    assert units == expected_units and len(set(ids)) == 8
    for round_index in range(2):
        for i, article in enumerate(chosen):
            starts = sorted(sorted(range(0, 62*5, 62), key=lambda start: (
                sha(f'local-bert-mlm-v2|window|validation|{article.article_id}|{start}'), start))[:2])
            position = 4*round_index + i
            start = starts[round_index]
            assert descriptions[position]['start'] == start
            assert windows[position] == [101, *token_table[article.text][start:start+62], 102]
    reversed_articles = list(reversed(articles))
    assert result == select_role_windows(reversed_articles, plan_article_groups(reversed_articles),
        'validation', token_table.__getitem__, 101, 102, 4, 2)


def test_duplicate_aliases_cannot_supply_requested_independent_units():
    articles = [Article(sha('same'), 'same', 'body', sha('body'), 'test')] * 3
    with pytest.raises(ValueError, match='Insufficient independent'):
        select_role_windows(articles, plan_article_groups(articles), 'test', lambda _: list(range(200)),
                            101, 102, 2, 2)
