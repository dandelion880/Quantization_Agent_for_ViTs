"""Display-only optimizations preserve tail counterevidence, exact paths and JSON thresholds."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.cards import canonical, fingerprint
from core.evidence import collection_summary, project_observations, _bounded_json_size


class EvidenceProjectionSpeedRegression(unittest.TestCase):
    def test_size_threshold_matches_canonical_with_escaped_and_unicode_values(self):
        values = [{}, [], {'斜线/~': '换行\n' * 20, 'float': -1.1e200}, list(range(1000)),
                  [{'k': ['\\', '"', None, True, -0.0]} for _ in range(100)]]
        for value in values:
            exact = len(canonical(value))
            for limit in (0, 1, 40, exact - 1, exact, exact + 1):
                with self.subTest(limit=limit, exact=exact):
                    size = _bounded_json_size(value, limit)
                    self.assertEqual(size > limit, exact > limit)
                    if exact <= limit:
                        self.assertEqual(size, exact)

    def test_complete_streaming_statistics_keep_tail_extrema_and_first_ties(self):
        rows = [{'x': value, 'state': str(i % 10), 'boolean': bool(i % 2), 'null': None}
                for i, value in enumerate([2., 2., 0., -1., 10., -20.])]
        result = collection_summary(rows, 'source', '/payload/a~1b', max_characters=20000)
        stats = result['numeric_summary']['/*/x']
        self.assertEqual(stats['minimum'], {'value': -20., 'field': '/payload/a~1b/5/x'})
        self.assertEqual(stats['maximum'], {'value': 10., 'field': '/payload/a~1b/4/x'})
        self.assertEqual((stats['negative'], stats['zero'], stats['count']), (2, 1, 6))
        self.assertNotIn('/*/boolean', result['numeric_summary'])
        self.assertEqual(result['sha256'], fingerprint(rows))
        tied = collection_summary([{'x': 2}, {'x': 2}], 'source', '/v')
        self.assertEqual(tied['numeric_summary']['/*/x']['minimum']['field'], '/v/0/x')

    def test_site_grouping_paths_and_omitted_conflicts_remain_visible(self):
        value = {f'blocks.{i}.attn': {f'a/~{j}': (-1 if i == 29 else i) + j / 100}
                  for i in range(30) for j in [0]}
        result = collection_summary(value, 'source', '/payload/observations', max_characters=20000)
        stats = result['numeric_summary']['/*site*/a~1~00']
        self.assertEqual(stats['count'], 30)
        self.assertEqual(stats['minimum']['field'], '/payload/observations/blocks.29.attn/a~1~00')
        wide = [{f'error_{j}': (-1 if i == 39 else i) for j in range(50)} for i in range(40)]
        result = project_observations(wide, 'source', max_characters=1000)
        self.assertGreater(result['display_coverage']['omitted_mixed_sign_metrics'], 0)
        self.assertEqual(result['display_coverage']['mixed_sign_metrics_total'], 50)

    def test_small_views_are_independent_and_long_scientific_context_is_locatable(self):
        value = {'conditions': {'n': 32}, 'rows': [1, 2]}
        view = project_observations(value, 'source')
        view['conditions']['n'] = 99
        self.assertEqual(value['conditions']['n'], 32)
        value = {'conditions': 'Long condition\n\n' + 'x' * 20000,
                 'rows': [{'error': -i} for i in range(3000)]}
        view = project_observations(value, 'source')
        self.assertEqual(view['conditions']['read_card']['field'], '/payload/observations/conditions')
        self.assertFalse(view['conditions']['complete'])


class JournalFieldAllocationRegression(unittest.TestCase):
    def test_first_field_read_does_not_copy_the_full_cached_card(self):
        import threading
        import core.journal as journal
        from core.cards import make_card
        from tests.support import hypothesis
        card = make_card('HypothesisCard', 'run', hypothesis(1, 'H1'))
        store = journal.Journal.__new__(journal.Journal)
        store.root = Path('.')
        store.cards = {card['id']: {'sha256': card['content_sha256']}}
        store._mutex, store._read_cache = threading.RLock(), None
        with patch.object(journal, 'read_json', return_value=card), patch.object(Path, 'stat'), \
                patch.object(journal, 'deepcopy', wraps=journal.deepcopy) as copy:
            with store.read_session():
                self.assertEqual(store.card(card['id'], field=('kind',)), 'HypothesisCard')
                self.assertEqual([call.args[0] for call in copy.call_args_list], ['HypothesisCard'])
        self.assertIsNone(store._read_cache)


if __name__ == '__main__':
    unittest.main()
