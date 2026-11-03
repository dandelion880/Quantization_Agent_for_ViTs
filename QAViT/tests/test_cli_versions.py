"""CPU CLI upgrade gates; no model calls or real account operations."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import codex_cli
from runtime.subscription import SubscriptionTransport
from ptq_agent.schemas import PTQError


class CLIVersionGates(unittest.TestCase):
    def client(self):
        client = SubscriptionTransport.__new__(SubscriptionTransport)
        client.cli = '/cpu-fixture/codex'
        client.model, client.effort = 'cpu-fixture-model', 'medium'
        return client

    def preflight(self, version, *, tool_mode='code_mode_only', effort='medium'):
        model = {'slug': 'cpu-fixture-model', 'supported_reasoning_levels': [{'effort': effort}],
                 'context_window': 272000, 'tool_mode': tool_mode}
        with patch('runtime.subscription.cli.run', side_effect=[
                (0, version + '\n', ''), (0, json.dumps({'models': [model]}), '')]), \
                patch('runtime.subscription.cli.account_status', return_value={'auth_type': 'chatgpt'}):
            return self.client().preflight()

    def test_reviewed_0166_and_legacy_0164_have_distinct_version_receipts(self):
        old = self.preflight('codex-cli 0.164.0')
        new = self.preflight('codex-cli 0.166.0')
        self.assertEqual(new['cli_version'], 'codex-cli 0.166.0')
        self.assertNotEqual(old['version_review'], new['version_review'])
        self.assertIn('cli166_resume_20261031_r1', new['version_review'])

    def test_unknown_and_similar_versions_fail_before_account_or_model_catalog(self):
        for version in ('codex-cli 0.165.0', 'codex-cli 0.166.1', 'codex-cli 0.166.0-dev', 'codex-cli 1.0.0'):
            with self.subTest(version=version), \
                    patch('runtime.subscription.cli.run', return_value=(0, version, '')) as command, \
                    patch('runtime.subscription.cli.account_status') as account:
                with self.assertRaises(PTQError) as error:
                    self.client().preflight()
                self.assertEqual(error.exception.code, 'llm_cli_version')
                self.assertEqual(command.call_count, 1)
                account.assert_not_called()

    def test_new_version_does_not_relax_model_effort_or_tool_isolation(self):
        for kwargs in ({'tool_mode': 'native'}, {'effort': 'low'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(PTQError) as error:
                self.preflight('codex-cli 0.166.0', **kwargs)
            self.assertEqual(error.exception.code, 'llm_tool_isolation')

    def test_pinned_missing_executable_does_not_fall_back(self):
        with patch('runtime.codex_cli.shutil.which', return_value='/unreviewed/newer/codex') as which:
            with self.assertRaises(PTQError) as error:
                codex_cli.executable({'cli_path': '/not-installed/pinned-codex'})
            self.assertEqual(error.exception.code, 'llm_not_configured')
            which.assert_not_called()

    def test_command_keeps_schema_no_native_tools_and_readonly(self):
        command = self.client().command(Path('/cpu-fixture/job'))
        for flag in ('--json', '--output-schema', '--output-last-message', '--strict-config',
                     '--ignore-user-config', '--ignore-rules', '--ephemeral'):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index('--sandbox') + 1], 'read-only')
        disabled = [command[i + 1] for i, value in enumerate(command[:-1]) if value == '--disable']
        self.assertTrue({'shell_tool', 'unified_exec', 'code_mode_host', 'multi_agent', 'apps', 'plugins'} <= set(disabled))
