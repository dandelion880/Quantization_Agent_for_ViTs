"""Feedback versus complete expansion, checked on real next request projections."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.cards import canonical
from tests.test_infrastructure import TempCase
from tests import test_feedback_delivery as feedback
from tests.support import meta

class FeedbackOverlap(TempCase):
    task=feedback.FeedbackDeliveryRegression.task
    begin=feedback.FeedbackDeliveryRegression.begin
    tools=feedback.FeedbackDeliveryRegression.tools
    chosen=feedback.FeedbackDeliveryRegression.chosen
    publish_meta=feedback.FeedbackDeliveryRegression.publish_meta

    def setup_case(self, large=False):
        wf=self.workflow(1);chosen=self.chosen(wf)
        value=meta(1,[chosen['id']]);value['judgment_audit'][0]['reasoning']='UNIQUE_DEFAULT_AUDIT'
        value['generation_guidance']='EXPANDED_ONLY_INFORMATION'+('x'*26000 if large else '')
        card=self.publish_meta(wf,value,[chosen['id']])
        task,c=self.begin(wf,self.task(wf,refs=[chosen['id']],round=2,hypothesis_ref=chosen['id']))
        self.assertEqual(canonical(c).count('UNIQUE_DEFAULT_AUDIT'),1)
        self.assertNotIn('EXPANDED_ONLY_INFORMATION',canonical(c))
        return wf,task,card

    def assert_once(self,wf,task):
        spec=canonical(wf.request_body(task)['spec'])
        self.assertEqual(spec.count('UNIQUE_DEFAULT_AUDIT'),1)
        self.assertEqual(spec.count('EXPANDED_ONLY_INFORMATION'),1)

    def test_short_whole_card_replaces_default_subfield(self):
        wf,task,card=self.setup_case()
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id']})])
        self.assert_once(wf,task)

    def test_ancestor_replaces_prior_child_and_default(self):
        wf,task,card=self.setup_case()
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'field':'/payload/judgment_audit/0'})])
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'field':'/payload'})])
        self.assert_once(wf,task)
        self.assertEqual(len(c['tool_results']),2)  # retain both reading records

    def test_complete_root_pages_replace_default_but_gap_keeps_it(self):
        wf,task,card=self.setup_case(True)
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'offset':0}),('read_card',{'ref':card['id'],'offset':24000})])
        self.assertTrue(any('value' in x for x in c['memory']['bound_feedback']))
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'offset':12000})])
        self.assert_once(wf,task)
        self.assertEqual(len(c['tool_results']),3)

    def test_complete_raw_card_asset_and_card_have_one_body(self):
        wf,task,card=self.setup_case()
        task,c=self.tools(wf,task,[('read_asset',{'asset':str(wf.journal.card_path(card['id']))})])
        self.assert_once(wf,task)
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id'],'field':'/payload'})])
        self.assert_once(wf,task)

    def test_read_other_version_does_not_elide_current_audit(self):
        wf,task,card=self.setup_case()
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id']})])
        chosen=wf.journal.card(task['inputs']['context']['hypothesis_ref'])
        value=meta(2,[chosen['id']]);value['judgment_audit'][0]['reasoning']='NEW_VERSION_AUDIT'
        new=self.publish_meta(wf,value,[chosen['id']])
        c=wf.request_body(task)['spec']['context']
        self.assertIn('NEW_VERSION_AUDIT',canonical(c))
        self.assertTrue(any(e.get('value',{}).get('reasoning')=='NEW_VERSION_AUDIT' for e in c['memory']['bound_feedback']))

    def test_same_id_unverified_result_survives_valid_complete_ancestor(self):
        from copy import deepcopy
        from unittest.mock import patch
        wf,task,card=self.setup_case()
        task,c=self.tools(wf,task,[('read_card',{'ref':card['id']})])
        rows=list(wf.result_values(task));row,child,result=rows[0]
        other=deepcopy(child);other['task_id']='unverified-result'
        rows.append((row,other,{'id':card['id'],'extra':'UNVERIFIED_BODY_MUST_SURVIVE'}))
        bad=deepcopy(child);bad['task_id']='bad-page'
        rows.append((row,bad,{'ref':card['id'],'text':'UNVERIFIED_PAGE_MUST_SURVIVE','sha256':'bad'}))
        with patch.object(wf,'result_values',side_effect=lambda _:iter(rows)):
            bodies=wf.complete_tool_bodies(task)
            output=wf.compact_results(task,{},complete_bodies=bodies)
        self.assertIn('UNVERIFIED_BODY_MUST_SURVIVE',canonical(output))
        self.assertIn('UNVERIFIED_PAGE_MUST_SURVIVE',canonical(output))
