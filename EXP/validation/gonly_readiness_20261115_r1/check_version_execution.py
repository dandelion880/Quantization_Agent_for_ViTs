"""Host-only exact candidate binding through plan -> request -> execution card.
No numeric worker/model call; the real GPU check is separate.
"""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'QAViT'))
from tests.test_gonly import GOnlyRegression
from tests.support import experiment_payload
from core.cards import ContractError
from tests.test_plan_execution_consistency import numeric_plan

class BindingExecution(GOnlyRegression):
 def runTest(self):
  wf,plan=self.drive(self.workflow_case(),choice='H2',stop=lambda t:t['inputs']['purpose']=='plan_l3')
  expected={'selection_ref':plan['inputs']['context']['selection_ref'],
            'hypothesis_ref':plan['inputs']['context']['hypothesis_ref']}
  def draft():
   value=experiment_payload('L3')
   model_plan=numeric_plan()
   for row in model_plan['layers']:
    row['activation'].update(bits=4,format='INT4')
    row['weight'].update(bits=4,format='INT4')
   value.update(measurements=['top1'],scope={})
   value['execution']={'service':'quantize_evaluate','plan':model_plan,'seed':42,
      'resolved_choices':{'training':False},'calibration':{'split':'calibration','count':2},
      'evaluations':[{'kind':'classification','split':'final','count':2}]}
   return value
  wrong=draft();wrong['candidate_binding']={**expected,'hypothesis_ref':'wrong@1'}
  with self.assertRaises(ContractError):wf.submit(plan,wrong,None)
  wf.submit(plan,draft(),None)
  wf.advance();request=wf.tasks()['r001/L3']
  self.assertEqual(request['inputs']['draft']['candidate_binding'],expected)
  wf.queue.ready();wf.progress_request(wf.queue.claim(request['task_id']))
  request=wf.queue.get(request['task_id'])
  card=wf.journal.card(request['experiment_ref'])
  self.assertEqual(card['payload']['candidate_binding'],expected)
  self.assertIn(expected['hypothesis_ref'],card['refs'])
  self.assertIn(expected['selection_ref'],card['refs'])

if __name__=='__main__':
 result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([BindingExecution()]))
 raise SystemExit(not result.wasSuccessful())
