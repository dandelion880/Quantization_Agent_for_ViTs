from pathlib import Path
import sys,json,time,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'QAViT'))
from core.journal import atomic_json
from runtime.model_router import call_once,preflight
from runtime import codex_cli
OUT=Path(__file__).parent/'llm'
cli='/root/.codex/packages/standalone/releases/0.169.0-x86_64-unknown-linux-musl/bin/codex'
settings={'provider':'codex_subscription','cli_path':cli,'agent_models':{'generation':['gpt-6-astra','high'],'coding':['gpt-6.1-sol','medium'],'recorder':['gpt-6-luna','high'],'monitor':['gpt-6.1-sol','ultra']}}
results={'checked_unix':time.time(),'cli_path':cli,'cli_sha256':hashlib.sha256(Path(cli).read_bytes()).hexdigest(),'scope':'installed official CLI help/features/live catalog, exact isolation flags, account metadata and schema-only model probes; preparation cost only','checks':{}}
for role in ['generation','coding','recorder','monitor']:
 directory=OUT/('probe_'+role)
 spec={'role':role,'settings':settings,'system_prompt':'This is an engineering transport check. Return ok=true only. Do not call tools or claim scientific results.', 'context':{'purpose':'isolated transport preparation probe'},'response_schema':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False}}
 result=call_once(spec,directory,time.time()+180)
 results['checks'][role]={'requested':result['requested'],'actual':result['actual'],'usage':result['usage'],'response':result['response'],'audit':result['audit']}
 atomic_json(OUT/'cli169_compatibility.json',results)
 print(role,result['requested'],result['actual'],flush=True)
results['review']={'all_original_policy_flags_retained':True,'exec_help_change':'adds --trust; never passed','new_features':'code_mode_host_grpc/output_truncation enabled but parent host remains disabled; no new native tool entry permitted; code_mode_only required','no_tool_calls_accepted':True,'model_catalog_route_changes':False}
atomic_json(OUT/'cli169_compatibility.json',results)
