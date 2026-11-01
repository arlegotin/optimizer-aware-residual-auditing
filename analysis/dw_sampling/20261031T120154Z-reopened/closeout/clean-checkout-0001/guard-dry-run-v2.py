# SPDX-License-Identifier: AGPL-3.0-only
"""Run the unmodified driver with no-Torch/no-body/no-network audit sentinels."""
import json, os, runpy, sys
from pathlib import Path
state=json.loads((Path(__file__).parent/'state.json').read_text())
root=Path.cwd().resolve();original=Path(state['original_root']).resolve()
assert str(root)==state['clone']
observed={'imports':[],'subprocesses':[],'violations':[]}
def deny(message):
 observed['violations'].append(message)
 raise RuntimeError(message)
def audit(event,args):
 if event=='import':
  name=args[0];observed['imports'].append(name)
  if name.split('.')[0] in ('torch','pyarrow') or name in ('oar.sampling_lm','oar.sampled_linear'):deny('body/training/device import: '+name)
 elif event=='open':
  path=args[0]
  if isinstance(path,(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(path)).resolve()
   if '.cache' in p.parts or p.suffix in ('.pt','.pth','.parquet','.safetensors'):deny('private body open: '+str(p))
   if p.is_relative_to(original) and not p.is_relative_to(original/'.venv'):deny('original source/cache fallback: '+str(p))
 elif event in ('socket.connect','socket.getaddrinfo'):deny('network event: '+event)
 elif event=='subprocess.Popen':
  command=args[1];observed['subprocesses'].append(command)
  if (not isinstance(command,list) or len(command)<2 or command[0]!='git'
      or command[1] not in ('cat-file','rev-parse','status')):deny('unexpected subprocess: '+repr(command))
sys.addaudithook(audit)
sys.argv=['analysis/run_dw_sampling.py',*sys.argv[1:]]
exit_code=0
try:runpy.run_path(str(root/'analysis/run_dw_sampling.py'),run_name='__main__')
except SystemExit as exc:exit_code=exc.code or 0
assert exit_code==0,exit_code
assert not observed['violations']
assert not any(n=='torch' or n.startswith('torch.') for n in sys.modules)
observed.update({'status':'pass','driver_argv':sys.argv,'exit_code':exit_code,'torch_imported':False,'corpus_or_checkpoint_opened':False,'network_accessed':False,'original_source_or_cache_used':False,'training_launched':False})
with (Path(state['base'])/'dry-run-sentinel.json').open('x') as f:json.dump(observed,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in observed.items() if k!='imports'},indent=2))
