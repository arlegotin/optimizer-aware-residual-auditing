# SPDX-License-Identifier: AGPL-3.0-only
from pathlib import Path
import datetime, json, os, shlex, subprocess, sys, time
state=json.loads((Path(__file__).parent/'state.json').read_text())
label,cwd,*command=sys.argv[1:]
out=Path(state['public'])
env=os.environ.copy()
env['PYTHONPATH']='src'
env['PYTHONDONTWRITEBYTECODE']='1'
env['HF_HUB_OFFLINE']='1'
env['HF_DATASETS_OFFLINE']='1'
env['TRANSFORMERS_OFFLINE']='1'
env['HF_HOME']=str(Path(state['base'])/'empty-hf-cache')
env['XDG_CACHE_HOME']=str(Path(state['base'])/'empty-xdg-cache')
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
wall=time.monotonic()
with (out/(label+'.stdout.txt')).open('x') as stdout, (out/(label+'.stderr.txt')).open('x') as stderr:
    result=subprocess.run(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr)
receipt={'command':command,'shell_command':shlex.join(command),'cwd':cwd,'environment_overrides':{k:env[k] for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','HF_HOME','XDG_CACHE_HOME')},'started_at_utc':start,'wall_seconds':time.monotonic()-wall,'exit_code':result.returncode}
with (out/(label+'.json')).open('x') as f: json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt,indent=2))
print((out/(label+'.stdout.txt')).read_text()[-14000:])
print((out/(label+'.stderr.txt')).read_text()[-6000:],file=sys.stderr)
sys.exit(result.returncode)
