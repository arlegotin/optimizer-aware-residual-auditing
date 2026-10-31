# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded reproduction review: no hardware measurements or stage reopening."""
from pathlib import Path
import datetime, json, os, subprocess, sys, tempfile, time
ORIGINAL = Path('/Users/artemlegotin/Documents/git_alter/optimizer-aware-residual-auditing')
LOG = ORIGINAL/'analysis/dw_sampling/20261031T105213Z/reviews/clean_checkout'
PYTHON = str(ORIGINAL/'.venv/bin/python')
BASE = Path(tempfile.mkdtemp(prefix='dw-sampling-review-', dir='/tmp'))
CLONE = BASE/'checkout'
commands = []
def run(name, argv, cwd, env=None):
    start = time.perf_counter()
    r = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    (LOG/(name+'.stdout.txt')).write_text(r.stdout)
    (LOG/(name+'.stderr.txt')).write_text(r.stderr)
    item = dict(name=name, argv=argv, cwd=str(cwd), exit_code=r.returncode, duration_s=time.perf_counter()-start,
                environment={k:env[k] for k in ('PYTHONPATH','PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL')} if env else None)
    commands.append(item)
    (LOG/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print(name, r.returncode, r.stdout[-1000:], r.stderr[-500:], flush=True)
    return r
assert run('01-clone',['git','clone','--no-hardlinks','--no-checkout',str(ORIGINAL),str(CLONE)],BASE).returncode == 0
assert run('02-checkout',['git','checkout','--detach','48d02b26caac2fa24bd95eac4745b334e8d32af7'],CLONE).returncode == 0
ENV = dict(os.environ, PYTHONPATH=str(CLONE/'src'),PYTHONDONTWRITEBYTECODE='1', PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0',PYTORCH_MPS_PREFER_METAL='0')
(LOG/'locations.json').write_text(json.dumps(dict(base=str(BASE),clone=str(CLONE),python=PYTHON,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2)+'\n')
run('03-identity',[PYTHON,'-c', '''import sys,os,json,platform,subprocess,torch,oar,oar.sampled_linear
print(json.dumps(dict(executable=sys.executable,version=sys.version,machine=platform.machine(),platform=platform.platform(),mac_ver=platform.mac_ver(),torch_version=torch.__version__,torch_file=torch.__file__,oar_file=oar.__file__,sampler_file=oar.sampled_linear.__file__,mps_built=torch.backends.mps.is_built(),mps_visible=torch.backends.mps.is_available(),cpu_used_for_suite=True,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),shallow=subprocess.check_output(['git','rev-parse','--is-shallow-repository'],text=True).strip(),status=subprocess.check_output(['git','status','--porcelain'],text=True),flags={k:os.environ[k] for k in ('PYTHONPATH','PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','PYTORCH_MPS_PREFER_METAL')}),indent=2))'''],CLONE,ENV)
run('04-cpu-tests-sandbox',[PYTHON,'-m','pytest','-q','-m','not mps'],CLONE,ENV)
run('05-driver-help',[PYTHON,'analysis/run_dw_sampling.py','--help'],CLONE,ENV)
run('06-historical-renderer',[PYTHON,'-m','oar.cli','report','--gates','decisions/gates.jsonl','--run-index','analysis/evidence_index.json','--out',str(BASE/'historical-rendered.md')],CLONE,ENV)
run('07-summary',[PYTHON,'analysis/run_dw_sampling.py','--stage','summarize','--config','configs/dw_sampling.yaml','--out','analysis/dw_sampling/20261031T105213Z'],CLONE,ENV)
run('08-post-status',['git','status','--porcelain'],CLONE,ENV)
