# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded native-only numerical diagnosis; never a sampler/learning result."""
import time
START = time.perf_counter()
import json
import hashlib
import os
from pathlib import Path
import torch
from oar.sampling_lm import ModelFixture, load_worker_data


def main():
    assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] == '0'
    assert torch.backends.mps.is_available()
    torch.set_float32_matmul_precision('highest')
    root = Path('analysis/dw_sampling/20261031T120154Z-reopened')
    manifest = json.loads((root/'pilot/attempt-0001/data_manifest.json').read_text())
    file = manifest['processed_files']['fitting-development']
    data = load_worker_data(file['path'],file['sha256'],phase='development')
    cases=[]
    for name,left,right,shared in [('independent_single_AA','single','single',False),
                                  ('independent_fused_scalar','fused','single',False),
                                  ('identical_gradient_fused_scalar','fused','single',True)]:
        actual=ModelFixture(data,base_seed=61001,variant=left)
        reference=ModelFixture(data,base_seed=61001,variant=right)
        assert actual.model.initial_parameters_sha256==reference.model.initial_parameters_sha256
        if shared:
            native_step=actual.opt.step
            def paired_step(*args,**kwargs):
                reference.opt.zero_grad(set_to_none=True)
                for key,p in actual.p.items():
                    reference.p[key].grad=p.grad.detach().clone()
                result=native_step(*args,**kwargs)
                reference.opt.step()
                return result
            actual.opt.step=paired_step
        rows=[]
        for step in range(1,6):
            actual.step()
            if not shared:reference.step()
            torch.mps.synchronize()
            worst=[];total=0;max_parameter=max_gradient=max_moment=0.
            for key,p in actual.p.items():
                q=reference.p[key]
                difference=(p.detach()-q.detach()).abs()
                count=int((difference>(1e-6+1e-4*q.detach().abs())).sum().item())
                error=float(difference.amax().item());grad_error=float((p.grad-q.grad).abs().amax().item())
                moment_error=float((actual.opt.state[p]['exp_avg']-reference.opt.state[q]['exp_avg']).abs().amax().item())
                total+=count;max_parameter=max(max_parameter,error);max_gradient=max(max_gradient,grad_error);max_moment=max(max_moment,moment_error)
                if count:worst.append({'parameter':key,'mismatches':count,'max_parameter_abs':error,'max_gradient_abs':grad_error,'max_moment_abs':moment_error})
            rows.append({'step':step,'parameter_mismatches_fixed_tolerance':total,'max_parameter_abs':max_parameter,
                         'max_gradient_abs':max_gradient,'max_moment_abs':max_moment,'failing_parameters':worst})
        cases.append({'case':name,'shared_gradient_injection':shared,'steps':rows})
        del actual,reference
    result={'kind':'bounded native-only diagnosis','cases':cases,'data_sha256':file['sha256'],
            'tolerances':{'rtol':1e-4,'atol':1e-6},'wall_s_including_torch_import':time.perf_counter()-START,
            'sampler_used':False,'protected_evaluation_used':False}
    with (Path(__file__).parent/'result.json').open('x') as out:json.dump(result,out,indent=2)
    for case in cases:print(case['case'],[(r['step'],r['parameter_mismatches_fixed_tolerance'],r['max_parameter_abs'],r['max_gradient_abs']) for r in case['steps']])


if __name__=='__main__':
    main()
