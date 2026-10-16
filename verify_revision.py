"""Independent checks for the revised experiment, without a second simulator."""
import json
from pathlib import Path
import numpy as np
import evaluation_controls as c
import evaluation_original as old
from run_revision import replay, verify_auc

verify_auc()
c.set_seed(42)
tx=c.gen_reconciliation_corpus(); legacy=old.reconcile(tx); revised=c.reconcile(tx)
assert set(legacy).issubset(revised)
recovered=[t for t in tx if t['id'] in revised and t['id'] not in legacy]
assert len(recovered)==31
assert {t['label'] for t in recovered}=={'missing_promo_credit'}
assert not [t for t in tx if t['label'] is None and t['id'] in revised]
for seed in range(42,52):
    c.set_seed(seed); tx=c.gen_reconciliation_corpus()
    for floor in (.10,.05,.01):
        pred=c.reconcile(tx,floor=floor)
        assert all(pred[t['id']][0]==t['label'] for t in tx
                   if t['label']=='missing_promo_credit' and t['id'] in pred)

case=dict(at=1,mag=.1,sign=1,ref=np.ones(4)*1000,noise=np.zeros(4),factors=np.ones(3))
clean=dict(at=-1,mag=0,sign=1,ref=np.ones(4)*1000,noise=np.zeros(4),factors=np.ones(3))
assert replay([case,clean],0.,'posthoc')['material_receiving_steps_per_injected_trace']==2
assert replay([case,clean],0.,'inline')['material_receiving_steps_per_injected_trace']==0
assert replay([case,clean],0.,'direct_read')['material_receiving_steps_per_injected_trace']==0
result=json.loads((Path(__file__).parent/'results.revision.json').read_text())
for amp in (.07,.1):
    observed=result['boundary_noise_sweep'][str(amp)]['clean_fpr']['mean']
    assert abs(observed-(1-.05/amp))<.002
f=result['primary']['fraud']
assert f['universal_0.65']['precision']>f['cohort_0.6']['precision']
assert f['universal_0.65']['recall']>f['cohort_0.6']['recall']
assert sum(g['fp'] for name,g in f['universal_0.6']['groups'].items() if name in ['fashion','shoes'])==58
assert sum(g['fp'] for name,g in f['cohort_0.6']['groups'].items() if name in ['fashion','shoes'])==0
print('PASS: tie handling, credit recovery, class consistency, replay counters, analytical FPR, threshold and subgroup results.')
