"""Reproduce the v9 paper. Original model preserved; revisions are explicit.

Run: python run_revision.py. Requires only NumPy plus the standard library.
Seeds 42--51; common random numbers across each sensitivity comparison.
"""
from pathlib import Path
import json
import platform
import hashlib
import numpy as np
import evaluation_controls as c
import evaluation_original as old

OUT = Path(__file__).resolve().parent

def binary(tx, pred):
    tp = sum(t['label'] is not None and t['id'] in pred for t in tx)
    fp = sum(t['label'] is None and t['id'] in pred for t in tx)
    pos = sum(t['label'] is not None for t in tx)
    f1s=[]
    for kind in c.ANOMALY_TYPES:
        correct=sum(t['label']==kind and t['id'] in pred and pred[t['id']][0]==kind for t in tx)
        actual=sum(t['label']==kind for t in tx)
        proposed=sum(v[0]==kind for v in pred.values())
        f1s.append(2*correct/(actual+proposed) if actual+proposed else 0.)
    return dict(tp=tp, fp=fp, fn=pos-tp, n_anomalies=pos,macro_f1=float(np.mean(f1s)),
                precision=tp/(tp+fp) if tp+fp else 0., recall=tp/pos,
                clean_fpr=fp/(len(tx)-pos))

def mean_sd(rows):
    return {k: dict(mean=float(np.mean([x[k] for x in rows])),
                    sd=float(np.std([x[k] for x in rows], ddof=1))) for k in rows[0]}

def promo_audit(tx):
    pred = old.reconcile(tx)
    audit = dict(threshold=0, prior_suppression=0, missed_credit_dollars=0., total_credit_dollars=0.)
    prices=[]
    for t in tx:
        if t['label'] != 'missing_promo_credit': continue
        f,p,s = c.expected_components(t['cat'],t['price'],t['weight'],t['enrolled'])
        audit['total_credit_dollars'] += -p
        if t['id'] in pred: continue
        delta = sum(t['actual'].values()) - (f+p+s)
        key='threshold' if abs(delta)<=max(.10,.005*(f+s)) else 'prior_suppression'
        audit[key]+=1; audit['missed_credit_dollars'] += -p; prices.append(t['price'])
    audit['median_missed_price']=float(np.median(prices))
    return audit

def fraud(sellers, mode, threshold):
    scores,y = c.fraud_scores(sellers,mode=='cohort')
    pred = scores>=threshold
    auc,tpr,fpr = c._roc_auc(scores,y)
    out=dict(auc=auc,threshold=threshold,tp=int(sum(pred&y)),fp=int(sum(pred&~y)),
             recall=float(sum(pred&y)/sum(y)),precision=float(sum(pred&y)/sum(pred)),
             clean_fpr=float(sum(pred&~y)/sum(~y)),
             fpr_at_tpr_ge_085=float(fpr[np.searchsorted(tpr,.85)]))
    out['groups']={}
    for group in c.COHORTS:
        g=np.array([s['cohort']==group for s in sellers]); pos=int(sum(y&g)); neg=int(sum(~y&g))
        tp=int(sum(pred&y&g)); fp=int(sum(pred&~y&g))
        out['groups'][group]=dict(positive=pos,negative=neg,tp=tp,fp=fp,recall=tp/pos if pos else None,fpr=fp/neg)
    return out

def replay_cases(seed,n=2500):
    """Independent reference trajectory; no generated observation enters it."""
    r=np.random.default_rng(seed+1000); cases=[]
    for _ in range(n):
        length=int(r.integers(4,9)); inject=bool(r.random()<.07)
        at=int(r.integers(1,length-1)) if inject else -1
        mag=float(np.exp(r.uniform(np.log(.02),np.log(.60)))) if inject else 0.
        sign=1 if r.random()<.5 else -1
        base=float(r.uniform(500,20000)); factors=1+r.uniform(-.002,.002,length-1)
        ref=np.r_[base,base*np.cumprod(factors)]
        noise=r.uniform(-1,1,length)
        cases.append(dict(at=at,mag=mag,sign=sign,ref=ref,noise=noise,factors=factors))
    return cases

def replay(cases,amplitude,mode):
    """Count executed receiving steps >5% from the independent reference.

    The same exogenous innovations are replayed in all three modes. Inline
    checks outgoing values before a receiver consumes them. Direct-read
    replaces each incoming value with its reference. No latency cost modeled.
    """
    consumed=0; clean_halts=0; n_clean=0; n_inj=0; completed_clean=0
    for tr in cases:
        ref=tr['ref']; value=ref[0]; halted=False; injected=tr['at']>=0
        n_inj+=injected; n_clean+=not injected
        for k in range(len(ref)):
            if mode=='direct_read': value=ref[k]
            if injected and k>tr['at'] and abs(value-ref[k])/ref[k]>.05:
                consumed+=1
            output=value*(1+amplitude*tr['noise'][k])
            if k==tr['at']: output*=1+tr['sign']*tr['mag']
            if k==len(ref)-1: continue
            if mode=='inline' and abs(output-ref[k])/ref[k]>.05:
                halted=True; break
            value=output*tr['factors'][k]
        if not injected:
            clean_halts+=halted; completed_clean+=not halted
    return dict(material_receiving_steps_per_injected_trace=consumed/n_inj,
                clean_trace_halt_rate=clean_halts/n_clean,
                injected_traces=n_inj,clean_traces=n_clean)

def boundary_sweep(seed,amplitude,n=100000):
    """Independent one-boundary trials, inclusive noise on anomalous values.

    Prevalence 7%; errors log-uniform 2--60%, random sign. All uniforms are
    sampled before amplitude is applied to preserve common random numbers.
    """
    r=np.random.default_rng(seed+2000)
    y=r.random(n)<.07; mag=np.exp(r.uniform(np.log(.02),np.log(.60),n))
    sign=np.where(r.random(n)<.5,1,-1); unit=r.uniform(-1,1,n)
    residual=amplitude*unit+y*sign*mag
    pred=np.abs(residual)>.05
    return dict(precision=float(sum(pred&y)/sum(pred)),
                recall=float(sum(pred&y)/sum(y)),
                clean_fpr=float(sum(pred&~y)/sum(~y)),
                material_recall=float(sum(pred&y&(mag>.05))/sum(y&(mag>.05))))

def verify_auc():
    for seed in range(20):
        r=np.random.default_rng(seed); scores=r.integers(0,5,70); labels=r.random(70)<.3
        a=c._roc_auc(scores,labels)[0]; pos=scores[labels,None]; neg=scores[~labels][None,:]
        exact=float(np.mean((pos>neg)+.5*(pos==neg)))
        assert abs(a-exact)<1e-12
        order=r.permutation(70)
        assert abs(c._roc_auc(scores[order],labels[order])[0]-a)<1e-12
    assert c._roc_auc(np.zeros(6),np.array([1,1,0,0,0,0],bool))[0]==.5

def main():
    verify_auc()
    seeds=range(42,52); primary={}; legacy=[]; corrected=[]; fs=[]
    noise={str(x):[] for x in (1,2,4,8)}
    floors={str(x):[] for x in (.10,.05,.01)}
    bs={str(x):[] for x in (.01,.03,.05,.07,.10)}
    rp={f'{a}:{m}':[] for a in (.01,.05,.10) for m in ('posthoc','inline','direct_read')}
    for seed in seeds:
        c.set_seed(seed); tx=c.gen_reconciliation_corpus()
        a=old.reconcile(tx); b=c.reconcile(tx)
        legacy.append(binary(tx,a)); corrected.append(binary(tx,b))
        sellers=c.gen_fraud_corpus(); tr=c.gen_traces()
        fr={f'{mode}_{threshold}':fraud(sellers,mode,threshold) for mode in ('cohort','universal') for threshold in (.60,.65)}
        fs.append(fr)
        if seed==42:
            primary=dict(legacy_e1=c.score_reconciliation(tx,a),corrected_e1=c.score_reconciliation(tx,b),
                         legacy_e1_raw=binary(tx,a),corrected_e1_raw=binary(tx,b),promo_audit=promo_audit(tx),
                         no_prior_suppression=binary(tx,c.reconcile(tx,version_aware=False)),
                         fraud=fr,legacy_boundary_check=c.detect_cascades(tr))
            np.savez(OUT/'primary_fraud_scores.npz',labels=c.fraud_scores(sellers)[1],
                     cohort=c.fraud_scores(sellers,True)[0],universal=c.fraud_scores(sellers,False)[0],
                     groups=np.array([s['cohort'] for s in sellers]))
        for floor in (.10,.05,.01):
            p=c.reconcile(tx,floor=floor); item=binary(tx,p)
            item['promo_recall']=sum(t['label']=='missing_promo_credit' and t['id'] in p for t in tx)/sum(t['label']=='missing_promo_credit' for t in tx)
            floors[str(floor)].append(item)
        for scale in (1,2,4,8):
            c.set_seed(seed); noisy=c.gen_reconciliation_corpus(noise_scale=scale)
            assert [t['label'] for t in noisy]==[t['label'] for t in tx]
            noise[str(scale)].append(binary(noisy,c.reconcile(noisy)))
        for amp in (.01,.03,.05,.07,.10): bs[str(amp)].append(boundary_sweep(seed,amp))
        cases=replay_cases(seed)
        for amp in (.01,.05,.10):
            for mode in ('posthoc','inline','direct_read'):
                rp[f'{amp}:{mode}'].append(replay(cases,amp,mode))
        print(f'Seed {seed} complete',flush=True)
    result=dict(environment=dict(python=platform.python_version(),numpy=np.__version__),
                seeds=list(seeds),primary=primary,legacy_reconciliation=mean_sd(legacy),
                corrected_reconciliation=mean_sd(corrected),
                fraud_summary={key:mean_sd([{k:x[key][k] for k in ('auc','recall','precision','clean_fpr','fpr_at_tpr_ge_085')} for x in fs]) for key in fs[0]},
                reconciliation_noise_sweep={k:mean_sd(v) for k,v in noise.items()},
                reconciliation_floor_sweep={k:mean_sd(v) for k,v in floors.items()},
                boundary_noise_sweep={k:mean_sd(v) for k,v in bs.items()},
                execution_replay={k:mean_sd(v) for k,v in rp.items()},
                per_seed=dict(legacy=legacy,corrected=corrected,fraud=fs,noise=noise,floors=floors,boundary=bs,replay=rp))
    (OUT/'results.revision.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Wrote results.revision.json; tie-aware AUC tests passed.')

if __name__=='__main__': main()
