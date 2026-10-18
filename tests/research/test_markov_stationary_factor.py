import copy
import json
from fractions import Fraction as Q

import numpy as np
import pytest

from research import channel_noise_mechanism as cm
from research import markov_stationary_factor as factor
from research import stationary_noise_artifact as dense
from research.execution import UnidentifiableFitError
from transfer.noise_model import metadata_digest
from transfer.temporal_noise import TemporalNoise


def mm(a,b):
    return [[sum((x*y for x,y in zip(row,col,strict=True)),Q()) for col in zip(*b,strict=True)] for row in a]

def trans(a):
    return list(map(list,zip(*a,strict=True)))

def solve(a,b):
    n=len(b)
    rows=[list(row)+[value] for row,value in zip(a,b,strict=True)]
    for i in range(n):
        pivot=next(j for j in range(i,n) if rows[j][i])
        rows[i],rows[pivot]=rows[pivot],rows[i]
        divisor=rows[i][i]
        rows[i]=[v/divisor for v in rows[i]]
        for j in range(n):
            if j!=i:
                multiplier=rows[j][i]
                rows[j]=[v-multiplier*w for v,w in zip(rows[j],rows[i],strict=True)]
    return [row[-1] for row in rows]

def inverse(a):
    n=len(a)
    cols=[solve(a,[Q(i==j) for i in range(n)]) for j in range(n)]
    return trans(cols)

def independent_covariance(reflections,sigma,prefix):
    c,p=np.asarray(reflections).shape
    q=[[Q.from_float(float(v)) for v in row] for row in sigma]
    if not p:
        return [[q[i%c][j%c] if i//c==j//c else Q() for j in range(c*prefix)] for i in range(c*prefix)]
    coefficients=[]
    for k in reflections:
        a=[-Q.from_float(float(k[0]))]
        if p==2:
            k2=Q.from_float(float(k[1]))
            a=[a[0]*(1+k2),-k2]
        elif p>2:
            polynomial=[]
            for value in k:
                reflection=Q.from_float(float(value))
                polynomial=[polynomial[i]+reflection*polynomial[-i-1] for i in range(len(polynomial))]+[reflection]
            a=[-value for value in polynomial]
        coefficients.append(a)
    n=p*c
    f=[[Q() for _ in range(n)] for _ in range(n)]
    for i in range(c):
        for lag in range(p):
            f[i][lag*c+i]=coefficients[i][lag]
    for i in range(c,n):
        f[i][i-c]=Q(1)
    matrix=[[Q(int(i==k and j==column))-f[i][k]*f[j][column]
             for k in range(n) for column in range(n)] for i in range(n) for j in range(n)]
    rhs=[q[i][j] if i<c and j<c else Q() for i in range(n) for j in range(n)]
    state=solve(matrix,rhs)
    state=[state[i*n:(i+1)*n] for i in range(n)]
    powers=[[[Q(i==j) for j in range(n)] for i in range(n)]]
    for _ in range(1,prefix):
        powers.append(mm(f,powers[-1]))
    lags=[mm(power,state) for power in powers]
    return [[lags[i//c-j//c][i%c][j%c] if i>=j else lags[j//c-i//c][j%c][i%c]
             for j in range(c*prefix)] for i in range(c*prefix)]


@pytest.mark.parametrize('reflections',[[[],[]],[[-.625],[-.375]],[[-.625],[-.625]],
    [[-.5,.125],[-.25,-.125]],[[-.5,.125],[-.5,.125]],[[0.],[0.]]])
@pytest.mark.parametrize('scale',[1.,1e-320,1e200])
def test_exact_certificate_dominates_independent_dense_inverse(reflections,scale):
    sigma=scale*np.array([[1.,.2],[.2,.75]])
    root,state=factor.compile_markov_stationary_root(reflections,sigma,4)
    inv=inverse([[Q.from_float(float(v)) for v in row] for row in root])
    law=independent_covariance(reflections,sigma,4)
    pulled=mm(mm(inv,law),trans(inv))
    squared=sum(((v-Q(i==j))**2 for i,row in enumerate(pulled) for j,v in enumerate(row)),Q())
    bound=Q.from_float(state['certificate']['joint_law_pullback_F_bound'])
    assert squared<=bound*bound
    assert bound<Q.from_float(1e-8*np.sqrt(8))
    original=dense.stationary_root(reflections,sigma,4)
    scale_root=np.max(np.abs(root))
    np.testing.assert_allclose(root/scale_root,original/scale_root,rtol=2e-14,atol=1e-94)
    assert factor.admit_markov_stationary_root(reflections,sigma,4,root,state)==state['certificate']


def test_weak_direction_not_removed_and_changed_direction_rejected():
    sigma=np.diag([1.,2.**-700])
    root,state=factor.compile_markov_stationary_root([[-.625],[-.375]],sigma,4)
    assert np.all(np.diag(root)>0)
    changed=root.copy()
    changed[1,1]*=1.001
    with pytest.raises(UnidentifiableFitError,match='original law policy'):
        factor.admit_saved_markov_stationary_root([[-.625],[-.375]],sigma,4,changed)
    assert factor.admit_saved_markov_stationary_root([[-.625],[-.375]],sigma,4,root)


def test_covariance_and_saved_state_counterexamples():
    for sigma in (np.diag([1.,0.]),[[1.,2.],[2.,1.]]):
        with pytest.raises(UnidentifiableFitError):
            factor.compile_markov_stationary_root([[-.5],[-.3]],sigma,4)
    root,state=factor.compile_markov_stationary_root([[-.5],[-.3]],[[1.,.2],[.2,.7]],4)
    changed=copy.deepcopy(state)
    changed['certificate']['joint_law_pullback_F_bound']=0.
    with pytest.raises(ValueError,match='certificate'):
        factor.admit_markov_stationary_root([[-.5],[-.3]],[[1.,.2],[.2,.7]],4,root,changed)
    with pytest.raises(ValueError,match='stationary law'):
        factor.admit_markov_stationary_root([[-.5],[-.4]],[[1.,.2],[.2,.7]],4,root,state)


def make_noise(family='private'):
    temporal=tuple(TemporalNoise.from_reflections([value],innovation_variance=7.,reference_id='independent-shape')
                   for value in (-.625,-.375 if family=='private' else -.625))
    geometry=cm.NoiseGeometry(('a','b'),1000.,'V','nose','native',2)
    return cm.FittedChannelNoise.from_markov_stationary_factors(temporal,np.array([[1.,.2],[.2,.7]])*1e-12,
        geometry=geometry,conditioning_samples=4,family=family,
        provenance={'fit_scope':'analytic_reference','reference_id':'factor-control','records':[]})

@pytest.mark.parametrize('family',['private','shared'])
def test_new_codec_cold_no_any_cholesky_compiler(tmp_path,monkeypatch,family):
    noise=make_noise(family)
    state=json.loads(json.dumps(noise.record()))
    assert state['schema']=='n2p3_channel_noise_mechanism/5'
    state['prefix_compiler']['source_sha256']='a'*64
    state['prefix_compiler']['factor_law_source_sha256']='b'*64
    path=tmp_path/'new.json'
    path.write_text(json.dumps({'state':state,'state_sha256':metadata_digest(state)}),encoding='utf8')
    samples=np.arange(24).reshape(12,2)*1e-7
    white=noise.whiten(samples)
    def forbidden(*args,**kwargs):
        raise AssertionError('cold /5 entered a root compiler')
    for owner,name in ((factor,'_cholesky'),(dense,'_cholesky'),(cm,'_prefix_root'),
                       (factor,'compile_markov_stationary_root')):
        monkeypatch.setattr(owner,name,forbidden)
    restored=cm.FittedChannelNoise.load(path)
    assert json.loads(json.dumps(restored.record()))==state
    np.testing.assert_array_equal(restored.compiled_prefix_root,noise.compiled_prefix_root)
    np.testing.assert_array_equal(restored.whiten(samples),white)
    assert restored.whitening_logabsdet(12)==noise.whitening_logabsdet(12)


def test_old_private_codec_preserves_root_and_record_without_dense_admission(tmp_path,monkeypatch):
    noise=make_noise()
    original=cm.FittedChannelNoise(noise.temporal,noise.innovation_covariance,noise.geometry,4,'private',noise.provenance,
                                   noise.compiled_prefix_root)
    path=tmp_path/'old.json'
    original.save(path)
    record=original.record()
    def forbidden(*args,**kwargs):
        raise AssertionError('old /3 private invoked full-prefix admission')
    monkeypatch.setattr(cm,'stationary_root',forbidden)
    restored=cm.FittedChannelNoise.load(path)
    assert json.loads(json.dumps(restored.record()))==json.loads(json.dumps(record))
    np.testing.assert_array_equal(restored.compiled_prefix_root,original.compiled_prefix_root)
    assert restored.saved_root_admission['state_role'].startswith('consumer_witness_only')


def test_general_order_three_and_near_stability_boundary():
    for reflections,prefix in (([[-.375,.125,-.0625],[-.25,-.125,.0625]],5),
                               ([[-1.+2.**-35],[-.5]],4)):
        sigma=np.array([[1.,-.2],[-.2,.75]])
        root,state=factor.compile_markov_stationary_root(reflections,sigma,prefix)
        inv=inverse([[Q.from_float(float(v)) for v in row] for row in root])
        pulled=mm(mm(inv,independent_covariance(reflections,sigma,prefix)),trans(inv))
        squared=sum(((value-Q(i==j))**2 for i,row in enumerate(pulled) for j,value in enumerate(row)),Q())
        bound=Q.from_float(state['certificate']['joint_law_pullback_F_bound'])
        assert squared<=bound*bound


def test_prefix_without_innovation_tail_keeps_original_exact_entry(monkeypatch):
    calls=[]
    original=cm._prefix_root
    def track(*args,**kwargs):
        calls.append(True)
        return original(*args,**kwargs)
    monkeypatch.setattr(cm,'_prefix_root',track)
    temporal=TemporalNoise.from_reflections([-.5,.125],innovation_variance=7.,reference_id='no-tail')
    geometry=cm.NoiseGeometry(('a','b'),1000.,'V','nose','native',2)
    noise=cm.FittedChannelNoise.from_markov_stationary_factors((temporal,temporal),np.array([[1.,.2],[.2,.7]]),
        geometry=geometry,conditioning_samples=2,family='shared',
        provenance={'fit_scope':'analytic_reference','reference_id':'no-tail','records':[]})
    assert calls==[True]
    assert noise.record()['schema']=='n2p3_channel_noise_mechanism/3'


def test_certificate_does_not_call_platform_float_linear_algebra(monkeypatch):
    reflections=[[-.625],[-.375]]
    sigma=np.array([[1.,1.-2.**-40],[1.-2.**-40,1.]])
    root,state=factor.compile_markov_stationary_root(reflections,sigma,4)
    def forbidden(*args,**kwargs):
        raise AssertionError('certificate entered platform floating linear algebra')
    for name in ('inv','solve','cholesky','svd','eigvalsh','norm'):
        monkeypatch.setattr(np.linalg,name,forbidden)
    assert factor.admit_markov_stationary_root(reflections,sigma,4,root,state)==state['certificate']


def test_fit_opt_in_changes_only_prefix_root_producer_and_codec():
    geometry=cm.NoiseGeometry(('a','b'),1000.,'V','nose','native',2)
    samples=np.random.default_rng(13).normal(size=(256,2))*1e-5
    residual=cm.ResidualRecord.from_arrays(samples=samples,design=np.zeros((256,1)),kernel_mean=np.zeros((1,2)),
        nuisance=np.zeros_like(samples),record_id='synthetic-C',participant_key='software-only',
        recording_id='synthetic-record',sample_start=0,geometry=geometry,role='target_calibration',
        source_model_sha256='a'*64,source_input_sha256='b'*64)
    original=cm.FittedChannelNoise.fit((residual,),family='private',order=1,conditioning_samples=4)
    changed=cm.FittedChannelNoise.fit((residual,),family='private',order=1,conditioning_samples=4,
                                     prefix_root_program='markov_innovations')
    assert original.record()['schema']=='n2p3_channel_noise_mechanism/3'
    assert changed.record()['schema']=='n2p3_channel_noise_mechanism/5'
    assert original.provenance==changed.provenance
    assert original.temporal[0].record()==changed.temporal[0].record()
    assert original.temporal[1].record()==changed.temporal[1].record()
    np.testing.assert_array_equal(original.innovation_covariance,changed.innovation_covariance)
