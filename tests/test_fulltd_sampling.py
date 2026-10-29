import math
import numpy as np
import pytest

from fulltd.protocol import make_manifest, validate_config
from fulltd.sampling import LocalPairLaw, rng_for, sources_for_round


def test_prespecified_matrix_and_budget():
    manifest = make_manifest()
    assert len(manifest) == 630
    assert sum(r["batch_size"] == 256 for r in manifest) == 450
    assert sum(r["batch_size"] == 128 for r in manifest) == 180
    assert {r["seed"] for r in manifest} == set(range(5))
    assert all(r["validation_epochs"] == list(range(10, 101, 10)) for r in manifest)
    assert all(r["selection_epochs"] == [80, 90, 100] for r in manifest)
    invalid = dict(manifest[0], epochs=99)
    with pytest.raises(ValueError):
        validate_config(invalid)


@pytest.mark.parametrize("name,lags", [("electricity", (1,24)), ("metr_la", (1,3)), ("fd001", (1,5))])
def test_pair_metadata_boundaries_no_gaps_no_other_entities(name, lags):
    origins = np.r_[np.arange(60), np.arange(80,120), np.arange(60)]
    entities = np.r_[np.zeros(100), np.ones(60)].astype(int)
    law = LocalPairLaw(entities, origins, lags)
    ids = np.arange(len(origins))
    partner, types, probabilities = law.enumerate(ids)
    assert np.allclose(probabilities.sum(axis=1), 1)
    for row, source in enumerate(ids):
        expected = []
        for slot, target in enumerate(partner[row]):
            if probabilities[row,slot]:
                assert target >= 0
                assert entities[source] == entities[target]
                assert abs(origins[source] - origins[target]) == lags[types[row,slot]]
                expected.append((target,types[row,slot]))
        available_types = {t for j,t in expected}
        for target, t in expected:
            slot = np.where(partner[row] == target)[0][0]
            assert probabilities[row,slot] == 1 / len(available_types) / sum(tp==t for j,tp in expected)
    # A singleton source is identified for common exclusion rather than wrapped.
    isolated = LocalPairLaw([0,0,1], [0,1,10], lags)
    assert isolated.nonisolated().tolist() == [0,1]


def test_source_partner_sequences_are_shared_and_independent_of_batch_grouping():
    n = 2501
    for dataset, lags in [("electricity",(1,24)),("metr_la",(1,3)),("fd001",(1,5))]:
        law = LocalPairLaw(np.zeros(n), np.arange(n), lags)
        a = sources_for_round(dataset,n,2,10)
        b = sources_for_round(dataset,n,2,10)
        assert np.array_equal(a,b)
        if dataset == "electricity":
            assert len(a) == math.ceil(n/24)
        else:
            assert sorted(a.tolist()) == list(range(n))
            assert len(a)%256 != 0  # final short batch retained
        pa,ta = law.sample(a,rng_for(dataset,2,10,22))
        rng = rng_for(dataset,2,10,22)
        parts = [law.sample(a[i:i+128],rng) for i in range(0,len(a),128)]
        assert np.array_equal(pa,np.concatenate([x[0] for x in parts]))
        assert np.array_equal(ta,np.concatenate([x[1] for x in parts]))
        assert not np.array_equal(a,sources_for_round(dataset,n,3,10))


def test_exact_expected_objective_and_empirical_unbiased_sampling():
    n=49
    law=LocalPairLaw(np.zeros(n),np.arange(n),(1,24))
    residual=np.sin(np.arange(n)*.23)+np.arange(n)/100
    theta=np.array([.4,-.3])
    p,t,w=law.enumerate(np.arange(n))
    safe=np.maximum(p,0)
    exact=np.sum(w*.5*(residual[:,None]-theta[t]*residual[safe])**2)/n
    rng=np.random.default_rng(876)
    source=rng.integers(0,n,size=300000)
    partner,kind=law.sample(source,rng)
    empirical=np.mean(.5*(residual[source]-theta[kind]*residual[partner])**2)
    assert empirical == pytest.approx(exact,abs=.002)
    assert np.max(np.abs(np.bincount(source,minlength=n)/len(source)-1/n)) < .001
    # Round partners are allowed outside the two source draws of that round.
    round_sources=sources_for_round("electricity",n,0,1)
    partners,_=law.sample(round_sources,rng_for("electricity",0,1,22))
    assert any(x not in round_sources for x in partners)
