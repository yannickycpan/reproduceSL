"""Read-only export regression tests; no model checkpoint or TEST labels required."""
import json
import numpy as np
import pytest

from dscpuscripts.fd001_all_checkpoint_export import checkpoint_steps, _cached


def test_complete_snapshot_grid_and_validation_evidence():
    for batch, last_step in ((256, 5500), (128, 11000)):
        config = dict(run_id=f'test_b{batch}', batch_size=batch,
                      learning_curve_updates=500)
        curve = [dict(optimizer_steps=t) for t in range(0,last_step+1,500)]
        assert checkpoint_steps(config,13989,curve) == list(range(0,last_step+1,500))
        with pytest.raises(ValueError, match='Incomplete/unexpected'):
            checkpoint_steps(config,13989,curve[:-1])


def test_export_cache_validates_signature_and_arrays(tmp_path):
    a = tmp_path/'run.json'
    b = tmp_path/'run.npz'
    expected = dict(revision='test',run_id='run')
    steps = np.array([0,500,1000])
    a.write_text(json.dumps(dict(signature=expected, rows=[{}, {}, {}])))
    np.savez_compressed(b,steps=steps,engine_ids=np.array([1,2]),
                        target_rul_cycles=np.array([4.,5.]),
                        predictions=np.array([[5.,6.],[3.,8.],[4.,5.]]))
    assert _cached(a,b,expected,steps,2)['signature'] == expected
    assert _cached(a,b,expected,np.array([0,500]),2) is None
    with pytest.raises(ValueError,match='different identity'):
        _cached(a,b,dict(expected,run_id='other'),steps,2)
    b.unlink()
    assert _cached(a,b,expected,steps,2) is None
