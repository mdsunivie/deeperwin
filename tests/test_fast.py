import deeperwin
import deeperwin.main
import deeperwin.utilities.erwinConfiguration
import pytest
import tensorflow as tf
import numpy as np

@pytest.fixture(scope='session', params=['H2', 'LiH'])
def config_minimal(request):
    config = deeperwin.utilities.erwinConfiguration.DefaultConfig(quality='minimal')
    config.optimization.n_epochs = 3
    config.evaluation.n_epochs_max = 3
    config.evaluation.n_epochs_min = 1
    config.physical.set_by_molecule_name(request.param)
    return config

@pytest.fixture(scope='session')
def wavefunction_minimal(config_minimal):
    wf = deeperwin.main.WaveFunction(config_minimal)
    return wf

@pytest.fixture(scope='session')
def wf_traced(wavefunction_minimal):
    wf = wavefunction_minimal
    x = tf.random.normal([wf.config.integration.train.n_walkers, wf.n_electrons, 3])
    E = wavefunction_minimal.model.call(x)
    return wf

def test_CASSCF_energy(wavefunction_minimal):
    E_CASSCF = dict(H2=-1.1532864051173193, LiH=-8.004394134974454)
    E_ref = E_CASSCF[wavefunction_minimal.config.physical.name]
    assert pytest.approx(wavefunction_minimal.model.slater_model.total_energy, abs=1e-5) == E_ref

def test_optimize_and_eval(wf_traced):
    wf_traced.optimize()
    wf_traced.compile_evaluation()
    wf_traced.evaluate()

    assert not np.isnan(wf_traced.total_energy)
    assert np.isnan(wf_traced.forces).sum() == 0

if __name__ == '__main__':
    pytest.main()
