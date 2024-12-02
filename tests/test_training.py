from deeperwin.configuration import Configuration, PhysicalConfig
from deeperwin.process_molecule import process_single_molecule


def test_training():
    """
    Test code for simple training on H2 molecule
    """

    # construct configuration - small system + small model + few epochs
    config = Configuration()
    config.computation.rng_seed = 1234
    config.physical = PhysicalConfig(name="H2")
    config.pre_training.n_epochs = 500
    config.optimization.n_epochs = 20
    config.optimization.mcmc.n_burn_in = 100
    config.evaluation.n_epochs = 20
    config.model.embedding.n_hidden_one_el = [16, 16, 16, 16]
    config.model.embedding.n_hidden_two_el = [4, 4, 4]
    config.model.embedding.n_hidden_el_ions = [4, 4, 4]
    config.model.embedding.n_iterations = 1
    config.logging.basic.log_level = "WARNING"
    config.save("./tests/test_training.yaml")

    # run the test training
    process_single_molecule("./tests/test_training.yaml")


if __name__ == "__main__":
    test_training()
