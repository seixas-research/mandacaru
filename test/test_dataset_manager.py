from carcara.dataset_manager import DatasetManager
import pytest

@pytest.fixture
def setup_dataset():
    dataset = DatasetManager(filename="test/samples.xyz", seed=123)
    return dataset


def test_dataset_manager_initialization(setup_dataset, filename="test/samples.xyz", seed=123):
    dataset = setup_dataset
    assert dataset.filename == filename
    assert dataset.seed == seed
    assert len(dataset.atoms) > 0
    assert dataset.total_configs == len(dataset.atoms)


def test_get_shuffled_atoms(setup_dataset):
    dataset = setup_dataset
    shuffled_atoms = dataset._get_shuffled_atoms()
    assert len(shuffled_atoms) == dataset.total_configs
    assert set(id(atom) for atom in shuffled_atoms) == set(id(atom) for atom in dataset.atoms)
    # Check that the order is different from the original
    assert any(id(shuffled_atoms[i]) != id(dataset.atoms[i]) for i in range(len(dataset.atoms)))


def test_split_ratios_sum_to_one(setup_dataset):
    dataset = setup_dataset
    with pytest.raises(ValueError, match="The ratios must sum to 1.0"):
        dataset.split(ratios={"train": 0.7, "test": 0.4}, verbose=False)


def test_split_invalid_ratios(setup_dataset):
    dataset = setup_dataset
    with pytest.raises(ValueError, match="Invalid ratio for 'train': -0.1. Ratios must be between 0 and 1."):
        dataset.split(ratios={"train": -0.1, "test": 1.1}, verbose=False)
    with pytest.raises(ValueError, match="Invalid ratio for 'test': 1.1. Ratios must be between 0 and 1."):
        dataset.split(ratios={"train": 0.5, "test": 1.1}, verbose=False)


def test_split_valid_ratios(setup_dataset):
    dataset = setup_dataset
    splits = dataset.split(ratios={"train": 0.7, "test": 0.3}, verbose=False)
    assert len(splits) == 2
    assert len(splits[0]) == int(0.7 * dataset.total_configs)
    assert len(splits[1]) == dataset.total_configs - int(0.7 * dataset.total_configs)


def test_train_test_split(setup_dataset):
    dataset = setup_dataset
    train, test = dataset.train_test_split(train_ratio=0.75, verbose=False)
    assert len(train) == int(0.75 * dataset.total_configs)
    assert len(test) == dataset.total_configs - int(0.75 * dataset.total_configs)


def test_train_validation_test_split(setup_dataset):
    dataset = setup_dataset
    train, valid, test = dataset.train_validation_test_split(train_ratio=0.6, valid_ratio=0.2, verbose=False)
    assert len(train) == int(0.6 * dataset.total_configs)
    assert len(valid) == int(0.2 * dataset.total_configs)
    assert len(test) == dataset.total_configs - int(0.6 * dataset.total_configs) - int(0.2 * dataset.total_configs)


def test_different_samples_with_different_seeds(setup_dataset):
    dataset1 = setup_dataset
    dataset2 = DatasetManager(filename="test/samples.xyz", seed=456)
    
    train1, test1 = dataset1.train_test_split(train_ratio=0.8, verbose=False)
    train2, test2 = dataset2.train_test_split(train_ratio=0.8, verbose=False)
    
    # Check the coordinates of the first atom, in the first configuration.
    assert not all(train1[0].get_positions()[0] == train2[0].get_positions()[0])
    # Check the coordinates of the first atom, in the first configuration of the test set.
    assert not all(test1[0].get_positions()[0] == test2[0].get_positions()[0])


def test_same_samples_with_same_seeds(setup_dataset):
    dataset1 = setup_dataset
    dataset2 = DatasetManager(filename="test/samples.xyz", seed=123)
    
    train1, test1 = dataset1.train_test_split(train_ratio=0.8, verbose=False)
    train2, test2 = dataset2.train_test_split(train_ratio=0.8, verbose=False)

    # Check the coordinates of the first atom, in the first configuration.
    assert all(train1[0].get_positions()[0] == train2[0].get_positions()[0])
    # Check the coordinates of the first atom, in the first configuration of the test set.
    assert all(test1[0].get_positions()[0] == test2[0].get_positions()[0])



    


