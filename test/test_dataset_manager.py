from carcara.core.dataset_manager import DatasetManager
import pytest
from ase.io import read

@pytest.fixture
def setup_dataset():
    dataset = DatasetManager(filename="test/data/samples.xyz", seed=123)
    return dataset


def test_dataset_manager_initialization(setup_dataset, filename="test/data/samples.xyz", seed=123):
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
        dataset.split(ratios={"train": 0.7, "test": 0.4})


def test_split_invalid_ratios(setup_dataset):
    dataset = setup_dataset
    with pytest.raises(ValueError, match="Invalid ratio for 'train': -0.1. Ratios must be between 0 and 1."):
        dataset.split(ratios={"train": -0.1, "test": 1.1})
    with pytest.raises(ValueError, match="Invalid ratio for 'test': 1.1. Ratios must be between 0 and 1."):
        dataset.split(ratios={"train": 0.5, "test": 1.1})


def test_train_test_split(setup_dataset):
    dataset = setup_dataset
    dataset.train_test_split(train_ratio=0.75)
    assert len(dataset.split_data["train"]) == int(0.75 * dataset.total_configs)
    assert len(dataset.split_data["test"]) == dataset.total_configs - int(0.75 * dataset.total_configs)


def test_train_valid_test_split(setup_dataset):
    dataset = setup_dataset
    dataset.train_valid_test_split(train_ratio=0.6, valid_ratio=0.2)
    assert len(dataset.split_data["train"]) == int(0.6 * dataset.total_configs)
    assert len(dataset.split_data["valid"]) == int(0.2 * dataset.total_configs)
    assert len(dataset.split_data["test"]) == dataset.total_configs - int(0.6 * dataset.total_configs) - int(0.2 * dataset.total_configs)


def test_different_samples_with_different_seeds(setup_dataset):
    dataset1 = setup_dataset
    dataset2 = DatasetManager(filename="test/data/samples.xyz", seed=456)
    
    dataset1.train_test_split(train_ratio=0.8)
    dataset2.train_test_split(train_ratio=0.8)
    
    # Check the coordinates of the first atom, in the first configuration.
    assert not all(dataset1.split_data["train"][0].get_positions()[0] == dataset2.split_data["train"][0].get_positions()[0])
    # Check the coordinates of the first atom, in the first configuration of the test set.
    assert not all(dataset1.split_data["test"][0].get_positions()[0] == dataset2.split_data["test"][0].get_positions()[0])


def test_same_samples_with_same_seeds(setup_dataset):
    dataset1 = setup_dataset
    dataset2 = DatasetManager(filename="test/data/samples.xyz", seed=123)
    
    dataset1.train_test_split(train_ratio=0.8)
    dataset2.train_test_split(train_ratio=0.8)

    # Check the coordinates of the first atom, in the first configuration.
    assert all(dataset1.split_data["train"][0].get_positions()[0] == dataset2.split_data["train"][0].get_positions()[0])
    # Check the coordinates of the first atom, in the first configuration of the test set.
    assert all(dataset1.split_data["test"][0].get_positions()[0] == dataset2.split_data["test"][0].get_positions()[0])

def test_write_datasets(setup_dataset, tmp_path):
    dataset = setup_dataset
    dataset.train_test_split(train_ratio=0.8)
    dataset.write_datasets(directory=tmp_path)

    train_file = tmp_path / f"train_seed_{dataset.seed}.xyz"
    test_file = tmp_path / f"test_seed_{dataset.seed}.xyz"

    assert train_file.exists()
    assert test_file.exists()

def test_write_datasets_with_filenames(setup_dataset, tmp_path):
    dataset = setup_dataset
    dataset.train_test_split(train_ratio=0.8)
    custom_filenames = ["custom_train.xyz", "custom_test.xyz"]
    dataset.write_datasets(directory=tmp_path, filenames=custom_filenames)

    train_file = tmp_path / custom_filenames[0]
    test_file = tmp_path / custom_filenames[1]

    assert train_file.exists()
    assert test_file.exists()


def test_write_datasets_without_split(setup_dataset, tmp_path):
    dataset = setup_dataset
    with pytest.raises(ValueError, match="No split data found. Please run the split method before writing datasets."):
        dataset.write_datasets(directory=tmp_path)


def test_write_datasets_with_mismatched_filenames(setup_dataset, tmp_path):
    dataset = setup_dataset
    dataset.train_test_split(train_ratio=0.8)
    custom_filenames = ["custom_train.xyz"]  # Only one filename for two splits
    with pytest.raises(ValueError, match="Number of filenames \\(1\\) must match the number of splits \\(2\\)."):
        dataset.write_datasets(directory=tmp_path, filenames=custom_filenames)


def test_write_datasets_without_xyz_extension(setup_dataset, tmp_path):
    dataset = setup_dataset
    dataset.train_test_split(train_ratio=0.8)
    custom_filenames = ["custom_train", "custom_test"]  # Missing .xyz extension
    dataset.write_datasets(directory=tmp_path, filenames=custom_filenames)

    train_file = tmp_path / "custom_train.xyz"
    test_file = tmp_path / "custom_test.xyz"

    assert train_file.exists()
    assert test_file.exists()


def test_write_datasets_with_nonexistent_directory(setup_dataset, tmp_path):
    dataset = setup_dataset
    dataset.train_test_split(train_ratio=0.8)
    non_existent_dir = tmp_path / "non_existent_directory"
    dataset.write_datasets(directory=non_existent_dir)

    train_file = non_existent_dir / f"train_seed_{dataset.seed}.xyz"
    test_file = non_existent_dir / f"test_seed_{dataset.seed}.xyz"

    assert train_file.exists()
    assert test_file.exists()


def test_if_datasets_were_written_correctly(setup_dataset, tmp_path):
    dataset = setup_dataset
    dataset.train_test_split(train_ratio=0.8)
    dataset.write_datasets(directory=tmp_path)

    train_file = tmp_path / f"train_seed_{dataset.seed}.xyz"
    test_file = tmp_path / f"test_seed_{dataset.seed}.xyz"
    
    train_configs = read(train_file, index=":")
    test_configs = read(test_file, index=":")

    assert len(train_configs) == int(0.8 * dataset.total_configs)
    assert len(test_configs) == dataset.total_configs - int(0.8 * dataset.total_configs)
