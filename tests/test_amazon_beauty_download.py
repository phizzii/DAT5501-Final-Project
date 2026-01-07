import importlib
import sys
import types


def test_script_smoke_import_and_saves(monkeypatch):

    class FakeStream:
        def __iter__(self):
            # first = next(iter(dataset_stream)) needs at least one dict
            yield {"asin": "B000TEST", "overall": 5}

        def shuffle(self, buffer_size, seed):
            return self

    class FakeDataset:
        def __init__(self, items):
            self.items = items
            self.saved_path = None

        def save_to_disk(self, path):
            self.saved_path = path

    fake_stream = FakeStream()
    created = {"dataset": None}

    fake_datasets = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        return fake_stream

    class Dataset:
        @staticmethod
        def from_list(items):
            created["dataset"] = FakeDataset(items)
            return created["dataset"]

    fake_datasets.load_dataset = load_dataset
    fake_datasets.Dataset = Dataset

    # inject fake module BEFORE importing
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    module_name = "scripts/create_amazon_beautypersonalcare_reviews_dataset.py"

    if module_name in sys.modules:
        del sys.modules[module_name]

    importlib.import_module(module_name)

    # assert created
    assert created["dataset"] is not None
    assert created["dataset"].saved_path == "datasets/amazon_beauty_reviews_10k"
