import importlib
import sys
import types


def test_script_smoke_import_and_saves(monkeypatch):

    class FakeYelpDataset:
        features = {"text": "string", "label": "int64"}

        def __init__(self):
            self.saved_path = None

        def shuffle(self, seed):
            return self
        
        def select(self, indices):
            return self 
        
        def save_to_disk(self, path):
            self.saved_path = path

    fake_dataset = FakeYelpDataset()
    created = {"dataset": fake_dataset}

    fake_datasets = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        return fake_dataset

    fake_datasets.load_dataset = load_dataset

    # inject fake module BEFORE importing
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    module_name = "scripts.create_yelp_reviews_dataset"

    if module_name in sys.modules:
        del sys.modules[module_name]

    importlib.import_module(module_name)

    # assert created
    assert created["dataset"] is not None
    assert created["dataset"].saved_path == "datasets/yelp_reviews_25k"
