import importlib
import sys
import types

from regex import F


def test_script_smoke_import_and_saves(monkeypatch):

    class FakeHFDataset:
        def __init__(self):
            self.saved_path = None

        def save_to_disk(self, path):
            self.saved_path = path

    created = {"dataset": None}

    fake_datasets = types.ModuleType("datasets")

    class Dataset:
        @staticmethod
        def from_pandas(df):
            created["dataset"] = FakeHFDataset()
            return created["dataset"]
        
        def from_list(items):
            created["dataset"] = FakeHFDataset()
            return created["dataset"]

    fake_datasets.Dataset = Dataset
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    fake_langdetect = types.ModuleType("langdetect")
    fake_langdetect.detect = lambda text: "en"

    class _DetectorFactory:
        seed = 0

    fake_langdetect.DetectorFactory = _DetectorFactory
    fake_langdetect.LangDetectException = Exception
    monkeypatch.setitem(sys.modules, "langdetect", fake_langdetect)

    monkeypatch.setattr("os.listdir", lambda _: ["fake.csv"])

    import pandas as pd
    monkeypatch.setattr(
        pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame([{"review": "Great game", "recommendationid": 1}]),
    )

    module_name = "scripts.create_steam_reviews_dataset"

    if module_name in sys.modules:
        del sys.modules[module_name]

    importlib.import_module(module_name)

    # assert created
    assert created["dataset"] is not None
    assert created["dataset"].saved_path == "datasets/steam_reviews_25k"
