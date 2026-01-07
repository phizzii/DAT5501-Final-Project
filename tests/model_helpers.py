import types
import importlib.util
import pandas as pd
from pandas.core.generic import NDFrame
import sys
import pathlib


class FakeArrowDataset:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


def install_fake_datasets_load_from_disk(monkeypatch, df: pd.DataFrame):
    fake_datasets = types.ModuleType("datasets")

    def load_from_disk(path):
        return FakeArrowDataset(df)

    fake_datasets.load_from_disk = load_from_disk
    monkeypatch.setitem(__import__("sys").modules, "datasets", fake_datasets)


def install_fake_textblob(monkeypatch, polarity=0.0, subjectivity=0.0):
    class _Sentence:
        def __init__(self, text):
            self.words = text.split()

    class _Sentiment:
        def __init__(self, p, s):
            self.polarity = p
            self.subjectivity = s

    class FakeTextBlob:
        def __init__(self, text):
            self._text = text
            self.sentiment = _Sentiment(polarity, subjectivity)
            self.sentences = [_Sentence(text)] if text.strip() else []

    fake_textblob_module = types.ModuleType("textblob")
    fake_textblob_module.TextBlob = FakeTextBlob
    monkeypatch.setitem(__import__("sys").modules, "textblob", fake_textblob_module)


def capture_to_csv_calls(monkeypatch):
    paths = []

    def _fake_to_csv(self, path, *args, **kwargs):
        paths.append(path)

    monkeypatch.setattr(NDFrame, "to_csv", _fake_to_csv, raising=True)
    return paths

def run_script_by_path(script_path: str):
    path = pathlib.Path(script_path)
    assert path.exists(), f"Script not found: {path.resolve()}"

    spec = importlib.util.spec_from_file_location("__main__", path)
    module = importlib.util.module_from_spec(spec)

    module.__dict__["__name__"] = "__main__"

    sys.modules["__main__"] = module
    spec.loader.exec_module(module)
    return module

import types

def install_fake_matplotlib_pyplot(monkeypatch, saved_paths: list):
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")

    def figure(*args, **kwargs):
        return None

    def barh(*args, **kwargs):
        return None

    def violinplot(*args, **kwargs):
        return None

    def xticks(*args, **kwargs):
        return None

    def title(*args, **kwargs):
        return None

    def xlabel(*args, **kwargs):
        return None

    def ylabel(*args, **kwargs):
        return None

    def tight_layout(*args, **kwargs):
        return None

    def close(*args, **kwargs):
        return None

    def savefig(path, *args, **kwargs):
        saved_paths.append(path)

    fake_pyplot.figure = figure
    fake_pyplot.barh = barh
    fake_pyplot.violinplot = violinplot
    fake_pyplot.xticks = xticks
    fake_pyplot.title = title
    fake_pyplot.xlabel = xlabel
    fake_pyplot.ylabel = ylabel
    fake_pyplot.tight_layout = tight_layout
    fake_pyplot.close = close
    fake_pyplot.savefig = savefig

    fake_matplotlib.pyplot = fake_pyplot

    import sys
    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot)
