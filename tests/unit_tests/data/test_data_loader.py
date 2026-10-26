"""Unit tests for the example dataset loader."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from eurybia.data import data_loader

DATASETS = [
    ("house_prices", "house_prices_dataset.csv", "id,value\n1,10\n2,20\n", ["value"], [1, 2], True),
    ("titanic", "titanicdata.csv", "id,name\n1,Ada\n2,Grace\n", ["name"], [1, 2], False),
    ("us_car_accident", "US_Accidents_extract.csv", "id,name\n1,Ada\n2,Grace\n", ["id", "name"], [0, 1], False),
]


@pytest.mark.parametrize(
    "dataset,filename,csv_content,columns,index,has_labels",
    DATASETS,
)
def test_data_loading_reads_local_dataset(
    tmp_path, monkeypatch, dataset, filename, csv_content, columns, index, has_labels
):
    """Read each supported dataset without attempting a download."""
    monkeypatch.setattr(data_loader, "__file__", str(tmp_path / "data_loader.py"))
    (tmp_path / filename).write_text(csv_content)
    if has_labels:
        (tmp_path / "house_prices_labels.json").write_text(json.dumps({"value": "Value"}))
    download = Mock()
    monkeypatch.setattr(data_loader, "urlretrieve", download)

    result = data_loader.data_loading(dataset)

    if has_labels:
        dataframe, labels = result
        assert labels == {"value": "Value"}
    else:
        dataframe = result
    assert list(dataframe.columns) == columns
    assert list(dataframe.index) == index
    download.assert_not_called()


@pytest.mark.parametrize(
    "dataset,filename,csv_content,columns,index,has_labels",
    DATASETS,
)
def test_data_loading_downloads_missing_dataset(
    tmp_path, monkeypatch, dataset, filename, csv_content, columns, index, has_labels
):
    """Download missing files through a mocked URL retriever."""
    monkeypatch.setattr(data_loader, "__file__", str(tmp_path / "data_loader.py"))

    def download_file(url, filename):
        assert url.startswith("https://github.com/MAIF/eurybia/raw/master/eurybia/data/")
        destination_path = Path(filename)
        if destination_path.name == "house_prices_labels.json":
            destination_path.write_text(json.dumps({"value": "Value"}))
        else:
            destination_path.write_text(csv_content)

    download = Mock(side_effect=download_file)
    monkeypatch.setattr(data_loader, "urlretrieve", download)

    result = data_loader.data_loading(dataset)

    if has_labels:
        dataframe, labels = result
        assert labels == {"value": "Value"}
        assert download.call_count == 2
    else:
        dataframe = result
        assert download.call_count == 1
    assert list(dataframe.columns) == columns
    assert list(dataframe.index) == index
    assert all(call.args[0].endswith(Path(call.kwargs["filename"]).name) for call in download.call_args_list)


def test_data_loading_rejects_unknown_dataset():
    """Reject dataset names outside the documented set."""
    with pytest.raises(ValueError, match="Dataset not found"):
        data_loader.data_loading("unknown")
