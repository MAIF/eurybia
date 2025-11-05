"""
Unit test io.py
"""
import unittest
from os import path
from pathlib import Path

from eurybia.utils.io import load_yml, load_pickle, save_pickle
import pytest
import tempfile

class Testio(unittest.TestCase):
    """
    Unit test io.py
    """

    def test_load_yml(self):
        """
        Unit test load_yml method
        """
        script_path = Path(path.abspath(__file__)).parent.parent.parent
        project_info_path = path.join(script_path, "data/project_info.yml")

        with self.assertRaises(ValueError):
            load_yml(path=0)

        load_yml(project_info_path)


    def test_save_pickle(self):
        with pytest.raises(ValueError):
            save_pickle(dict(), 1)

        with pytest.raises(ValueError):
            save_pickle(dict(), ".", "1")

    def test_load_pickle(self):
        with pytest.raises(ValueError):
            load_pickle(1)

    def test_save_and_load_pickle(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_file = f"{tmpdirname}/test_save_and_load_pickle.pkl"
            in_data = {"foo": "bar"}
            save_pickle(in_data, path=tmp_file)
            out_data = load_pickle(path=tmp_file)
            assert in_data == out_data
