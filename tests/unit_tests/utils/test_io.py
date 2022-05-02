"""
Unit test io.py
"""
import unittest
from os import path
from pathlib import Path

from eurybia.utils.io import load_yml


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
