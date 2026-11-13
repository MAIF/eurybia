"""
Unit tests of the report tabs built by generation.py
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import panel as pn
from panel.links import Callback

from eurybia.report import generation
from eurybia.report.generation import HIDE_INACTIVE_TABS_CSS, _get_project_information_panel, execute_report


class TestReportTabs(unittest.TestCase):
    """
    Tabs must be left to Panel: no inline ``display: none`` and no custom JS ``active`` callback,
    otherwise Panel >= 1.10 renders every tab except the first one blank.
    """

    def _build_report(self) -> pn.Tabs:
        """Run execute_report with lightweight tabs and return the pn.Tabs that would be saved."""
        saved = {}

        def fake_save(tabs, *args, **kwargs):
            saved["tabs"] = tabs

        def fake_panel(name):
            return lambda *args, **kwargs: pn.Column(pn.pane.Markdown(name), name=name)

        fake_dr = SimpleNamespace(smartdrift=SimpleNamespace(data_modeldrift=[1]), title_story="Report")
        with (
            patch.object(generation, "DriftReport", return_value=fake_dr),
            patch.object(generation, "_get_index_panel", fake_panel("Index")),
            patch.object(generation, "_get_project_information_panel", fake_panel("Project information")),
            patch.object(generation, "_get_consistency_analysis_panel", fake_panel("Consistency Analysis")),
            patch.object(generation, "_get_data_drift_panel", fake_panel("Data drift")),
            patch.object(generation, "_get_model_drift_panel", fake_panel("Model drift")),
            patch.object(generation, "_apply_html_head_metadata"),
            patch.object(pn.Tabs, "save", fake_save),
        ):
            execute_report(smartdrift=None, explainer=None, project_info_file="info.yml", output_file="report.html")
        return saved["tabs"]

    def test_tabs_are_not_hidden_inline(self):
        tabs = self._build_report()
        self.assertEqual(
            tabs._names, ["Index", "Project information", "Consistency Analysis", "Data drift", "Model drift"]
        )
        for tab in tabs.objects:
            self.assertNotEqual(tab.styles.get("display"), "none", tab.name)

    def test_no_custom_active_callback(self):
        tabs = self._build_report()
        callbacks = Callback.registry.get(tabs, [])
        self.assertFalse([cb for cb in callbacks if "active" in cb.code])

    def test_inactive_tabs_are_collapsed_by_stylesheet(self):
        tabs = self._build_report()
        self.assertIn(HIDE_INACTIVE_TABS_CSS, tabs.stylesheets)
        self.assertIn("visibility: hidden", HIDE_INACTIVE_TABS_CSS)
        self.assertIn("display: none", HIDE_INACTIVE_TABS_CSS)

    def test_project_information_panel_is_visible(self):
        panel = _get_project_information_panel(SimpleNamespace(metadata={"General": {"Owner": "eurybia"}}))
        self.assertEqual(panel.name, "Project information")
        self.assertNotEqual(panel.styles.get("display"), "none")
