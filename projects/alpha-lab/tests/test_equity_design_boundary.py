import json
from unittest.mock import patch

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")
from alpha_lab.equity_pilot import PublicStrategyDesigner, TOOLS


def test_design_tools_have_no_data_access_or_results(tmp_path):
    designer=PublicStrategyDesigner(tmp_path)
    with patch("alpha_lab.equity_pilot.MarketNeutralBacktest",side_effect=AssertionError("Data access forbidden")), \
         patch("alpha_lab.equity_pilot.prepare_panel",side_effect=AssertionError("Data access forbidden")):
        rules=json.loads(designer("research_rules",{})["output"])
        assert set(rules)=={"features","definitions","construction","predetermined_costs",
                            "timing_rule","design_limit","privacy"}
        result=designer("register_strategy",{"name":"quality", "factors":[{"feature":"roa","coefficient":1}]})
        assert set(json.loads(result["output"]))=={"registered","testing"}
        assert "sharpe" not in result["output"] and "ticker" not in result["output"]
        assert not hasattr(designer,"engine") and not hasattr(designer,"manifest")
    assert {tool["name"] for tool in TOOLS}=={"research_rules","register_strategy","report_to_user"}
