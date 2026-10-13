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


def test_expired_pilot_starts_no_provider_or_local_experiment(tmp_path):
    from alpha_lab.equity_pilot import run_pilot
    with patch('alpha_lab.equity_pilot.bounded_provider',side_effect=AssertionError('API after cutoff')), \
         patch('alpha_lab.equity_pilot.EquityResearchTools',side_effect=AssertionError('Experiment after cutoff')), \
         patch('alpha_lab.equity_pilot.load_dotenv',side_effect=AssertionError('No key loading needed')):
        result=run_pilot(tmp_path/'panel',tmp_path/'run','gpt-5.6-sol','medium',
                         deadline_utc='2000-01-01T00:00:00Z')
    assert result['stopped_reason']=='authorization_deadline_reached'
    assert result['experiment_count']==0 and result['agent_finished'] is False
    assert json.loads((tmp_path/'run'/'run.json').read_text())==result
