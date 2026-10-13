import json
from unittest.mock import patch

import pytest

np=pytest.importorskip("numpy")
pd=pytest.importorskip("pandas")
from alpha_lab.equity_validation import validate_research, block_bootstrap_sharpe


def test_selection_is_persisted_before_any_holdout_call(tmp_path):
    panel=tmp_path/"panel";panel.mkdir()
    (panel/"panel.json").write_text(json.dumps({"limitations":[]}))
    run=tmp_path/"run";run.mkdir()
    specs=[{"name":"weaker","factors":{"roa":1}},{"name":"stronger","factors":{"roa":2}}]
    (run/"candidates.json").write_text(json.dumps(specs))
    output=tmp_path/"validation"
    holdout_seen=[]
    class Engine:
        def __init__(self,panel):pass
        def evaluate(self,spec,include_holdout=False,**kwargs):
            score=.2 if spec["name"]=="weaker" else .3
            result={"spec":spec,"max_abs_net_exposure":0.,"missing_exposure_events":0,
                    "missing_event_details":[],"train":{"sharpe_zero_rf":score,"realized_spy_beta":0.},
                    "validation":{"sharpe_zero_rf":score,"realized_spy_beta":0.}}
            if include_holdout:
                freeze=json.loads((output/"selection-frozen.json").read_text())
                assert freeze["selected"]["result"]["spec"]["name"]=="stronger"
                assert spec["name"]=="stronger"
                holdout_seen.append(spec["name"])
                result["holdout"]={"sharpe_zero_rf":-.5}
            daily=pd.DataFrame({"date":pd.to_datetime(['2023-01-03','2023-01-04','2023-01-05']),
                                "net":[.001,-.002,.003]})
            return result,daily
    empty=pd.DataFrame(columns=["ticker","date","action"])
    with patch("alpha_lab.equity_validation.MarketNeutralBacktest",Engine), \
         patch("alpha_lab.equity_validation.duckdb.read_parquet") as reader:
        reader.return_value.df.return_value=empty
        result=validate_research(panel,[run],output)
    assert result["selected"]["name"]=="stronger" and result["holdout_tuning_performed"] is False
    assert len(holdout_seen)==5  # one base artifact plus four predetermined scenarios
    with pytest.raises(ValueError,match="never overwrite"):
        validate_research(panel,[run],output)


def test_block_bootstrap_is_reproducible():
    x=np.random.default_rng(7).normal(.0001,.01,100)
    assert block_bootstrap_sharpe(x,trials=50)==block_bootstrap_sharpe(x,trials=50)
