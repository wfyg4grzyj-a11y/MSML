import json
import pytest
pd=pytest.importorskip('pandas')
from alpha_lab.equity_panel_audit import compare_panels
from alpha_lab.market_data.sharadar import file_hash
from alpha_lab.market_neutral import FEATURES


def setup_comparison(tmp_path,monkeypatch,*,future=False):
    panels=[]
    for label in ('original','corrected'):
        panel=tmp_path/label;panel.mkdir()
        (panel/'panel.json').write_text(json.dumps({'label':label}))
        (panel/'sources.json').write_text('{}')
        panels.append(panel)
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps({'selected':{'result':{'spec':{'name':'frozen','factors':{'roa':1}}}},
                                     'panel_manifest_sha256':file_hash(panels[0]/'panel.json')}))
    output=tmp_path/'comparison'
    calls=[]
    class Engine:
        def __init__(self,panel):
            self.features=pd.DataFrame({'ticker':['S'],'date':pd.to_datetime(['2014-01-31']),
                                        **{name:[.1] for name in FEATURES}})
            self.panel_integrity_verified=True
        def evaluate(self,spec,*,include_holdout):
            assert include_holdout is False
            assert json.loads((output/'comparison-frozen.json').read_text())['holdout_evaluated'] is False
            assert spec=={'name':'frozen','factors':{'roa':1}}
            calls.append(spec)
            return ({'train':{},'validation':{},'missing_price_observations':0},
                    pd.DataFrame({'date':pd.to_datetime(['2023-01-03' if future else '2020-01-03']),'net':[0.]}))
    monkeypatch.setattr('alpha_lab.equity_panel_audit.MarketNeutralBacktest',Engine)
    return panels,selection,output,calls


def test_comparison_commits_to_frozen_spec_without_holdout_reselection(tmp_path,monkeypatch):
    panels,selection,output,calls=setup_comparison(tmp_path,monkeypatch)
    expected=file_hash(selection)
    result=compare_panels(*panels,selection,output)
    assert len(calls)==2 and file_hash(selection)==expected
    assert result['new_strategy_trials']==0 and result['holdout_evaluated'] is False
    with pytest.raises(ValueError,match='immutable'):
        compare_panels(*panels,selection,output)


def test_comparison_rejects_accidental_protected_period_returns(tmp_path,monkeypatch):
    panels,selection,output,calls=setup_comparison(tmp_path,monkeypatch,future=True)
    with pytest.raises(ValueError,match='protected evaluation period'):
        compare_panels(*panels,selection,output)
    assert not (output/'summary.json').exists()


def test_raw_snapshot_changes_cannot_masquerade_as_methodology_correction(tmp_path,monkeypatch):
    panels,selection,output,calls=setup_comparison(tmp_path,monkeypatch)
    (panels[1]/'sources.json').write_text('{"stocks":"different"}')
    with pytest.raises(ValueError,match='identical raw-source'):
        compare_panels(*panels,selection,output)
    assert not calls
