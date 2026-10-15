"""Saved-forecast integration check; illustrative weights, no strategy selection.

Run from AlphaLab with its Python environment. All output stays local. Does not
fit models, contact APIs, access old holdouts, or execute trades.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from alpha_lab.research.contracts import digest
from alpha_lab.research.risk_tools import PortfolioRiskService, save_risk_operation


def verify(library, as_of, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    risk = PortfolioRiskService(library)
    # Explicit sample; this is not a recommendation or a model ranking.
    positions = [{'security': ticker, 'weight': w} for ticker,w in
                 zip(['NVDA','AAPL','MSFT','AMZN'], [.25,-.25,.25,-.25])]
    base = dict(as_of=as_of, positions=positions, model='market', horizon=1, identifier='ticker')
    security = dict(as_of=as_of,securities=['NVDA','AAPL','MSFT','AMZN'],model='market',identifier='ticker')
    queries = {
        'risk_catalog': {},
        'risk_universe': dict(as_of=as_of,model='market',offset=0,limit=10),
        'risk_portfolio': dict(base,benchmark=None),
        'risk_contributions': dict(base,top_n=None),
        'risk_portfolio_correlations': {k:v for k,v in dict(base,portfolios=[
            {'name':'sample','positions':positions},{'name':'reverse','positions':[
                dict(row,weight=-row['weight']) for row in positions]}]).items() if k!='positions'},
        'risk_security_matrices': dict(security,horizon=1),
        'risk_exposures': {k:v for k,v in dict(base,benchmark=None).items() if k!='horizon'},
        'risk_security_exposures': security,
        'risk_trade_impact': dict(base,trades=[{'security':'NVDA','weight':.01}]),
        'risk_hedge': dict(base,hedge_positions=[{'security':'NVDA','weight':1.}]),
        'risk_scale': dict(base,target_annualized_volatility=.1),
        'risk_compare_models': {k:v for k,v in dict(base,benchmark=None,models=None).items() if k!='model'},
    }
    responses = {}
    for tool, arguments in queries.items():
        saved = save_risk_operation(library,{'tool':tool,'arguments':arguments},output/tool)
        if saved['status']!='completed':
            raise RuntimeError('Risk integration query failed: '+tool)
        responses[tool] = json.loads((output/tool/'response.json').read_text())['result']
    comparison = responses['risk_compare_models']['results']
    assert len(comparison)==3 and all(row['metadata']['research_only'] for row in comparison)
    summary, contributions = responses['risk_portfolio'], responses['risk_contributions']
    assert np.isclose(sum(row['variance_contribution'] for row in contributions['contributions']),summary['variance'],atol=1e-12,rtol=1e-10)
    matrix = np.array(responses['risk_security_matrices']['covariance'])
    weights = np.array([row['weight'] for row in positions])
    assert np.isclose(weights@matrix@weights,summary['variance'],atol=1e-12,rtol=1e-10)
    trade = responses['risk_trade_impact']
    # Check the package's exact interaction identity independently of its report.
    delta = np.array([.01,0.,0.,0.])
    expected = 2*delta@matrix@weights+delta@matrix@delta
    actual = risk.invoke('risk_portfolio',dict(base,positions=[dict(row,weight=row['weight']+(.01 if i==0 else 0.))
                  for i,row in enumerate(positions)],benchmark=None))['result']['variance']-summary['variance']
    assert np.isclose(actual,expected,atol=1e-12,rtol=1e-10)
    # Same exact-date model/horizon check, not a backtest or model selection.
    for model in ('market','fundamental','direct'):
        for horizon in (1,5,21):
            saved=save_risk_operation(library,{'tool':'risk_portfolio','arguments':dict(base,model=model,horizon=horizon,benchmark=None)},
                                     output/f'{model}-h{horizon}')
            assert saved['status']=='completed'
    result={'local_tools_verified':12,'models':3,'horizons':[1,5,21], 'library_id':risk.binding['library_id'],
            'contributions_reconcile':True,'matrix_order_and_variance_verified':True,'trade_interaction_verified':True,
            'illustrative_request_sha256':digest(base),'llm_calls':0,'orders_submitted':0,'investment_ready':False}
    (output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--as-of',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(verify(args.library,args.as_of,args.output),indent=2))
