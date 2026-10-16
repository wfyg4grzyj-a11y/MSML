"""Causal close-observed controls: fixed next-close reductions, never alpha sizing."""
from __future__ import annotations

import numpy as np

from .contracts import ResearchError
from .portfolio import coverage, exposures


def reduction_targets(inputs, ids, covariance, nav, hard, controls):
    reason = coverage(inputs, ids, covariance, hard, nav, covariance_already_validated=True)
    if reason:
        raise ResearchError(reason)
    weights = np.array([p.held_shares*(p.raw_price_usd or 0.)/nav for p in inputs])
    before, violations = exposures(inputs, weights, covariance, nav, hard)
    exits = [i for i,p in enumerate(inputs) if p.held_shares and
             controls.exit_ineligible_holdings and (not p.eligible or p.score is None)]
    if not violations and not exits:
        return {'status':'no_trigger','observed_exposures':before,'violations':[]}
    target = weights.copy()
    target[exits] = 0.
    values, _ = exposures(inputs, target, covariance, nav, hard)
    scale = 1.
    for size, limit in ((values['gross'],controls.max_gross), (values['name_max'],controls.max_name),
                        (abs(values['net']),controls.net_tolerance), (abs(values['beta']),controls.beta_tolerance)):
        if size > limit:
            scale = min(scale,limit/size)
    # The generic hard policy can also constrain issuer and sector exposure.
    for field,limit in (('issuer_gross',hard.max_issuer_gross),('sector_net',hard.max_sector_net)):
        if limit is not None:
            size = max((abs(v) for v in values[field].values()),default=0.)
            if size > .8*limit:
                scale = min(scale,.8*limit/size)
    target *= scale
    turnover = float(np.abs(target-weights).sum())
    fraction = min(1., .9*hard.max_turnover/turnover) if turnover else 1.
    target = weights + fraction*(target-weights)
    # Only securities actually held at the decision close receive orders.
    # In particular an as-yet-unreceived distribution is never pre-ordered.
    shares = {p.security_id: p.held_shares*(target[i]/weights[i])
              for i,p in enumerate(inputs) if p.held_shares and weights[i]}
    planned, remaining = exposures(inputs, target, covariance, nav, hard)
    return {'status':'reduction_planned','observed_exposures':before,'violations':violations,
            'exit_security_ids':[inputs[i].security_id for i in exits],
            'uniform_scale':scale,'turnover_fraction':fraction,'target_shares':shares,
            'planned_exposures_at_decision_close':planned,'planned_residual_violations':remaining,
            'fill_guaranteed':False}
