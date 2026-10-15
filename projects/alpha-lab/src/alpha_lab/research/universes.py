"""Dated cap/liquidity membership, independent of signals and future outcomes."""
from __future__ import annotations

from typing import Annotated, Literal
import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from .contracts import ResearchError, StrictModel, digest


class UniversePolicy(StrictModel):
    methodology: Literal['smoothed_cap_liquidity_hysteresis_v1'] = 'smoothed_cap_liquidity_hysteresis_v1'
    target: Annotated[int, Field(ge=2, le=2000)] = 1100
    lower: Annotated[int, Field(ge=2, le=2000)] = 900
    upper: Annotated[int, Field(ge=2, le=2000)] = 1300
    entry_rank: Annotated[int, Field(ge=2, le=2000)] = 1000
    entry_cap: float = .5e9
    entry_dollar_volume: float = 5e6
    entry_price: float = 5.
    retention_cap: float = .4e9
    retention_dollar_volume: float = 2.5e6
    retention_price: float = 3.
    observations: int = 42
    listing_age_sessions: int = 63

    @model_validator(mode='after')
    def consistent(self):
        if not 2 <= self.lower <= self.target <= self.upper or self.entry_rank > self.upper:
            raise ValueError('Inconsistent membership bands')
        for field in ('cap','dollar_volume','price'):
            if not 0 < getattr(self,'retention_'+field) <= getattr(self,'entry_'+field):
                raise ValueError('Retention filters must be positive and no tighter than entry')
        if not 1 <= self.observations <= 63 or self.listing_age_sessions < 63:
            raise ValueError('Require independent 63-session windows and explicit IPO age')
        return self


def select_members(candidates, previous, policy):
    """One contemporaneous cross-section; retain an audit row for every candidate."""
    columns={'security_id','cap_median','dollar_volume_median','cap_count','quote_count','raw_price','listing_age'}
    if not columns <= set(candidates) or candidates.security_id.duplicated().any():
        raise ResearchError('Universe needs unique permanent identities and independent metric coverage')
    frame=candidates.copy()
    frame['security_id']=frame.security_id.astype(str)
    if frame.security_id.duplicated().any():raise ResearchError('Ambiguous universe identities')
    finite=np.isfinite(frame[['cap_median','dollar_volume_median','raw_price','listing_age']]).all(axis=1)
    covered=finite & frame.cap_count.ge(policy.observations) & frame.quote_count.ge(policy.observations)
    covered &= frame.listing_age.ge(policy.listing_age_sessions)
    def screen(prefix):
        return (covered & frame.cap_median.ge(getattr(policy,prefix+'_cap'))
                & frame.dollar_volume_median.ge(getattr(policy,prefix+'_dollar_volume'))
                & frame.raw_price.ge(getattr(policy,prefix+'_price')))
    frame['entry_eligible']=screen('entry');frame['retention_eligible']=screen('retention')
    # One common ranking population; entry and retention do not get different ranks.
    frame=frame.sort_values(['cap_median','security_id'],ascending=[False,True],na_position='last').reset_index(drop=True)
    frame['size_rank']=np.nan
    eligible=frame.retention_eligible
    frame.loc[eligible,'size_rank']=np.arange(1,int(eligible.sum())+1)
    previous=set(previous)
    if not previous:
        chosen=set(frame.loc[frame.entry_eligible,'security_id'].head(policy.target))
    else:
        keep=frame.security_id.isin(previous) & frame.retention_eligible & frame.size_rank.le(policy.upper)
        enter=frame.entry_eligible & frame.size_rank.le(policy.entry_rank)
        chosen=set(frame.loc[keep | enter,'security_id'])
        if len(chosen)<policy.lower:
            for sid in frame.loc[frame.entry_eligible,'security_id']:
                chosen.add(sid)
                if len(chosen)>=policy.target:break
        if len(chosen)>policy.upper:
            chosen=set(frame.loc[frame.security_id.isin(chosen),'security_id'].head(policy.upper))
    frame['member']=frame.security_id.isin(chosen)
    frame['previous_member']=frame.security_id.isin(previous)
    frame['reason']=np.select([frame.member & frame.previous_member,frame.member,~frame.retention_eligible,
                              frame.previous_member],['retained','entered','ineligible_or_missing','exited_rank_buffer'],default='outside_entry_rank')
    return frame,{'names':len(chosen),'shortfall':len(chosen)<policy.lower,
                  'entries':len(chosen-previous),'exits':len(previous-chosen),
                  'lost_previous_identity_ids':sorted(previous-set(frame.security_id)),
                  'policy_id':digest(policy.model_dump())}
