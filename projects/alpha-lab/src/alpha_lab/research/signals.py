"""Causal labels, fold-local transforms and descriptive cross-sectional diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import ResearchError
from .features import KEYS


def make_labels(members, arrays, policy):
    """Decision at t close; enter t+lag close; earn returns AFTER that close."""
    dates = pd.DatetimeIndex(arrays['dates'])
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ResearchError('Labels require ordered unique official sessions')
    positions = {t: i for i, t in enumerate(arrays['tickers'].tolist())}
    out = members[KEYS].copy().reset_index(drop=True)
    rows = []
    for row in out.itertuples(index=False):
        decision = dates.get_indexer([row.date])[0]
        if decision < 0 or row.ticker not in positions:
            raise ResearchError('Decision/security is absent from the pinned session arrays')
        entry = decision + policy.execution_lag_sessions
        end = entry + policy.horizon_sessions
        available = end + policy.availability_lag_sessions
        entry_date = dates[entry] if entry < len(dates) else pd.NaT
        end_date = dates[end] if end < len(dates) else pd.NaT
        available_date = dates[available] if available < len(dates) else pd.NaT
        value, status = np.nan, 'right_censored'
        if end < len(dates):
            column = positions[row.ticker]
            path = arrays['returns'][entry+1:end+1, column].astype('float64')
            quotes = arrays['price_observed'][entry:end+1, column]
            benchmark = arrays['spy'][entry+1:end+1].astype('float64')
            if available >= len(dates):
                status = 'availability_censored'
            elif not quotes.all() or not np.isfinite(path).all():
                status = 'missing_asset_path'
            elif policy.target == 'simple_excess_spy' and not np.isfinite(benchmark).all():
                status = 'missing_benchmark_path'
            else:
                value = float(np.prod(1 + path) - 1)
                if policy.target == 'simple_excess_spy':
                    value -= float(np.prod(1 + benchmark) - 1)
                status = 'observed'
        rows.append((entry_date, end_date, available_date, value, status == 'observed', status))
    extra = pd.DataFrame(rows, columns=['label_entry', 'label_end', 'label_available', 'label',
                                        'label_observed', 'label_status'])
    return pd.concat([out, extra], axis=1)


def training_mask(frame, fold, sessions):
    """Purge the actual outcome/availability interval, with whole-session grouping."""
    sessions = pd.DatetimeIndex(sessions)
    first = sessions.searchsorted(pd.Timestamp(fold.validation_start))
    cutoff_index = first - fold.embargo_sessions
    if first >= len(sessions) or cutoff_index < 0:
        raise ResearchError('Fold has no supported validation cutoff/embargo')
    cutoff = sessions[cutoff_index]
    candidates = frame.date.between(fold.train_start, fold.train_end)
    before = frame.label_end.lt(cutoff) & frame.label_available.lt(cutoff)
    fit = candidates & frame.label_observed & before
    return fit, {'cutoff': cutoff.isoformat(), 'training_candidates': int(candidates.sum()),
                 'unobserved_training_labels': int((candidates & ~frame.label_observed).sum()),
                 'purged_overlap_or_availability': int((candidates & frame.label_observed & ~before).sum()),
                 'admitted_training_rows': int(fit.sum())}


def transform_feature(frame, column, fit, method):
    values = frame[column]
    if method == 'cross_sectional_rank':
        return values.groupby(frame.date).rank(method='average', pct=True) - .5, {
            'method': method, 'fit': 'none', 'population': 'all_feature_available_source_members_per_session'}
    train = values.loc[fit & values.notna()]
    mean, scale = (float(train.mean()), float(train.std(ddof=0))) if len(train) else (None, None)
    evidence = {'method': method, 'fit_rows': len(train), 'mean': mean, 'std_population': scale,
                'status': 'no_training_values' if not len(train) else ('constant' if scale == 0 else 'fitted')}
    transformed = values * np.nan if mean is None else ((values - mean) / scale if scale else values * 0.)
    return transformed, evidence


def score_fold(frame, spec, fold, sessions):
    selected = frame.date.between(fold.train_start, fold.train_end) | frame.date.between(fold.validation_start, fold.validation_end)
    work = frame.loc[selected].copy().reset_index(drop=True)
    fit, evidence = training_mask(work, fold, sessions)
    work['fold'] = fold.name
    work['role'] = np.where(work.date.between(fold.validation_start, fold.validation_end), 'validation', 'train')
    work['fit_admitted'] = fit
    recipe = spec.recipe.computation()
    names = list(recipe['factors'])
    fitted = {}
    for name in names:
        work['transformed_'+name], fitted[name] = transform_feature(work, 'raw_'+name, fit, recipe['transform'])
    components = work[['transformed_'+n for n in names]]
    work['feature_available'] = work[['raw_'+n for n in names]].notna().all(axis=1)
    work['transform_available'] = components.notna().all(axis=1)
    if recipe['operator'] == 'linear':
        score = sum(work['transformed_'+n] * coefficient for n, coefficient in recipe['factors'].items())
    else:
        quality = (work.transformed_roa + work.transformed_gross_profitability - work.transformed_leverage) / 3
        score = work.transformed_momentum_12_1 * (quality + .5)
    work['score'] = score.where(work.transform_available)
    work['score_available'] = np.isfinite(work.score)
    # Buckets are determined BEFORE looking at label availability. Equal scores
    # stay together, so ties may leave some buckets empty.
    rank = work.score.groupby(work.date).rank(method='average', pct=True)
    work['quantile'] = np.ceil(rank * spec.quantiles).clip(1, spec.quantiles).astype('Int64')
    evidence['features'] = fitted
    evidence['scored_validation_rows'] = int((work.score_available & work.role.eq('validation')).sum())
    return work, evidence


def rank_correlation(left, right, minimum):
    pair = pd.concat([left.rename('x'), right.rename('y')], axis=1).dropna()
    if len(pair) < minimum:
        return None, 'insufficient_pairs'
    ranks = pair.rank(method='average')
    if ranks.x.nunique() < 2 or ranks.y.nunique() < 2:
        return None, 'constant_values'
    return float(ranks.x.corr(ranks.y)), 'observed'


def diagnose(predictions, spec):
    daily, buckets = [], []
    for fold, whole in predictions.loc[predictions.role.eq('validation')].groupby('fold', sort=True):
        previous = None
        for date, section in whole.groupby('date', sort=True):
            scored = section.loc[section.score_available]
            paired = scored.loc[scored.label_observed]
            ic, reason = rank_correlation(paired.score, paired.label, spec.min_cross_section)
            score_map = scored.set_index('security_id').score
            top = set(scored.loc[scored['quantile'].eq(spec.quantiles), 'security_id'])
            stability, churn, overlap, entrants, exits = None, None, 0, 0, 0
            if previous is not None:
                prior_map, prior_top = previous
                shared = score_map.index.intersection(prior_map.index)
                overlap = len(shared)
                stability, _ = rank_correlation(score_map.reindex(shared), prior_map.reindex(shared), spec.min_cross_section)
                churn = 1 - len(top & prior_top) / len(prior_top) if prior_top else None
                entrants, exits = len(top - prior_top), len(prior_top - top)
            means = {}
            for q in range(1, spec.quantiles+1):
                group = scored.loc[scored['quantile'].eq(q)]
                observed = group.loc[group.label_observed, 'label']
                means[q] = float(observed.mean()) if len(observed) else None
                buckets.append({'fold': fold, 'date': date, 'quantile': q, 'scored': len(group),
                                'observed': len(observed), 'mean_label': means[q]})
            spread = (means[spec.quantiles] - means[1]) if all(means[q] is not None for q in (1, spec.quantiles)) else None
            daily.append({'fold': fold, 'date': date, 'members': len(section),
                          'feature_available': int(section.feature_available.sum()), 'scored': len(scored),
                          'labels_observed': int(section.label_observed.sum()), 'paired': len(paired),
                          'rank_ic': ic, 'ic_status': reason, 'top_minus_bottom_label': spread,
                          'score_rank_autocorrelation': stability, 'score_overlap': overlap,
                          'top_membership_churn': churn, 'top_entrants': entrants, 'top_exits': exits})
            previous = score_map, top
    daily, buckets = pd.DataFrame(daily), pd.DataFrame(buckets)
    if daily.empty:
        raise ResearchError('No validation decisions in the requested folds')
    summaries = []
    for fold, group in daily.groupby('fold'):
        values = group.rank_ic.dropna()
        summaries.append({'fold': fold, 'decision_sessions': len(group), 'ic_sessions': len(values),
                          'mean_rank_ic': float(values.mean()) if len(values) else None,
                          'median_rank_ic': float(values.median()) if len(values) else None,
                          'mean_paired_coverage': float((group.paired/group.members).mean()),
                          'missing_outcome_rows': int((group.members-group.labels_observed).sum())})
    return daily, buckets, summaries


def evaluate_signal(frame, spec, sessions):
    predictions, fits = [], {}
    for fold in spec.folds:
        out, fits[fold.name] = score_fold(frame, spec, fold, sessions)
        predictions.append(out)
    predictions = pd.concat(predictions, ignore_index=True)
    daily, quantiles, summary = diagnose(predictions, spec)
    return predictions, daily, quantiles, fits, summary
