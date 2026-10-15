"""Independent hand-derived expectations; P1 does not implement a share ledger."""


def accounting_reference():
    return {
        'status': 'reference_expectations_only_new_ledger_deferred_to_P3',
        'legacy_accounting_verified': False,
        'cases': [
            {'case': 'long_purchase', 'initial_cash_usd': 1000, 'shares': 10, 'price_usd': 100,
             'expected_cash_usd': 0, 'expected_nav_usd': 1000},
            {'case': 'short_sale', 'initial_cash_usd': 1000, 'shares': -10, 'price_usd': 100,
             'expected_cash_usd': 2000, 'expected_nav_usd': 1000,
             'rule': 'Short proceeds are offset by the marked liability, not profit'},
            {'case': 'two_for_one_split', 'old_shares': 10, 'old_price_usd': 100,
             'expected_shares': 20, 'new_price_usd': 50, 'expected_asset_value_usd': 1000},
            {'case': 'long_cash_dividend', 'shares': 10, 'price_before_usd': 100, 'ex_price_usd': 99,
             'dividend_per_share_usd': 1, 'expected_receivable_usd': 10,
             'expected_asset_value_including_receivable_usd': 1000,
             'rule': 'Payment changes receivable to cash; do not also earn adjusted-price dividend'},
            {'case': 'short_dividend', 'cash_usd': 2000, 'shares': -10, 'ex_price_usd': 99,
             'expected_dividend_payable_usd': 10, 'expected_nav_usd': 1000},
            {'case': 'flat_price_round_trip', 'initial_cash_usd': 1000, 'each_trade_notional_usd': 1000,
             'one_way_cost_bps': 10, 'expected_total_cost_usd': 2, 'expected_nav_usd': 998},
            {'case': 'weekend_borrow_assumption', 'short_notional_usd': 1000, 'annual_rate': .03,
             'calendar_days': 3, 'day_count': 'ACT365F_hypothetical_not_broker_quote',
             'expected_charge_usd': 0.2465753424657534},
            {'case': 'unresolved_terminal', 'last_quote_usd': 100, 'new_quote': None,
             'expected_economic_proceeds': None,
             'rule': 'Last quote does not prove an executable liquidation; retain unresolved exposure'},
        ],
        'legacy_limitations': [
            'Monthly signals with daily target maintenance, not monthly buy-and-hold',
            'Fixed bps costs and 252-session borrow assumption; no cash or financing',
            'Zero marked missing return plus assumed stale-price exit is not verified liquidation',
            'Projection caps reduce gross; missing securities can force a zero repair portfolio',
            'Price-only signals inherit the legacy filing-data requirement',
            'Current corrected history and static eligibility are not attested historical feed replay',
        ],
        'pending_P3_cases': ['Partial/unfilled orders', 'Dividend entitlement boundaries',
                             'Cash/stock merger delivery', 'Stale quote recovery',
                             'Issuer/class concentration', 'Infeasible risk/borrow constraints'],
    }
