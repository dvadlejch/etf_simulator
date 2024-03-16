import numpy as np
import pytest

from etf_simulator.base import Etf
from etf_simulator.investment_strategy import (
    IndividualEtfInvestmentStrategy,
    get_investment_matrix,
    CombinedEtfInvestmentStrategy,
    WeightedEtfInvestmentStrategy,
)


@pytest.fixture(scope="module")
def etfs():
    etfs = [
        Etf(name="Vanguard", daily_returns=np.array([0.1, 0.1, -0.05, 0.05, 0.089])),
        Etf(
            name="ishares",
            daily_returns=np.array([0.01, 0.025, -0.04, 0.07, 0.09]),
        ),
    ]
    return etfs


@pytest.fixture(scope="module")
def investment_strategies(etfs):
    investment_strategies = [
        IndividualEtfInvestmentStrategy(
            etf=etfs[0],
            investment_amounts=np.array([50, 0, 0, 0, 0]),
            buy_fee=0,
        ),
        IndividualEtfInvestmentStrategy(
            etf=etfs[1],
            investment_amounts=np.array([10, 0, 0]),
            buy_fee=0,
        ),
    ]
    return investment_strategies


@pytest.fixture(scope="module")
def weighted_investment_strategy(etfs):
    weighted_strategy = WeightedEtfInvestmentStrategy(
        etfs=etfs,
        investment_amounts=np.array([100, 0, 50, 0]),
        weights=np.array([0.1, 0.9]),
        buy_fees=np.array([0, 0]),
    )

    expected_investment_matrix = np.array(
        [
            [10, 0, 5, 0],
            [90, 0, 45, 0],
        ]
    )
    return weighted_strategy, expected_investment_matrix


def test_get_investment_matrix(investment_strategies):
    expected_investment_matrix = np.array(
        [
            [50, 0, 0, 0, 0],
            [0, 0, 10, 0, 0],
        ]
    )

    investment_matrix = get_investment_matrix(investment_strategies)

    np.testing.assert_equal(investment_matrix, expected_investment_matrix)


def test_combined_etf_investment_strategy(etfs, investment_strategies):
    expected_investment_matrix = np.array(
        [
            [50, 0, 0, 0, 0],
            [0, 0, 10, 0, 0],
        ]
    )
    expected_etfs = etfs

    combined_strategy = CombinedEtfInvestmentStrategy(
        individual_investment_strategies=investment_strategies
    )

    np.testing.assert_equal(
        combined_strategy.investment_matrix, expected_investment_matrix
    )
    assert combined_strategy.etfs == expected_etfs


def test_weighted_etf_investment_strategy(etfs, weighted_investment_strategy):
    weighted_strategy, expected_investment_matrix = weighted_investment_strategy
    combined_strategy = weighted_strategy.combined_strategy

    assert combined_strategy.etfs == etfs
    np.testing.assert_equal(
        expected_investment_matrix, combined_strategy.investment_matrix
    )
