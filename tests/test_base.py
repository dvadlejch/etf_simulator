import numpy as np
import pytest

from etf_simulator.base import (
    calculate_return_portfolio,
    calculate_return_portfolio_time_series,
)


@pytest.mark.parametrize(
    "daily_returns_matrix,weights,investment_amounts,buy_fee,expected_return",
    [
        [
            np.array([[0.1, 0.05, 0.0, -0.01, -0.1, 0.05, 0.3]]),
            np.array([1.0]),
            np.array([1000, 200, 0, 500, 100, 0, 100]),
            np.array([0]),
            0.247904605,
        ],
        [
            np.array(
                [
                    [0.1, -0.1, 0.2],
                    [-0.1, 0.0, 0.4],
                ]
            ),
            np.array([0.7, 0.3]),
            np.array([100, 200, 0]),
            np.array([0, 0]),
            0.232,
        ],
        [
            np.array(
                [
                    [0.1, -0.1, 0.2],
                    [-0.1, 0.0, 0.4],
                ]
            ),
            np.array([0.7, 0.3]),
            np.array([100, 200, 10000]),
            np.array([0, 0]),
            0.006757282,
        ],
        [
            np.array(
                [
                    [0.1, -0.1, 0.2],
                    [-0.1, 0.0, 0.4],
                ]
            ),
            np.array([0.7, 0.3]),
            np.array([100, 200, 10000]),
            np.array([10, 5]),
            0.001728155,
        ],
    ],
)
def test_calculate_return_portfolio(
        daily_returns_matrix, weights, investment_amounts, buy_fee, expected_return
):
    overall_return = calculate_return_portfolio(
        daily_returns_matrix,
        weights,
        investment_amounts,
        buy_fee,
    )

    np.testing.assert_almost_equal(overall_return, expected_return, decimal=7)


@pytest.mark.parametrize(
    "daily_returns_matrix,weights,investment_amounts,buy_fee,expected_time_series",
    [
        [
            np.array(
                [
                    [0.1, -0.1, 0.2],
                    [-0.1, 0.0, 0.4],
                ]
            ),
            np.array([0.7, 0.3]),
            np.array([100, 200, 0]),
            np.array([0, 0]),
            np.array([100, 293, 369.6]),
        ],
    ],
)
def test_calculate_return_portfolio_time_series(
        daily_returns_matrix, weights, investment_amounts, buy_fee, expected_time_series
):
    time_series = calculate_return_portfolio_time_series(
        daily_returns_matrix, weights, investment_amounts, buy_fee
    )

    np.testing.assert_allclose(time_series, expected_time_series)


def test_get_corelated_samples():
    assert False
