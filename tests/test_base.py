import numpy as np
import pytest
import scipy.stats

from etf_simulator.base import (
    calculate_return_portfolio,
    calculate_return_portfolio_time_series,
    get_correlated_samples,
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


@pytest.mark.parametrize(
    "dist, covariance_matrix",
    [
        (
            [scipy.stats.logistic, scipy.stats.norm(loc=0.5, scale=2)],
            np.array([[1, 0.25], [0.25, 1]]),
        ),
        (
            [scipy.stats.logistic(loc=2, scale=3), scipy.stats.logistic],
            np.array([[1, -0.7], [-0.7, 1]]),
        ),
        (
            [
                scipy.stats.maxwell(loc=2, scale=1),
                scipy.stats.logistic(loc=0.5, scale=2.5),
                scipy.stats.norm(loc=0.01, scale=0.05),
            ],
            np.array([[1, 0.3, -0.7], [0.3, 1, 0.15], [-0.7, 0.15, 1]]),
        ),
    ],
)
def test_get_correlated_samples(dist, covariance_matrix):
    np.random.seed(42)

    n_of_samples = 500

    expected_medians = np.array([distribution.median() for distribution in dist])
    expected_means = np.array([distribution.mean() for distribution in dist])
    expected_stds = np.array([distribution.std() for distribution in dist])

    # generate sample from the given distributions
    sample = get_correlated_samples(
        [distribution.ppf for distribution in dist],
        covariance_matrix,
        n_of_samples,
    )
    covariance_of_sampe = np.corrcoef(sample)

    # Check that covariance matrix is almost equal to the desired one.
    np.testing.assert_allclose(covariance_matrix, covariance_of_sampe, atol=1e-1)

    np.testing.assert_allclose(
        expected_medians, np.median(sample, axis=1), rtol=1e-4, atol=6e-1
    )
    np.testing.assert_allclose(
        expected_means, np.mean(sample, axis=1), rtol=1e-4, atol=6e-1
    )
    np.testing.assert_allclose(
        expected_stds, np.std(sample, axis=1), rtol=1e-4, atol=6e-1
    )
