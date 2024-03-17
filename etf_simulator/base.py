"""Module containing an ETF class"""
import typing

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def calculate_return_portfolio(
    daily_returns_matrix: npt.NDArray,
    weights: npt.NDArray,
    investment_amounts: npt.NDArray,
    buy_fee: npt.NDArray,
):
    j_dim, i_dim = daily_returns_matrix.shape
    assert len(weights) == j_dim
    assert len(investment_amounts) == i_dim

    p_ji = np.array(
        [
            np.prod((daily_returns_matrix + 1)[:, i + 1 :], axis=1)
            for i in range(daily_returns_matrix.shape[1])
        ]
    )
    invested_matrix = np.tensordot(weights, investment_amounts, axes=0)
    for i in range(len(buy_fee)):
        invested_matrix[i, :][invested_matrix[i, :] > 0] -= buy_fee[i]
    return np.sum(np.diag(invested_matrix @ p_ji)) / np.sum(investment_amounts) - 1


def calculate_return_portfolio_time_series(
    daily_returns_matrix: npt.NDArray,
    weights: npt.NDArray,
    investment_amounts: npt.NDArray,
    buy_fee: npt.NDArray,
):
    j_dim, i_dim = daily_returns_matrix.shape
    assert len(weights) == j_dim
    assert len(investment_amounts) == i_dim

    p_ji = np.array(
        [
            [
                np.prod((daily_returns_matrix + 1)[:, i + 1 : i_dim - j], axis=1)
                for i in range(daily_returns_matrix.shape[1])
            ]
            for j in range(daily_returns_matrix.shape[1])
        ]
    )

    invested_matrix = np.tensordot(weights, investment_amounts, axes=0)
    for i in range(len(buy_fee)):
        invested_matrix[i, :][invested_matrix[i, :] > 0] -= buy_fee[i]

    amount_daily_matrices = invested_matrix @ p_ji
    amount_daily = np.zeros(i_dim - 1)
    for i in range(i_dim - 1):
        amount_daily[i] = np.sum(np.diag(amount_daily_matrices[i, :, :]))

    return np.flip(np.append(amount_daily, investment_amounts[0]))


def get_correlated_samples(
    ppfs: typing.List[typing.Callable[[np.ndarray], np.ndarray]],
    # List of $num_dims percentile point functions
    cov_matrix: np.ndarray,  # covariance matrix, shape($num_dims, $num_dims)
    num_samples: int,  # number of random samples to draw
):
    num_variables = len(ppfs)
    rand = np.random.multivariate_normal(
        np.zeros(num_variables),
        cov_matrix,
        (num_samples,),
        check_valid="raise",
    )

    rand_uniform = norm.cdf(rand)

    rand_final_dist = np.zeros((num_variables, num_samples))
    for i, ppf in enumerate(ppfs):
        rand_final_dist[i, :] = ppf(rand_uniform[:, i])

    return rand_final_dist
