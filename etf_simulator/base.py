"""Module containing an ETF class"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import numpy.typing as npt
import scipy.stats
import yfinance as yf


@dataclass
class Etf:
    """Class representing ETF data."""

    name: str
    ticker: str | yf.Ticker
    price_data: pd.DataFrame | None = None
    daily_returns: npt.NDArray | None = None
    daily_returns_histogram: scipy.stats.rv_histogram | None = None

    def __post_init__(self):
        if isinstance(self.ticker, str):
            self.ticker = yf.Ticker(self.ticker)

        if self.price_data is None:
            self.price_data = self._get_price_data()

        if self.daily_returns is None:
            self.daily_returns = self._get_daily_returns()

        if self.daily_returns_histogram is None:
            self.daily_returns_histogram = self._get_daily_returns_histogram()

    def _get_price_data(self) -> pd.DataFrame:
        return self.ticker.history(period="max")

    def _get_daily_returns(self) -> npt.NDArray:
        if self.price_data is None:
            raise ValueError(
                "Price dataframe is not set. Please get the price dataframe from which "
                "the daily returns can be calculated."
            )
        return self.price_data["Close"].pct_change().to_numpy[1:]

    def _get_daily_returns_histogram(self) -> scipy.stats.rv_histogram:
        if self.daily_returns is None:
            raise ValueError(
                "Daily returns are not set and the histogram can't be calculated."
            )

        return scipy.stats.rv_histogram(
            np.histogram(
                self.daily_returns,
                bins=len(self.daily_returns),
                density=True,
            ),
            density=True,
        )


@dataclass
class InvestmentStrategy:
    """contains parameters of investment strategy."""

    etf_weights: npt.NDArray
    investment_amounts: npt.NDArray
    buy_fee: npt.NDArray


class Portfolio:
    """Class representing a portfolio containing one or more ETFs."""

    def __init__(self, etfs: list[Etf], investment_strategy: InvestmentStrategy):
        self.etfs = etfs
        self.investment_strategy = investment_strategy

        self._daily_returns_covariance: npt.NDArray | None = None
        self._daily_returns_data_matrix: npt.NDArray | None = None

    @property
    def daily_returns_data_matrix(self) -> npt.NDArray:
        """TODO: figure out how to deal with time"""
        if self._daily_returns_data_matrix is None:
            max_data = np.min([len(etf.daily_returns) for etf in self.etfs])
            self._daily_returns_data_matrix = np.vstack(
                [etf.daily_returns[-max_data:] for etf in self.etfs]
            )
        return self._daily_returns_data_matrix

    @property
    def daily_returns_covariance(self) -> npt.NDArray:
        """TODO"""
        if self._daily_returns_covariance is None:
            self._daily_returns_covariance = np.cov(self.daily_returns_data_matrix)
        return self._daily_returns_covariance

    def calculate_return(self, time_period_days: int | None = None):
        """TODO"""
        # calculate returns for defined amount of time since the start defined in
        # investment strategy. It needs to be figured out how to deal with time also
        # w.r.t. daily returns matrix.
