from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from etf_simulator.base import Etf


@dataclass
class IndividualEtfInvestmentStrategy:
    """TODO"""

    etf: Etf
    investment_amounts: npt.NDArray
    buy_fee: float


def get_investment_matrix(
    individual_investment_strategies: list[IndividualEtfInvestmentStrategy],
) -> npt.NDArray:
    """TODO"""

    # TODO: include buy_fees
    max_length = np.max(
        [
            len(strategy.investment_amounts)
            for strategy in individual_investment_strategies
        ]
    )
    return np.vstack(
        [
            np.pad(
                strategy.investment_amounts,
                pad_width=max_length - len(strategy.investment_amounts),
            )[:max_length]
            for strategy in individual_investment_strategies
        ]
    )


class CombinedEtfInvestmentStrategy:
    """TODO"""

    def __init__(
        self,
        *,
        individual_investment_strategies: list[IndividualEtfInvestmentStrategy]
        | None = None,
        etfs_investment_matrix: tuple[list[Etf], npt.NDArray] | None = None,
    ):
        if individual_investment_strategies is not None:
            self.etfs = [
                investment_strategy.etf
                for investment_strategy in individual_investment_strategies
            ]
            self.investment_matrix: npt.NDArray = get_investment_matrix(
                individual_investment_strategies
            )

        if etfs_investment_matrix is not None:
            self.etfs, self.investment_matrix = etfs_investment_matrix


@dataclass
class WeightedEtfInvestmentStrategy:
    """TODO"""

    etfs: list[Etf]
    investment_amounts: npt.NDArray
    weights: npt.NDArray
    buy_fees: npt.NDArray[float]

    _combined_strategy: CombinedEtfInvestmentStrategy | None = None

    def __post_init__(self):
        if len(self.etfs) != len(self.weights):
            raise ValueError(
                f"Number of weights {len(self.weights)} must correspond to"
                f" the number of etfs {len(self.etfs)}."
            )
        if np.sum(self.weights) != 1:
            # TODO: raise warning here.
            self.weights /= np.sum(self.weights)

        if len(self.etfs) != len(self.buy_fees):
            raise ValueError(
                f"Number of buy fees {len(self.buy_fees)} must be the same"
                f" as number of etfs {len(self.etfs)}."
            )

    @property
    def combined_strategy(self):
        if self._combined_strategy is None:
            self._combined_strategy = self.construct_combined_strategy()
        return self._combined_strategy

    def construct_combined_strategy(
        self,
    ) -> CombinedEtfInvestmentStrategy:
        investment_matrix = np.tensordot(self.weights, self.investment_amounts, axes=0)
        return CombinedEtfInvestmentStrategy(
            etfs_investment_matrix=(self.etfs, investment_matrix)
        )
        #
        # investment_strategies = []
        # for i in range(len(self.etfs)):
        #     investment_amounts_single_etf = self.weights[i] * self.investment_amounts
        #     investment_strategies.append(
        #         IndividualEtfInvestmentStrategy(
        #             self.etfs[i], investment_amounts_single_etf, self.buy_fees[i]
        #         )
        #     )
        # return CombinedEtfInvestmentStrategy(
        #     individual_investment_strategies=investment_strategies
        # )


# class InvestmentStrategy:
#     """contains parameters of investment strategy."""
#
#     def __init__(
#         self,
#         *,
#         etf_individual_strategies: list[IndividualEtfInvestmentStrategy] | None,
#         etf_weighted_strategy: WeightedEtfInvestmentStrategy | None,
#     ):
#         if etf_weighted_strategy is not None and etf_individual_strategies is not None:
#             raise ValueError(
#                 "Both individual strategies and weighted strategy given. "
#                 "Please decide which one should be used to initialize "
#                 "investment strategy and pass only that one."
#             )
#
#         if etf_weighted_strategy is None and etf_individual_strategies is None:
#             raise ValueError("No strategy given.")
#
#         if etf_individual_strategies is not None:
#             self.etf_strategies = etf_individual_strategies
#
#         if etf_weighted_strategy is not None:
#             self.etf_strategies = (
#                 etf_weighted_strategy.construct_individual_investment_strategies()
#             )
