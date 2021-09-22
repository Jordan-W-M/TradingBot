from datetime import datetime
from math import exp

import numpy as np
from typing import Dict

from pandas import DataFrame

from freqtrade.optimize.hyperopt import IHyperOptLoss


# Define some constants:

# set TARGET_TRADES to suit your number concurrent trades so its realistic
# to the number of days
TARGET_TRADES = 600
# This is assumed to be expected avg profit * expected trade count.
# For example, for 0.35% avg per trade (or 0.0035 as ratio) and 1100 trades,
# self.expected_max_profit = 3.85
# Check that the reported Σ% values do not exceed this!
# Note, this is ratio. 3.85 stated above means 385Σ%.
EXPECTED_MAX_PROFIT = 3.0

# max average trade duration in minutes
# if eval ends with higher value, we consider it a failed eval
MAX_ACCEPTED_TRADE_DURATION = 300


class HyperOptLoss(IHyperOptLoss):
    """
    Defines the default loss function for hyperopt
    This is intended to give you some inspiration for your own loss function.

    The Function needs to return a number (float) - which becomes smaller for better backtest
    results.
    """

    @staticmethod
    def hyperopt_loss_function(self, total_profit: list, trade_count: int) -> float:
        period = self.max_date - self.min_date
        days_period = period.days

        total_profit = total_profit - 0.0005
        expected_yearly_return = total_profit.sum()/days_period

        if (np.std(total_profit) != 0.):
            sharp_ratio = expected_yearly_return/np.std(total_profit)*np.sqrt(365)
        else:
            sharp_ratio = -20

        sharp_ratio = -sharp_ratio
        result = sharp_ratio
        self.resultloss = result
        return result
