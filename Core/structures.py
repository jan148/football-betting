import math
import matplotlib.pyplot as plt
from Common.common_imports import *

class Bet:
    def __init__(self, arr):
        self.arr = arr
        self.res = arr[0]
        self.win_ind = arr[1]
        self.date_num = arr[2]
        self.h_win = arr[3]
        self.draw = arr[4]
        self.a_win = arr[5]
        self.games_played_perc = arr[6]
        self.h_impl_prob = arr[7]
        self.d_impl_prob = arr[8]
        self.a_impl_prob = arr[9]
        self.bet_ind = int(arr[10])
        self.bet_odds = arr[11]
        self.league_index = arr[12]
        self.h_team_rank = arr[13]
        self.a_team_rank = arr[14]
        self.id = int(arr[15])

    def get_id(self):
        return self.id

    def is_favourite(self):
        return self.bet_ind == int(np.argmax([self.h_impl_prob, self.d_impl_prob, self.a_impl_prob]))

    def get_feature_vector(self):
        return self.arr[[3, 4, 5, 10, 11, 12, 13, 14]]

    def get_win_ind(self):
        return self.win_ind

    def get_p_implied(self):
        return [self.h_impl_prob, self.d_impl_prob, self.a_impl_prob][self.bet_ind]

    def get_gross_return_per_unit(self):
        return self.win_ind * self.bet_odds - 1

    # 1 = Bookie was WRONG
    def get_bookie_wrong_prediction_indicator(self):
        if self.h_impl_prob > self.d_impl_prob and self.h_impl_prob > self.a_impl_prob:
            bookie_prediction = 0
        elif self.d_impl_prob > self.h_impl_prob and self.d_impl_prob > self.a_impl_prob:
            bookie_prediction = 1
        else:
            bookie_prediction = 2
        return int(bookie_prediction != self.res)

    def get_bets_same_game(self, bets):
        return [bet for bet in bets if bet.get_id() == self.id ]

    @staticmethod
    def as_array(bets):
        bets_arr = []
        for bet in bets:
            bets_arr.append(bet.arr)
        return np.asarray(bets_arr, dtype=float)

    @staticmethod
    def get_implied_probs(bets):
        return np.asarray([bet.get_p_implied() for bet in bets], dtype=float)

class BettingSimulation:
    def __init__(self, id):
        self.id = id
        self.bankroll = []
        self.gross_returns = []
        self.n_bets_all = None
        self.n_bets_selected = None

    def simulate(self, wager_model, bets_all, init_br=1):
        # 1. Select bets
        bets_selected = wager_model.choose(bets_all)

        self.n_bets_all = len(bets_all)
        self.n_bets_selected = len(bets_selected)

        # 2. Simulate
        self.bankroll = [init_br]
        for bet in bets_selected:
            wager = wager_model.wager(bet, self.bankroll[-1])
            net_gain = wager * bet.get_gross_return_per_unit()
            self.bankroll.append(self.bankroll[-1] + net_gain)
        self.bankroll = np.asarray(self.bankroll, dtype=float)

        # 3. Gross returns
        self.gross_returns = self.bankroll[1:] / self.bankroll[:-1]

    # S_T / S_0 = R_T * R_T-1 * ... * R_1
    # ln(S_T / S_0) = ln(R_T) + ln(R_T-1) + ... + ln(R_1)
    # Assume that R_j, R_i for i \neq j iid.
    # Keep in mind: By lln, we can only estimate ln((S_T / S_0)^(1/n)) -> E[ln(R_i)]
    # !!! What we will actually get per bet on average will be e^E[ln(R_i)] !!! (On average: Playing bets drawn iid from the one bet distribution)
    def get_stats(self):
        n_bets = len(self.bankroll) - 1
        gross_return = self.bankroll[-1] / self.bankroll[0] # Br_final / B_0
        log_growth_factor_evolution = np.cumsum(np.log(self.gross_returns)) / np.arange(1, n_bets + 1)
        long_run_growth_factor_evolution = np.exp(log_growth_factor_evolution)
        log_growth_factor = log_growth_factor_evolution[-1]
        long_run_growth_factor= math.exp(log_growth_factor)
        std_dev_log_growths = np.std(np.log(self.gross_returns))
        stats_dict = {self.id:
                          {
                           "Total number of bets": self.n_bets_all,
                           "Number of selected bets": self.n_bets_selected,
                           "Gross return": gross_return,
                           "Log growth factor evolution": log_growth_factor_evolution,
                           "Log growth factor": log_growth_factor,
                           "Long run growth factor evolution": long_run_growth_factor_evolution,
                           "Long run growth factor": long_run_growth_factor,
                           "Std. dev. log growth factor": std_dev_log_growths}
                      }
        return stats_dict

class SimulationPlotter:
    @staticmethod
    def plot_simulations(simulations):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        fig.suptitle("Simulation results")

        axs[0].title.set_text("Bankroll evolution")
        axs[1].title.set_text("Average growth factor evolution")

        y_min_0, y_max_0 = 1, 1
        y_min_1, y_max_1 = 1, 1

        for simulation in simulations:
            axs[0].plot(simulation.bankroll)
            axs[1].plot(simulation.get_stats()[simulation.id]["Long run growth factor evolution"])

            y_min_0, y_max_0 = (min(y_min_0, simulation.bankroll[-1])
                                , max(y_max_0, simulation.bankroll[-1]))
            y_min_1, y_max_1 = (min(y_min_1, simulation.get_stats()[simulation.id]["Long run growth factor"])
                                , max(y_max_1, simulation.get_stats()[simulation.id]["Long run growth factor"]))

        axs[0].set_ylim([y_min_0, y_max_0])
        axs[1].set_ylim([y_min_1, y_max_1])

        pad = lambda arrs: [np.pad(np.asarray(a, dtype=float), (0, max(map(len, arrs)) - len(a)), constant_values=np.nan) for a in arrs]
        axs[0].plot(np.nanmean(pad([s.bankroll for s in simulations]), axis=0), color="black", linewidth=2, label="Mean")
        axs[1].plot(np.nanmean(pad([s.get_stats()[s.id]["Long run growth factor evolution"] for s in simulations]), axis=0), color="black", linewidth=2, label="Mean")

        axs[0].axhline(y=1, color="black", linestyle="dashed")
        axs[1].axhline(y=1, color="black", linestyle="dashed")

        plt.show()