import random
from models import *
import pandas as pd
import pickle
from config import c_w_model

def grid_search_kelly():
    edge_grid = np.arange(0.0, 0.5 + 0.1, 0.1)
    kelly_muliplier_grid = np.arange(0.0, 1. + 0.1, 0.1)

    time_start = 20200101
    time_end = 20251001

    res_dicts = {}
    for edge in edge_grid:
        for k_m in kelly_muliplier_grid:
            print(f"Started run for edge = {edge} and kelly multiplier = {k_m}")

            try:
                c_w_model.edge = edge
                c_w_model.kelly_multiplier = k_m

                res_dict, simulations = cross_validation(seeds = 1, time_start = time_start, time_end = time_end)

                growth_fac = res_dict[simulations[0].id]["Long run growth factor"]
                out_dict = {f"{simulations[0].id}": {"Edge": edge, "Kelly multiplier": k_m, "Growth factor": growth_fac}}
                res_dicts = res_dicts | out_dict
            except:
                continue

    df = pd.DataFrame(res_dicts).transpose().sort_values("Growth factor", ascending=False)
    return df

def grid_search_mispricing():
    threshold_grid = np.arange(0.0, 1. + 0.1, 0.2)
    kelly_muliplier_grid = np.arange(0.0, 1. + 0.1, 0.2)

    time_start = 20200101
    time_end = 20251001

    res_dicts = {}
    for thres in threshold_grid:
        for k_m in kelly_muliplier_grid:
            print(f"Started run for threshold = {thres} and kelly multiplier = {k_m}")

            try:
                c_w_model.p_bookie_wrong_thres = thres
                c_w_model.kelly_multiplier = k_m

                res_dict, simulations = cross_validation(seeds=1, time_start = time_start, time_end = time_end)

                growth_fac = res_dict[simulations[0].id]["Long run growth factor"]
                out_dict = {f"{simulations[0].id}": {"Probability bookie wrong": thres, "Kelly multiplier": k_m, "Growth factor": growth_fac}}
                res_dicts = res_dicts | out_dict
            except:
                continue

    df = pd.DataFrame(res_dicts).transpose().sort_values("Growth factor", ascending=False)
    return df

def cross_validation(seeds = 30, time_start = 20230101, time_end = 20251001):
    data_pts_isolated_bets = prepare_bets()

    seeds = np.random.randint(low=0, high=100000, size=seeds)
    res_dict = {}
    simulations = []
    for seed in seeds:
        time_pt_num_1, time_pt_num_2, t_end = get_cutoff_dates(time_start, time_end, seed)

        dates = data_pts_isolated_bets[:, 2]
        data_pts_isolated_bets_train = data_pts_isolated_bets[
            (dates > time_start) & (dates < time_pt_num_1)
            ]
        data_pts_isolated_bets_eval = data_pts_isolated_bets[
            (dates >= time_pt_num_1) & (dates < time_pt_num_2)
            ]
        data_pts_isolated_bets_test = data_pts_isolated_bets[
            (dates >= time_pt_num_2) & (dates < t_end)
            ]

        np.random.shuffle(data_pts_isolated_bets_train)
        np.random.shuffle(data_pts_isolated_bets_eval)
        np.random.shuffle(data_pts_isolated_bets_test)

        if len(data_pts_isolated_bets_train) == 0 or len(data_pts_isolated_bets_test) == 0:
            print("No test or train data for this seed, skipping.")
            continue

        # Train classifier
        bets_train = [Bet(arr) for arr in data_pts_isolated_bets_train]
        bets_eval = [Bet(arr) for arr in data_pts_isolated_bets_eval]
        bets_test = [Bet(arr) for arr in data_pts_isolated_bets_test]

        wager_model = Model()
        configure_model_c_w_model(wager_model, c_w_model)
        wager_model.train_probability_model(bets_train, bets_eval)
        selected_bets = wager_model.choose(bets_test)

        if len(selected_bets) == 0:
            continue

        # Simulate
        simulation = BettingSimulation(f"{seed}")
        simulation.simulate(wager_model, bets_test)
        simulations.append(simulation)
        res_dict = res_dict | simulation.get_stats()

    return res_dict, simulations

# ----- Utils -----
def prepare_bets():
    with open("../Data/all_bets.pkl", "rb") as f:
        bets_all = pickle.load(f)
    return bets_all

def get_cutoff_dates(start_num, end_num, seed):
    rng = random.Random(int(seed))
    start = pd.to_datetime(str(start_num))
    end = pd.to_datetime(str(end_num))
    delta = end - start

    rand_frac = 0.2 + rng.random() * 0.4    # uniform on [0.3, 0.65)
    frac_eval = 0.1
    frac_test = 0.1
    d1 = int((start + rand_frac * delta).strftime("%Y%m%d"))
    d2 = int((start + (rand_frac + frac_eval) * delta).strftime("%Y%m%d"))
    d3 = int((start + (rand_frac + frac_eval + frac_test) * delta).strftime("%Y%m%d"))
    return d1, d2, d3