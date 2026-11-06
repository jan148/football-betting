from config import *
from procedures import *
'''
# Grid search
df = grid_search_kelly() if type(c_w_model) == CWKellyEdgeProba else grid_search_mispricing()
print(df)
'''
# Cross-validation
res_dict, simulations = cross_validation()
df = pd.DataFrame(res_dict).transpose()
df_res = df[["Total number of bets", "Number of selected bets", "Gross return", "Long run growth factor"]].sort_values(by=["Total number of bets"], ascending=False)
print(df_res)
SimulationPlotter.plot_simulations(simulations)