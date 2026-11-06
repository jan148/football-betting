# Football bets wager model

Wager model for football bets (only general result bets). Note that this repo only contains a part of the project.

## Main

```python
# Grid search
df = grid_search_kelly() if type(c_w_model) == CWKellyEdgeProba else grid_search_mispricing()
print(df)

# Cross-validation
res_dict, simulations = cross_validation()
df = pd.DataFrame(res_dict).transpose()
df_res = df[["Total number of bets", "Number of selected bets", "Gross return", "Long run growth factor"]].sort_values(by=["Total number of bets"], ascending=False)
print(df_res)
SimulationPlotter.plot_simulations(simulations)
```

## Methodology

See [documentation](Documentation/Documentation.pdf).