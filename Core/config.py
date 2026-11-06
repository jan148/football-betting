from models import CWMispricing, CWKellyEdgeProba

# The fractional Kelly strategy
kelly_multiplier = 0.7
edge = 0.1
c_w_model = CWKellyEdgeProba(edge, kelly_multiplier)

'''
# The mispricing Kelly strategy
kelly_multiplier = 0.4
p_bookie_wrong_thres = 0.55
c_w_model = CWMispricing(kelly_multiplier, p_bookie_wrong_thres)
'''
