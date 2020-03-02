from scipy.stats import norm
import math




def bs_call_price(S_0, T, K, r, sigma):

    d_1 = ((math.log(S_0 / K) + T * (r + 0.5 * sigma ** 2))
          / (sigma * math.sqrt(T)))
    d_2 = d_1 - sigma * math.sqrt(T)
    return S_0 * norm.cdf(d_1) - K * math.exp(-r * T) * norm.cdf(d_2)


print(bs(31.55, 3.5, 22.75, 0.05, 0.5))
