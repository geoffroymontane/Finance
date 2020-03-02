from scipy.stats import norm
import math




def bs_call_price(S_0, T, K, r, sigma):

    d_1 = ((math.log(S_0 / K) + T * (r + 0.5 * sigma ** 2))
          / (sigma * math.sqrt(T)))
    d_2 = d_1 - sigma * math.sqrt(T)
    return S_0 * norm.cdf(d_1) - K * math.exp(-r * T) * norm.cdf(d_2)


def bs_call_price_derivative(S_0, T, K, r, sigma):

    d1 = ((math.log(S_0 / K) + T * (r + 0.5 * sigma ** 2))
          / (sigma * math.sqrt(T)))
    d2 = d1 - sigma * math.sqrt(T)

    d1_derivative = (math.sqrt(T) - (math.log(S_0 / K)
        + (1 + 0.5 * sigma ** 2) * T) / (math.sqrt(T) * sigma ** 2))
    d2_derivative = d1_derivative - math.sqrt(T)

    n1_derivative = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    n2_derivative = math.exp(-0.5 * d2 ** 2) / math.sqrt(2 * math.pi)

    return S_0 * n1_derivative * d1_derivative - K * math.exp(-r * T) * n2_derivative * d2_derivative 




def implied_vol_call(S_0, T, K, r, premium, start, precision):

    sigma = start

    error = bs_call_price(S_0, T, K, r, sigma) - premium
   
    while error > precision:
        sigma = (sigma - (bs_call_price(S_0, T, K, r, sigma) - premium)
                / bs_call_price_derivative(S_0, T, K, r, sigma))

        error = bs_call_price(S_0, T, K, r, sigma) - premium

    return sigma


sigma = implied_vol_call(31.55, 3.5, 22.75, 0.05, 13, 0.5, 0.001)
print(sigma)
print(bs_call_price(31.55, 3.5, 22.75, 0.05, sigma))
