import numpy as np
from scipy import stats as sps


def black_scholes(K: np.ndarray, F: np.ndarray,
                  T: np.ndarray, r: float, vol: float) -> np.ndarray:
    """
    Returns the price of a call option with passed parameters:

    Parameters
        ----------
            K: strike or array of strikes, 
            F: underlying price or array of underlying prices, 
            T: expiration time or array of expiration times, 
            r: interest rate, 
            vol: volatility in BS model

    Returns:
        ----------
            call_price: call price
    """
    d1 = (np.log(F / K) + 0.5 * vol ** 2 * T)         / (vol * np.sqrt(T) + 1e-10)
    d2 = d1 - vol * np.sqrt(T)
    D = np.exp(-r * T)
    call_price = D * (F * sps.norm.cdf(d1) - K * sps.norm.cdf(d2))
    return call_price

