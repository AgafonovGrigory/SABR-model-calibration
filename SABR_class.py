from black_scholes_model import black_scholes
from external_functions import Hagan_implied_vol, SABR_Jacobian, generate_random_params, SABR_proj
from external_functions import Levenberg_Marquardt_optimization
import numpy as np
import matplotlib.pyplot as plt





class SABR:
    def __init__(self, sabr_params=None):
        self.sabr_params = sabr_params

    def fit_iv(self,
               iv0: np.ndarray,
               K: np.ndarray,
               F: float,
               T: float,
               Niter: int = 100,
               weights=None,
               sabr_params=None) -> dict:
        """
        This method calibrates the parameters of the SABR model to market implied volatility and changes
        self.sabr_params to the calibrated ones


        Parameters
        `  ----------
                iv0: np.ndarray
                    array of market implied volatility
                K: np.ndarray
                    array of strikes
                F: np.ndarray
                    underlying futures price
                T: np.ndarray
                    expiration time
                Niter: int 
                    number of iterations
                weights:np.ndarray
                    array of weights
                sabr_params: np.ndarray 
                    initial model params

        Returns:
        `  ----------
                result: dict
                    dict of Levenberg_Marquardt_optimization
        """
        F = np.ones_like(iv0) * F
        T = np.ones_like(iv0) * T
        if weights is None:
            weights = np.ones_like(iv0)
        weights = weights / np.sum(weights)
        if sabr_params is None:
            sabr_params = generate_random_params()

        def get_residals(sabr_params: np.ndarray) -> tuple:
            '''
            This function calculates residuals and Jacobian matrix
            Parameters
        `         ----------
                    sabr_params: np.ndarray
                        model params
                Returns:
        `         ----------
                    res: np.ndarray
                        vector or residuals
                    J: np.ndarray
                        Jacobian
            '''
            alpha, beta, rho, v = sabr_params
            iv, iv_alpha, iv_beta, iv_rho, iv_v = SABR_Jacobian(
                K, F, T, alpha, beta, rho, v)
            res = iv - iv0
            J = np.asarray([iv_alpha, iv_beta, iv_rho, iv_v])
            return res * weights, J @ np.diag(weights)

        # optimization
        result = Levenberg_Marquardt_optimization(
            Niter, get_residals, SABR_proj, sabr_params)
        self.sabr_params = result['x']

        return result

    def predict_iv(self, K: np.ndarray,
                   F: np.ndarray,
                   T: np.ndarray) -> np.ndarray:
        """
        Method returns implied volatility for given parameters K, F, T 

        Parameters
        `  ----------
                K: np.ndarray
                    array of strikes
                F: np.ndarray
                    underlying futures price
                T: np.ndarray
                    expiration time
        Returns:
        `  ----------
                iv: np.ndarray
                    implied volatility
        """
        iv = Hagan_implied_vol(K, F, T, *self.sabr_params)
        return iv

    def predict_call_price(self, K: np.ndarray,
                           F: np.ndarray,
                           T: np.ndarray) -> np.ndarray:
        """
        The method returns call option price for given parameters K, F, T 

        Parameters
        `  ----------
                K: np.ndarray
                    array of strikes
                F: np.ndarray
                    underlying futures price
                T: np.ndarray
                    expiration time
        Returns:
        `  ----------
                C: np.ndarray
                    call option price
        """
        vol = self.predict_iv(K, F, T)
        C = black_scholes(K, F, T, 0, vol)
        return C

    def predict_put_price(self, K: np.ndarray,
                           F: np.ndarray,
                           T: np.ndarray) -> np.ndarray:
        """
        The method returns put option price for given parameters K, F, T 

        Parameters
        `  ----------
                K: np.ndarray
                    array of strikes
                F: np.ndarray
                    underlying futures price
                T: np.ndarray
                    expiration time
        Returns:
        `  ----------
                P: np.ndarray
                    put option price
        """
        C = self.predict_call_price(K, F, T)
        P = C + (K - F)
        return P

