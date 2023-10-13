import numpy as np
from typing import Callable





def Hagan_implied_vol(K: np.ndarray, F: np.ndarray, T: np.ndarray, alpha: float,
                      beta: float, rho: float, v: float) -> np.ndarray:
    """
    Function returns approximation for impied volatility in SABR model using Hagan formula

    Parameters
        ----------
        K : np.ndarray
            array of strikes
        F : np.ndarray
            futures price
        T : np.ndarray
            expiration time
        alpha: float
            parameter of SABR model
        beta: float
            parameter of SABR model
        rho: float
            parameter of SABR model
        v: float
            parameter of SABR model

    Returns
        ----------
        iv: np.ndarray
            implied volatility
    """
    F_mid = np.sqrt(F * K)
    r1 = (beta - 1) ** 2 * alpha ** 2 * F_mid ** (2 * beta - 2) / 24
    r2 = rho * beta * alpha * v * F_mid ** (beta - 1) / 4
    r3 = (2 - 3 * rho ** 2) / 24 * v ** 2
    S = 1 + T * (r1 + r2 + r3)
    z = v / alpha * F_mid ** (1 - beta) * np.log(F / K)
    sqrt = np.sqrt(1 - 2 * rho * z + z ** 2)
    X = np.log((sqrt + z - rho) / (1-rho))
    D = F_mid ** (1-beta) * (1 + (beta-1)**2/24 * (np.log(F/K))
                             ** 2 + (beta-1)**4/1920 * (np.log(F/K))**4)

    iv = alpha * S * z / D / X
    return iv




def SABR_Jacobian(K: np.ndarray, F: np.ndarray, T: np.ndarray, alpha: float,
                  beta: float, rho: float, v: float) -> tuple:
    """
    Function returns derivatives of implied volatility with respect to SABR parameters

    Parameters
        ----------
        K : np.ndarray
            array of strikes
        F : np.ndarray
            futures price
        T : np.ndarray
            expiration time
        alpha: float
            parameter of SABR model
        beta: float
            parameter of SABR model
        rho: float
            parameter of SABR model
        v: float
            parameter of SABR model

    Returns
        ----------
        iv: np.ndarray
            implied vol
        iv_alpha: np.ndarray
            derivative with respect to alpha
        iv_beta: np.ndarray
            derivative with respect to beta
        iv_rho: np.ndarray
            derivative with respect to rho
        iv_v: np.ndarray:
            derivative with respect to v(volatility of volatility)    
    """
    F_mid = np.sqrt(F * K)
    r1 = (beta - 1) ** 2 * alpha ** 2 * F_mid ** (2 * beta - 2) / 24
    r2 = rho * beta * alpha * v * F_mid ** (beta - 1) / 4
    r3 = (2 - 3 * rho ** 2) / 24 * v ** 2
    S = 1 + T * (r1 + r2 + r3)
    z = v / alpha * F_mid ** (1 - beta) * np.log(F / K)
    sqrt = np.sqrt(1 - 2 * rho * z + z ** 2)
    X = np.log((sqrt + z - rho) / (1-rho))
    D = F_mid ** (1-beta) * (1 + (beta-1)**2/24 * (np.log(F/K))
                             ** 2 + (beta-1)**4/1920 * (np.log(F/K))**4)

    iv = alpha * S * z / D / X

    X_z = 1 / sqrt

    S_alpha = T * (2 * r1 + r2) / alpha
    z_alpha = -z / alpha
    X_alpha = X_z * z_alpha

    S_rho = T * v * (beta * alpha * F_mid**(beta - 1) - rho * v) / 4
    X_rho = 1 / (1 - rho) - 1 / sqrt * (sqrt + z) / (sqrt + z - rho)

    S_v = T / v * (r2 + 2 * r3)
    z_v = z / v
    X_v = X_z * z_v

    z_beta = -np.log(F_mid) * z
    X_beta = X_z * z_beta
    S_beta = T * (2 * r1 * (1/(beta-1)+np.log(F_mid)) +
                  r2 * (1/beta + np.log(F_mid)))
    D_beta = -np.log(F_mid) * D + F_mid**(1-beta) * ((beta-1) /
                                                     12 * (np.log(F/K))**2 + (beta-1)**3/480 * (np.log(F/K))**4)

    logs_alpha = 1 / alpha + S_alpha / S + z_alpha / z - X_alpha / X
    logs_v = S_v / S + z_v / z - X_v / X
    logs_beta = S_beta / S - D_beta / D + z_beta / z - X_beta / X
    logs_rho = S_rho / S - X_rho / X

    iv_alpha = iv * logs_alpha
    iv_v = iv * logs_v
    iv_beta = iv * logs_beta
    iv_rho = iv * logs_rho

    return iv, iv_alpha, iv_beta, iv_rho, iv_v





def generate_random_params() -> np.ndarray:
    """
    This function gerenate random parameters for sabr model

    Returns:
        ----------
        sabr_params: np.ndarray
                generated sabr params
    """
    eps = 1e-5
    alpha = 0.3 * np.random.rand(1) + eps
    v = 1.0 * np.random.rand(1) + eps
    beta = 0.1 + 0.8 * np.random.rand(1)
    rho = -0.9 + (1.8) * np.random.rand(1)
    return np.asarray([alpha[0], beta[0], rho[0], v[0]])





def SABR_proj(sabr_params: np.ndarray) -> np.ndarray:
    """
    Funciton project sabr parameters into valid range

    Parameters
        ----------
        sabr_params: np.ndarray 
                model parameters

    Returns:
        ----------
        sabr_params: np.ndarray 
                clipped parameters
    """
    alpha, beta, rho, v = sabr_params

    eps = 1e-6

    alpha = max(alpha, eps)
    v = max(v, eps)
    rho = np.clip(rho, -1 + eps, 1 - eps)
    beta = np.clip(beta, eps, 1 - eps)

    return np.asarray([alpha, beta, rho, v])





def Levenberg_Marquardt_optimization(Niter: int,
                                     f: Callable,
                                     proj: Callable,
                                     x0: np.ndarray) -> dict:
    ''' 
    Nonlinear least squares method, Levenberg-Marquardt Method

    Parameters
        ----------
            Niter: int
                number of iteration
            f: callable 
                function gets vector of model parameters x as input, 
                returns tuple res, J, where res is numpy vector of residues, 
                J is jacobian of residues with respect to x 
            proj: callable
                function gets vector of model parameters x, returns vector of projected parameters 
            x0: np.ndarray
                initial parameters

    Returns:
        ----------
            result: dict
                dictionary with results
    '''
    x = x0.copy()

    mu = 100.0
    nu1 = 2.0
    nu2 = 2.0

    fs = []
    res, J = f(x)
    F = np.linalg.norm(res)

    result = {"xs": [x], "objective": [F], "x": None}

    for i in range(Niter):
        I = np.diag(np.diag(J @ J.T)) + 1e-5 * np.eye(len(x))
        dx = np.linalg.solve(mu * I + J @ J.T, J @ res)
        x_ = proj(x - dx)
        res_, J_ = f(x_)
        F_ = np.linalg.norm(res_)
        if F_ < F:
            x, F, res, J = x_, F_, res_, J_
            mu /= nu1
            result['xs'].append(x)
            result['objective'].append(F)
        else:
            i -= 1
            mu *= nu2
            continue
        eps = 1e-10
        if F < eps:
            break
        result['x'] = x
    return result

