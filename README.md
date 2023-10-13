# SABR-model-calibration
In this repo we calibrate SABR model for BTC options using market data from [Deribit](https://www.deribit.com/). 
Calibration is performed using the Levenberg-Marquardt Method method on OTM options.

Main results you can see [here](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/SABR_calibration.ipynb)

[External_functions.py](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/external_functions.py) file contains function for calculating 
implied volatility using the Hagan formula, function for Levenberg-Marquardt Method and some other functions. 

[SABR_class.py](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/SABR_class.py) file contains SABR class. This class has methods fit_iv, predict_iv, predict_call_price, predict_put_price. The first one fits the parameters of the SABR model to the market implied volatility while the others predict volatility call and put prices using calibrated parameters.

### SABR model definition
Under the SABR (stochastic $\\alpha, \\beta, \\rho$) model the stock price is governed by the following system of SDE:
```math
\begin{equation}
 \begin{cases}
   dF_t = \alpha_t F_t^{\beta} dW^1_t\\
   d\alpha_t = v \alpha_t dW^2_t\\
   dW^1_t dW^2_t = \rho dt
 \end{cases}
\end{equation}
```
with initial conditions  on $F, \\alpha$. The parameters satisfy conditions $0 \\leq \\beta \\leq 1$, $\\alpha \\geq 0, v \\geq 0$, $-1 \\leq \\rho \\leq 1$

### Hagan formula
The approximate formula for implied volatility is valid:
```math
\begin{aligned}
  &\: \sigma_{impl} = \frac{\alpha S}{D} \cdot \frac{z}{X(z, \rho)}
\end{aligned}
```
where 
```math
\begin{aligned}
 &\: F_{mid} = \sqrt{F K} \\
 &\: z = \frac{v}{\alpha}F_{mid}^{1-\beta}\log \frac{F}{K} \\
 &\: X(z, \rho) = \log \frac{\sqrt{1 - 2 z \rho + z^2} + z - \rho}{1-\rho} \\
 &\: r_1 = \frac{(\beta-1)^2 \alpha^2 F_{mid}^{2\beta - 2}}{24} \\
 &\: r_2 = \frac{\rho \beta \alpha v F_{mid}^{\beta - 1}}{4} \\
 &\: r_3 = \frac{2-3\rho^2}{24}v^2 \\
 &\: S = 1 + T(r_1 + r_2 + r_3) \\
 &\: D = F_{mid}^{1-\beta}\left[ 1 + \frac{(\beta-1)^2}{24} \log^2 \frac{F}{K} + \frac{(\beta-1)^4}{1920}\log^4 \frac{F}{K}\right]
\end{aligned}
```

