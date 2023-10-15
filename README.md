# SABR-model-calibration
In this repo we calibrate SABR model for BTC options using market data from [Deribit](https://www.deribit.com/). 
Calibration is performed using the Levenberg-Marquardt method for OTM options.

Main results can be seen [here](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/SABR_calibration.ipynb)

[External_functions.py](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/external_functions.py) file contains a function for calculating 
implied volatility using the Hagan formula,  a function for Levenberg-Marquardt method and some other functions. 

[SABR_class.py](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/SABR_class.py) file contains SABR class. This class has methods fit_iv, predict_iv, predict_call_price, predict_put_price. The first one fits the parameters of the SABR model to the market implied volatility while the others predict volatility, call and put prices using calibrated parameters.

### SABR model definition
Under the SABR (stochastic $\\alpha, \\beta, \\rho$) model the stock price obeys the following system of SDE:
```math
\begin{equation}
 \begin{cases}
   dF_t = \alpha_t F_t^{\beta} dW^1_t\\
   d\alpha_t = v \alpha_t dW^2_t\\
   dW^1_t dW^2_t = \rho dt\\
   F_0 = F &\text{initial condition}\\
   \alpha_0 = \alpha &\text{initial condition}
 \end{cases}
\end{equation}
```
The parameters satisfy conditions $0 \\leq \\beta \\leq 1$, $\\alpha \\geq 0, v \\geq 0$, $-1 \\leq \\rho \\leq 1$

### Hagan formula
The approximate formula for implied volatility is valid:
```math
\begin{aligned}
  &\: \hat{\sigma}(T,K,\alpha,\beta,\rho,v) = \frac{\alpha S}{D} \cdot \frac{z}{X(z)}
\end{aligned}
```
where 
```math
\begin{aligned}
 &\: \quad\quad\quad\quad\quad\quad\quad F_{mid} = \sqrt{F K} \\
 &\: \quad\quad\quad\quad\quad\quad\quad z = \frac{v}{\alpha}F_{mid}^{1-\beta}\log \frac{F}{K} \\
 &\: \quad\quad\quad\quad\quad\quad\quad X(z) = \log \frac{\sqrt{1 - 2 z \rho + z^2} + z - \rho}{1-\rho} \\
 &\: \quad\quad\quad\quad\quad\quad\quad r_1 = \frac{(\beta-1)^2 \alpha^2 F_{mid}^{2\beta - 2}}{24} \\
 &\: \quad\quad\quad\quad\quad\quad\quad r_2 = \frac{\rho \beta \alpha v F_{mid}^{\beta - 1}}{4} \\
 &\: \quad\quad\quad\quad\quad\quad\quad r_3 = \frac{2-3\rho^2}{24}v^2 \\
 &\: \quad\quad\quad\quad\quad\quad\quad S = 1 + T(r_1 + r_2 + r_3) \\
 &\: \quad\quad\quad\quad\quad\quad\quad D = F_{mid}^{1-\beta}\left[ 1 + \frac{(\beta-1)^2}{24} \log^2 \frac{F}{K} + \frac{(\beta-1)^4}{1920}\log^4 \frac{F}{K}\right]
\end{aligned}
```
### Levenberg–Marquardt algorithm
Let's denote $\left(K_i,T_i,\sigma^{\text{market}}(K_i,T_i)\right)_{i=1}^n$ - market data for OTM options. We'll optimize $\vec{\theta} = (\alpha,\beta,\rho,v)$ using Levenberg–Marquardt algorithm:
```math
\begin{equation}
     \vec{\theta}_{k+1} = \vec{\theta_{k}} - (\lambda\text{diag}(JJ^T) + JJ^T)^{-1}J\vec{r}
\end{equation}
```
where
```math
\vec{r}=\begin{pmatrix} \hat{\sigma}(K_1,T_1,\vec{\theta_k}) - \sigma^{\text{market}}(K_1,T_1) \\ \hat{\sigma}(K_2,T_2,\vec{\theta_k}) - \sigma^{\text{market}}(K_2,T_2)  \\ \vdots \\ \hat{\sigma}(K_n,T_n,\vec{\theta_k}) - \sigma^{\text{market}}(K_n,T_n)  \end{pmatrix}\;
 {J}=\begin{pmatrix}
\frac{\partial r_1}{\partial\alpha} & \frac{\partial r_2}{\partial\alpha} & \cdots & \frac{\partial r_n}{\partial\alpha} \\
\frac{\partial r_1}{\partial\beta} & \frac{\partial r_2}{\partial\beta} & \cdots & \frac{\partial r_n}{\partial\beta}\\
\frac{\partial r_1}{\partial\rho} & \frac{\partial r_2}{\partial\rho} & \cdots & \frac{\partial r_n}{\partial\rho}\\
\frac{\partial r_1}{\partial v} & \frac{\partial r_2}{\partial v} & \cdots & \frac{\partial r_n}{\partial v}
\end{pmatrix}
```
also we have loss function L to control $\lambda$ on each step
```math
\begin{equation}
   L(\vec{\theta})  = \sum_{i=1}^n(\hat{\sigma}(K_i,T_i,\vec{\theta}) - \sigma^{\text{market}}(K_i,T_i))^2
\end{equation}
```
