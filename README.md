# SABR-model-calibration
In this repo we calibrate SABR model for BTC options using market data from [Deribit](https://www.deribit.com/). 
Calibration is performed using the Levenberg-Marquardt Method method on OTM options.

Main results you can see [here](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/SABR_calibration.ipynb)

[External_functions.py](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/external_functions.py) file contains function for calculating 
implied volatility using the Hagan formula, function for Levenberg-Marquardt Method and some other functions. 

[SABR_class.py](https://github.com/AgafonovGrigory/SABR-model-calibration/blob/main/SABR_class.py) file contains SABR class. This class has methods fit_iv, predict_iv, predict_call_price, predict_put_price. The first one fits the parameters of the SABR model to the market implied volatility while the others predict volatility call and put prices using calibrated parameters.


