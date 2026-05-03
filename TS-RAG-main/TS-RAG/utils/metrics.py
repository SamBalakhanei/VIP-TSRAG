import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + 1e-8)))

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
    # return np.mean(200 * np.abs(pred - true) / (pred + true + 1e-8))

def ND(pred, true):
    return np.mean(np.abs(true - pred)) / np.mean(np.abs(true))

def MASE(pred, true, insample, seasonality=1):
    if len(insample) <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(insample[seasonality:] - insample[:-seasonality]))
    if scale <= 1e-12:
        return float("nan")
    return float(np.mean(np.abs(true - pred)) / scale)

def WQL(quantile_forecast, true, quantiles):
    # quantile_forecast shape: [num_quantiles, prediction_length]
    denom = np.sum(np.abs(true)) + 1e-8
    total = 0.0
    for qi, q in enumerate(quantiles):
        forecast_q = quantile_forecast[qi]
        diff = true - forecast_q
        loss_q = np.maximum(q * diff, (q - 1.0) * diff)
        total += np.sum(loss_q)
    return total / denom

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    smape = SMAPE(pred, true)
    nd = ND(pred, true)

    return mae, mse, rmse, mape, mspe, smape, nd
