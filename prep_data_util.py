import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
from pandarallel import pandarallel
from util import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_vega,
    calc_implied_volatility,
    SSVI,
    find_closest_option,
    find_optimal_k,
    df_to_dict,
)

pd.options.mode.chained_assignment = None
pandarallel.initialize(progress_bar=False)


def prep_single_data(spx, date, use_parallel=True):
    filtered_spx = spx[
        (spx["date"] == date)
        & (
            (spx["is_call"] == 1) & (spx["strike_price"] >= spx["forward_price"])
            | (spx["is_call"] != 1) & (spx["strike_price"] <= spx["forward_price"])
        )
    ]

    filtered_spx["log_moneyness"] = np.log(
        filtered_spx["strike_price"] / filtered_spx["forward_price"]
    )
    if use_parallel:
        filtered_spx["implied_volatility"] = filtered_spx.parallel_apply(
            calc_implied_volatility, axis=1
        )
    else:
        filtered_spx["implied_volatility"] = filtered_spx.apply(
            calc_implied_volatility, axis=1
        )
    filtered_spx["reproduced_price"] = black_scholes_price(
        filtered_spx["strike_price"],
        filtered_spx["tau"],
        filtered_spx["forward_price"],
        filtered_spx["is_call"],
        filtered_spx["implied_volatility"],
        filtered_spx["risk_free_rate"]
    )

    filtered_spx["absolute_error"] = abs(
        filtered_spx["reproduced_price"] - filtered_spx["option_price"]
    )

    filtered_spx_low_error = filtered_spx[
        (filtered_spx["absolute_error"] <= 0.01) & (filtered_spx["option_price"] >= 0.1)
    ]

    def objective_function(param, k, t, iv):
        w = SSVI(k, t, param)
        iv_pred = np.sqrt(w / t)
        return np.mean(np.abs(iv - iv_pred) ** 2)

    k = filtered_spx_low_error["log_moneyness"].values
    t = filtered_spx_low_error["tau"].values
    iv = filtered_spx_low_error["implied_volatility"].values

    # Initial guess for sigma, gamma, eta, rho
    initial_param = 0.2, 0.4, 0.2, -0.4

    result = minimize(
        objective_function,
        initial_param,
        args=(k, t, iv),
        bounds=[(1e-5, 10), (1e-5, 0.5 - 1e-5), (1e-5, 1e2), (-1 + 1e-5, 1 - 1e-5)],
    )

    # Best parameter for sigma
    best_param = result.x

    filtered_spx_low_error["iv_ssiv"] = np.sqrt(SSVI(k, t, best_param) / t)

    filtered_spx_low_error["ssvi_price"] = black_scholes_price(
        filtered_spx_low_error["strike_price"],
        filtered_spx_low_error["tau"],
        filtered_spx_low_error["forward_price"],
        filtered_spx_low_error["is_call"],
        filtered_spx_low_error["iv_ssiv"],
        filtered_spx["risk_free_rate"]
    )

    filtered_spx_low_error["delta"] = black_scholes_delta(
        filtered_spx_low_error["strike_price"].values,
        filtered_spx_low_error["tau"].values,
        filtered_spx_low_error["forward_price"].values,
        filtered_spx_low_error["is_call"],
        filtered_spx_low_error["implied_volatility"].values,
    )

    filtered_spx_low_error["vega"] = black_scholes_vega(
        filtered_spx_low_error["strike_price"].values,
        filtered_spx_low_error["tau"].values,
        filtered_spx_low_error["forward_price"].values,
        filtered_spx_low_error["implied_volatility"].values,
        filtered_spx_low_error["risk_free_rate"].values,
    )

    # Define the target deltas for ATM, 25 delta call, and 25 delta put
    target_deltas = [0, 0.25, -0.25]
    ttms = [7, 30, 90]
    criteria = [{"ttm": ttm, "delta": delta} for ttm in ttms for delta in target_deltas]

    closest_options = []
    for criterion in criteria:
        option = find_closest_option(filtered_spx_low_error, **criterion)
        closest_options.append(option)

    closest_options_df = pd.DataFrame(closest_options)

    results = []

    tau_u, F_u, r_u = (
        filtered_spx_low_error.groupby(["tau"])[["forward_price", "risk_free_rate"]]
        .apply(lambda x: x.iloc[0])
        .reset_index()
        .values.T
    )

    for ttm in ttms:
        tau = ttm / 365.0
        forward_price = float(
            CubicSpline(tau_u, F_u, bc_type='not-a-knot', extrapolate=True)(tau)
        )
        for target_delta in target_deltas:
            if target_delta == 0:
                optimal_k = 0
                is_call = 1
            else:
                is_call = 1 if target_delta > 0 else -1
                optimal_k = find_optimal_k(
                    target_delta, tau, forward_price, is_call, best_param
                )

            iv = np.sqrt(SSVI(optimal_k, tau, best_param) / tau)
            delta = black_scholes_delta(
                np.exp(optimal_k) * forward_price, tau, forward_price, is_call, iv
            )

            risk_free_rate = float(
                CubicSpline(tau_u, r_u, bc_type='not-a-knot', extrapolate=True)(tau)
            )

            results.append(
                {
                    "log_moneyness": optimal_k,
                    "tau": tau,
                    "is_call": is_call,
                    "implied_volatility": iv,
                    "delta": delta,
                    "risk_free_rate": risk_free_rate,
                    "forward_price": forward_price,
                }
            )

    results_df = pd.DataFrame(results)

    # Add the missing columns to results_df
    results_df["date"] = filtered_spx_low_error["date"].iloc[0]
    results_df["time_to_maturity"] = (results_df["tau"] * 365).astype(int)
    results_df["strike_price"] = results_df["forward_price"] * np.exp(
        results_df["log_moneyness"]
    )
    results_df["option_price"] = black_scholes_price(
        results_df["strike_price"].values,
        results_df["tau"].values,
        results_df["forward_price"].values,
        results_df["is_call"].values,
        results_df["implied_volatility"].values,
        results_df["risk_free_rate"].values
    )
    results_df["iv_ssiv"] = results_df["implied_volatility"]
    results_df["ssvi_price"] = results_df["option_price"]
    results_df["delta"] = black_scholes_delta(
        results_df["strike_price"].values,
        results_df["tau"].values,
        results_df["forward_price"].values,
        results_df["is_call"].values,
        results_df["implied_volatility"].values,
    )
    results_df["vega"] = black_scholes_vega(
        results_df["strike_price"].values,
        results_df["tau"].values,
        results_df["forward_price"].values,
        results_df["implied_volatility"].values,
        results_df["risk_free_rate"].values,
    )

    simplified_dict_full = df_to_dict(filtered_spx_low_error)
    simplified_dict_few = df_to_dict(closest_options_df)
    simplified_dict_virtual = df_to_dict(results_df)

    final_result = {
        "data": simplified_dict_full,
        "few": simplified_dict_few,
        "virtual": simplified_dict_virtual,
        "ssvi_param": best_param.tolist(),
    }

    return final_result
