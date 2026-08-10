import numpy as np
import pandas as pd
import arviz as az


def extract_theta_summary(
    idata: az.InferenceData,
    country_to_idx: dict,
    session_years: np.ndarray,
) -> pd.DataFrame:

    theta = idata.posterior["theta"]  # dims: chain, draw, theta_dim_0 (pays), theta_dim_1 (session)

    mean_theta = theta.mean(dim=("chain", "draw")).values
    q025 = theta.quantile(0.025, dim=("chain", "draw")).values
    q975 = theta.quantile(0.975, dim=("chain", "draw")).values

    idx_to_country = {v: k for k, v in country_to_idx.items()}
    num_countries, num_sessions = mean_theta.shape

    rows = []
    for c in range(num_countries):
        for s in range(num_sessions):
            rows.append({
                "country": idx_to_country[c],
                "session_idx": s,
                "year": session_years[s],
                "mean_theta": mean_theta[c, s],
                "q025": q025[c, s],
                "q975": q975[c, s],
            })

    return pd.DataFrame(rows)


def extract_beta_summary(idata: az.InferenceData, rcid_to_idx: dict) -> pd.DataFrame:

    beta = idata.posterior["beta"]

    mean_beta = beta.mean(dim=("chain", "draw")).values
    q025 = beta.quantile(0.025, dim=("chain", "draw")).values
    q975 = beta.quantile(0.975, dim=("chain", "draw")).values

    idx_to_rcid = {v: k for k, v in rcid_to_idx.items()}

    return pd.DataFrame({
        "rcid": [idx_to_rcid[i] for i in range(len(mean_beta))],
        "mean_beta": mean_beta,
        "q025": q025,
        "q975": q975,
    })