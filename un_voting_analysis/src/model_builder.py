import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

# Ancres d'identification -- reprises du script R original.
# Sans elles, theta et beta ne sont identifiés qu'au signe près
# (label switching possible entre chaînes).
ANCHOR_COUNTRIES = {
    "USA": 3.0,
    "GBR": 3.0,
    "ISR": 3.0,
    "RUS": -2.0,
}
ANCHOR_SIGMA = 0.1  # prior très resserré pour ces pays uniquement
DEFAULT_SIGMA = 1.0


def build_theta0_priors(country_to_idx: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit les vecteurs mu / sigma du prior sur theta_0 (position à la
    première session observée), avec des ancres fortes sur certains pays
    pour fixer le signe et l'échelle de l'axe latent.
    """
    num_countries = len(country_to_idx)
    mu = np.zeros(num_countries)
    sigma = np.full(num_countries, DEFAULT_SIGMA)

    for code, anchor_value in ANCHOR_COUNTRIES.items():
        if code in country_to_idx:
            idx = country_to_idx[code]
            mu[idx] = anchor_value
            sigma[idx] = ANCHOR_SIGMA

    return mu, sigma


def build_beta_initial_values(
    votes: pd.DataFrame,
    rcid_to_idx: dict,
    num_resolutions: int,
    us_code: str = "USA",
    uk_code: str = "GBR",
) -> np.ndarray:
    """
    Valeurs initiales de beta selon Bailey et al. :
    signe donné par le vote US (ou UK si US absent sur cette résolution).
    """
    vote_to_beta = {1: 1, 2: 0, 3: -1}  # Yes, Abstain, No

    beta_init = np.zeros(num_resolutions)

    us_votes = votes.loc[votes["country_code"] == us_code, ["rcid", "vote"]]
    us_resolution_ids = set()

    for rcid, vote in zip(us_votes["rcid"], us_votes["vote"]):
        idx = rcid_to_idx[rcid]
        beta_init[idx] = vote_to_beta[vote]
        us_resolution_ids.add(rcid)

    missing_us = set(rcid_to_idx.keys()) - us_resolution_ids

    uk_votes = votes.loc[votes["country_code"] == uk_code, ["rcid", "vote"]]
    uk_dict = dict(zip(uk_votes["rcid"], uk_votes["vote"]))

    for rcid in missing_us:
        idx = rcid_to_idx[rcid]
        beta_init[idx] = vote_to_beta[uk_dict[rcid]] if rcid in uk_dict else 0

    return beta_init


def build_model(
    country_idx: np.ndarray,
    session_idx: np.ndarray,
    resolution_idx: np.ndarray,
    observed_vote: np.ndarray,
    num_countries: int,
    num_sessions: int,
    num_resolutions: int,
    country_to_idx: dict,
) -> pm.Model:

    theta0_mu, theta0_sigma = build_theta0_priors(country_to_idx)

    with pm.Model() as model:

        sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.5)

        theta_0 = pm.Normal(
            "theta_0",
            mu=theta0_mu,
            sigma=theta0_sigma,
            shape=num_countries,
        )

        theta_innov = pm.Normal(
            "theta_innov",
            mu=0,
            sigma=sigma_theta,
            shape=(num_countries, num_sessions - 1),
        )

        theta = pm.Deterministic(
            "theta",
            pt.concatenate(
                [theta_0[:, None], theta_0[:, None] + pt.cumsum(theta_innov, axis=1)],
                axis=1,
            ),
        )

        beta = pm.Normal("beta", sigma=1, shape=num_resolutions)

        cutpoint_1 = pm.Normal("cutpoint_1", mu=0, sigma=0.3, shape=num_resolutions)
        cutpoint_distance = pm.HalfNormal("cutpoint_distance", sigma=0.2, shape=num_resolutions)
        cutpoint_2 = pm.Deterministic("cutpoint_2", cutpoint_1 + cutpoint_distance)

        theta_obs = theta[country_idx, session_idx]
        beta_obs = beta[resolution_idx]
        eta = beta_obs * theta_obs

        cutpoints_obs = pt.stack(
            [cutpoint_1[resolution_idx], cutpoint_2[resolution_idx]], axis=1
        )

        pm.OrderedProbit(
            "vote_obs",
            eta=eta,
            cutpoints=cutpoints_obs,
            observed=observed_vote,
        )

    return model