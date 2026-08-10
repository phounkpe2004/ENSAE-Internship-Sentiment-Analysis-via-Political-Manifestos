import numpy as np
import pymc as pm
import arviz as az


def run_sampling(
    model: pm.Model,
    beta_init: np.ndarray,
    draws: int = 100,
    tune: int = 10,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    

    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            initvals={"beta": beta_init},
            target_accept=target_accept,
            random_seed=random_seed,
        )
    return idata


def check_convergence(idata: az.InferenceData, r_hat_threshold: float = 1.01, verbose: bool = True) -> None:
    
    n_divergences = int(idata.sample_stats["diverging"].sum())
    n_total = idata.sample_stats["diverging"].size
    print(f"Divergences : {n_divergences} / {n_total} ({100 * n_divergences / n_total:.2f}%)")

    summary = az.summary(idata, var_names=["theta_0", "sigma_theta"]).astype("float")
    bad_rhat = summary[summary["r_hat"] > r_hat_threshold]
    if len(bad_rhat) > 0:
        print(f"{len(bad_rhat)} paramètres avec R-hat > {r_hat_threshold} :")
        if verbose:
            print(bad_rhat)
    else:
        print(f"Tous les R-hat vérifiés sont ≤ {r_hat_threshold}")
