import numpy as np
import scipy.stats as stat
from scipy.stats import hmean, gmean
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from scipy.stats import randint
from hidimstat.knockoffs.utils import quantile_aggregation

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

def get_conditional_distrib_conformal(n_cal):
    """
    From Gazin et al. AISTATS 2023
    """
    U = stat.dirichlet.rvs(alpha=np.ones(n_cal + 1))[0]
    p_values = (1 + np.arange(n_cal + 1)) / (n_cal + 1)

    distrib_cond = stat.rv_discrete(values=(p_values, U))
    return distrib_cond


def get_null_pvals_conformal(B, n_cal, n_test):
    pval0 = np.zeros((B, n_test))
    for b in range(B):
        distrib_cond = get_conditional_distrib_conformal(n_cal)
        pval0[b] = distrib_cond.rvs(size=n_test)
    pval0 = np.sort(pval0, axis=1)
    return pval0


def get_template_conformal(B, n_cal, n_test):
    pval0 = get_null_pvals_conformal(B, n_cal, n_test)
    pval0 = np.sort(pval0, axis=0)
    return pval0


def IterLambda_Tau(n, m, nIter=8):
    def lambDKW(delta):
        lamb = 1
        t_nm = n * m / (n + m)
        A1 = np.log(1 / delta)
        A2 = np.sqrt(2 * np.pi) * 2 * t_nm / np.sqrt(n + m)
        i = -1
        while i < nIter:
            i += 1
            Num = A1 + np.log(1 + A2 * (lamb))
            lamb = np.minimum(np.sqrt(Num / (2 * t_nm)), 1)
        return lamb

    return lambDKW


def my_score(y_pred, y):
    return np.abs(y_pred - y)


def hmean_pval(x, y, models, full_residuals):
    n_cal = full_residuals.shape[1]
    ranks = np.zeros(len(models))
    for i in range(len(models)):
        residual_test = my_score(models[i].predict(x), y)
        ranks[i] = np.searchsorted(full_residuals[i], residual_test)
    pvals = (n_cal + 1 - ranks) / (1 + n_cal)
    return hmean(pvals)


def adaptive_conformal_KOPI(X_test, model, full_residuals, alpha, n_points_y=100):
    y_points = np.random.normal(scale=4, size=n_points_y)
    func_values = np.zeros((len(X_test), n_points_y))
    for i in range(len(X_test)):
        for j in range(n_points_y):
            func_values[i][j] = hmean_pval(
                [X_test[i]], y_points[j], model, full_residuals
            )

    confidence_intervals = np.zeros((len(X_test), 2))

    for i in range(len(X_test)):
        f_bisect = interp1d(
            y_points, func_values[i] - alpha, kind="linear", fill_value="extrapolate"
        )

        try:
            confidence_intervals[i][0] = bisect(
                f_bisect, np.min(y_points), np.min(np.abs(y_points))
            )

            confidence_intervals[i][1] = bisect(
                f_bisect, np.min(np.abs(y_points)), np.max(y_points)
            )
        except ValueError:
            print("No root found, trivial interval returned")
            confidence_intervals[i][0] = np.min(y_points)
            confidence_intervals[i][1] = np.max(y_points)

    return confidence_intervals


def aggregated_models_conformal(
    X_test, X_train, y_train, X_cal, y_cal, alpha, n_points_y, models
):
    n_cal = len(y_cal)
    nb_models = len(models)

    l_alpha = quantile_null_hmean(alpha, n_cal, nb_models)

    full_residuals = np.zeros((len(models), n_cal))
    trained_models = []

    for i, model in enumerate(models):
        residuals, trained_model = compute_residuals(model, X_train, y_train, X_cal, y_cal)
        full_residuals[i] = residuals
        trained_models.append(trained_model)

    y_samples = np.linspace(np.min(y_cal), np.max(y_cal), n_points_y)
    ranks = np.zeros((nb_models, n_points_y))

    confidence_intervals = np.zeros((len(X_test), 2))

    for k in tqdm(range(len(X_test))):
        for i, model in enumerate(models):
            for j, y_sample in enumerate(y_samples):
                res_current = np.abs(y_sample - model.predict([X_test[k]]))
                ranks[i, j] = np.searchsorted(np.sort(full_residuals[i]), res_current)

        pvals = (n_cal + 1 - ranks) / (1 + n_cal)
        pvals_hmean = hmean(pvals)

        confidence_intervals[k] = find_interval(pvals_hmean, l_alpha, y_samples)

    return confidence_intervals, full_residuals, trained_models


def compute_residuals(model, X_train, y_train, X_cal, y_cal):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_cal)
    residuals = np.abs(y_cal - y_pred)
    return residuals, model


def find_interval(pvals_hmean, alpha, y_samples):
    current_conf = y_samples[pvals_hmean > alpha]
    if len(current_conf) > 1:
        return np.min(current_conf), np.max(current_conf)
    else:
        max_pvals = np.max(pvals_hmean)
        current_conf = y_samples[pvals_hmean > (max_pvals / 5)]
        if len(current_conf) > 1:
            return np.min(current_conf), np.max(current_conf)
        else:
            print("Trivial interval returned")
            return np.min(y_samples), np.max(y_samples)


def quantile_null_hmean(alpha, n_cal, K, nb_samples=1000):
    samples = randint.rvs(1, n_cal+1, size=(nb_samples, K)) / (n_cal + 1)
    res_hmean = hmean(samples, axis=1)
    l_alpha = np.quantile(res_hmean, alpha)
    return l_alpha


def degrade_dataset(X, num_fake_features, sigma=0.8, random_seed=None):
    """
    Degrade a dataset by adding fake features correlated with the original ones.
    
    Parameters:
    - X: numpy array, the original dataset (n_samples x n_features)
    - num_fake_features: int, the number of fake features to add
    - sigma: float, correlation strength between original and fake features (0 to 1)
    - random_seed: int, random seed for reproducibility
    
    Returns:
    - X_degraded: numpy array, the degraded dataset with additional fake features
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples, n_features = X.shape
    fake_features = np.zeros((n_samples, num_fake_features))
    
    # Generate fake features
    for i in range(num_fake_features):
        # Randomly select an original feature to correlate with
        feature_idx = np.random.randint(n_features)
        noise = np.random.normal(scale=(1 - sigma), size=n_samples)
        fake_features[:, i] = sigma * X[:, feature_idx] + noise
    
    # Concatenate the original features and fake features
    X_degraded = np.hstack((X, fake_features))

    scaler = StandardScaler()
    X_degraded = scaler.fit_transform(X_degraded)
    
    return X_degraded


def aggregated_models_conformal_all_methods(
    X_test, X_train, y_train, X_cal, y_cal, alpha, alpha_hat_kopi, n_points_y, models
):
    n_cal = len(y_cal)
    nb_models = len(models)

    full_residuals = np.zeros((len(models), n_cal))
    trained_models = []

    for i, model in enumerate(models):
        residuals, trained_model = compute_residuals(model, X_train, y_train, X_cal, y_cal)
        full_residuals[i] = residuals
        trained_models.append(trained_model)

    y_samples = np.linspace(np.min(y_cal), np.max(y_cal), n_points_y)
    ranks = np.zeros((nb_models, n_points_y))

    confidence_intervals = np.zeros((len(X_test), 2))
    confidence_intervals_amean = np.zeros((len(X_test), 2))

    for k in tqdm(range(len(X_test))):
        for i, model in enumerate(models):
            for j, y_sample in enumerate(y_samples):
                res_current = np.abs(y_sample - model.predict([X_test[k]]))
                ranks[i, j] = np.searchsorted(np.sort(full_residuals[i]), res_current)

        pvals = (n_cal + 1 - ranks) / (1 + n_cal)
        pvals_hmean = hmean(pvals)
        pvals_amean = 2 * np.mean(pvals, axis=0)

        confidence_intervals[k] = find_interval(pvals_hmean, alpha_hat_kopi, y_samples)
        confidence_intervals_amean[k] = find_interval(pvals_amean, alpha, y_samples)
    
    return confidence_intervals, confidence_intervals_amean, full_residuals, trained_models


def aggregated_models_conformal_rebuttal(
    X_test, X_train, y_train, X_cal, y_cal, alpha, alpha_hat_kopi, n_points_y, models
):
    n_cal = len(y_cal)
    nb_models = len(models)

    full_residuals = np.zeros((len(models), n_cal))
    trained_models = []

    for i, model in enumerate(models):
        residuals, trained_model = compute_residuals(model, X_train, y_train, X_cal, y_cal)
        full_residuals[i] = residuals
        trained_models.append(trained_model)

    y_samples = np.linspace(np.min(y_cal), np.max(y_cal), n_points_y)
    ranks = np.zeros((nb_models, n_points_y))

    confidence_intervals = np.zeros((len(X_test), 2))
    confidence_intervals_amean = np.zeros((len(X_test), 2))
    confidence_intervals_gmean = np.zeros((len(X_test), 2))
    confidence_intervals_qagg = np.zeros((len(X_test), 2))

    for k in tqdm(range(len(X_test))):
        for i, model in enumerate(models):
            for j, y_sample in enumerate(y_samples):
                res_current = np.abs(y_sample - model.predict([X_test[k]]))
                ranks[i, j] = np.searchsorted(np.sort(full_residuals[i]), res_current)

        pvals = (n_cal + 1 - ranks) / (1 + n_cal)

        pvals_hmean = hmean(pvals)
        pvals_amean = np.mean(pvals, axis=0)
        pvals_gmean = gmean(pvals)
        pvals_qagg = quantile_aggregation(pvals, gamma=0.3)

        confidence_intervals[k] = find_interval(pvals_hmean, alpha_hat_kopi, y_samples)
        confidence_intervals_amean[k] = find_interval(pvals_amean, alpha_hat_kopi, y_samples)
        confidence_intervals_gmean[k] = find_interval(pvals_gmean, alpha_hat_kopi, y_samples)
        confidence_intervals_qagg[k] = find_interval(pvals_qagg, alpha_hat_kopi, y_samples)

    return confidence_intervals, confidence_intervals_amean, confidence_intervals_gmean, confidence_intervals_qagg, full_residuals, trained_models