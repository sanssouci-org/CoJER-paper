# %%
import numpy as np
import openml
from utils_conformal import aggregated_models_conformal_all_methods

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sanssouci as sa
from joblib import Parallel, delayed
from utils_conformal import get_null_pvals_conformal, get_template_conformal
from posthoc_fmri import aggregate_list_of_matrices


models = [
    RandomForestRegressor(),
    Lasso(),
    MLPRegressor(),
    SVR(),
    KNeighborsRegressor()
]

n_points_y = 50
alpha = 0.1
delta = 0.2
B = 1000
n_points_test = 100
nIter = 8
nb_splits = 20
n_jobs = -1

datasets_id = [
    361072, 361073, 361074, 361076, 361077, 361279, 361078, 361079,
    361080, 361081, 361082, 361280, 361084, 361085,
    361086, 361087, 361088
]

nb_conf_methods = 3
all_coverages = np.zeros((len(datasets_id), nb_splits, nb_conf_methods))


for id in tqdm(range(len(datasets_id))):
    dataset = openml.tasks.get_task(datasets_id[id])

    X, y = dataset.get_X_and_y()

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    for spl in range(nb_splits):

        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.33)

        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_full, y_train_full, test_size=0.33
        )

        idx_choice_test = np.random.choice(len(X_test), n_points_test, replace=False)
        X_test = X_test[idx_choice_test]
        y_test = y_test[idx_choice_test]

        n_cal = len(y_cal)
        n_test = len(y_test)
        nb_models = len(models)

        if spl == 0:

            pval0_raw = get_null_pvals_conformal(B, n_cal, n_test)
            template = get_template_conformal(B, n_cal, n_test)

            calibrated_thr = sa.calibrate_jer(delta, template, pval0_raw, k_max=int(n_test))

            alpha_discretized = int(np.floor(alpha * n_test))
            alpha_hat_kopi = calibrated_thr[alpha_discretized]
    
        confidence_intervals, confidence_intervals_amean, full_residuals, trained_models = aggregated_models_conformal_all_methods(
            X_test, X_train, y_train, X_cal, y_cal, alpha, alpha_hat_kopi, n_points_y, models
        )
        
        alpha_corrected = alpha / nb_models
        quantile_scp_list = []

        for m in range(len(models)):
            quantile_scp_list.append(
                np.quantile(
                    full_residuals[m], np.ceil((1 - alpha_corrected) * (n_cal + 1)) / n_cal
                )
            )
        conf_intervals_all = np.zeros((nb_models, len(X_test), 2))
        for m in range(len(models)):
            y_pred_test = trained_models[m].predict(X_test)
            conf_intervals_all[m] = np.array([
                y_pred_test - quantile_scp_list[m],
                y_pred_test + quantile_scp_list[m]
            ]).T
        
        conf_intervals_intersect = np.zeros((len(X_test), 2))
        conf_intervals_intersect[:, 0] = np.max(conf_intervals_all[:, :, 0], axis=0)
        conf_intervals_intersect[:, 1] = np.min(conf_intervals_all[:, :, 1], axis=0)

        interval_lengths = confidence_intervals[:, 1] - confidence_intervals[:, 0]
        amean_interval_lengths = confidence_intervals_amean[:, 1] - confidence_intervals_amean[:, 0]
        intersect_interval_lengths = conf_intervals_intersect[:, 1] - conf_intervals_intersect[:, 0]

        np.save(f"results/intervals_length_{datasets_id[id]}_spl{spl}_delta{delta}_noagg_fcp_hmean.npy", interval_lengths)
        np.save(f"results/intervals_length_{datasets_id[id]}_spl{spl}_delta{delta}_noagg_fcp_amean.npy", amean_interval_lengths)
        np.save(f"results/intervals_length_{datasets_id[id]}_spl{spl}_delta{delta}_noagg_fcp_intersect.npy", intersect_interval_lengths)

        prop_covered = np.sum(
            (confidence_intervals[:, 0] <= y_test)
            * (y_test <= confidence_intervals[:, 1])
        ) / len(y_test)

        prop_covered_amean = np.sum(
            (confidence_intervals_amean[:, 0] <= y_test)
            * (y_test <= confidence_intervals_amean[:, 1])
        ) / len(y_test)

        prop_covered_intersect = np.sum(
            (conf_intervals_intersect[:, 0] <= y_test)
            * (y_test <= conf_intervals_intersect[:, 1])
        ) / len(y_test)

        all_coverages[id][spl] = np.array([prop_covered, prop_covered_amean, prop_covered_intersect])

np.save(f"results/coverages_openml_fcp_all_agg_methods_delta{delta}_noagg.npy", all_coverages)
# %%
