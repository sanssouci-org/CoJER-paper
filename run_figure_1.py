# %%
import numpy as np
import openml
from utils_conformal import aggregated_models_conformal, compute_residuals
from utils_conformal import get_null_pvals_conformal, get_template_conformal
import sanssouci as sa
from utils_conformal import IterLambda_Tau

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

models = [
    RandomForestRegressor(),
    LassoCV(),
    MLPRegressor(),
    SVR(),
    KNeighborsRegressor()
]

n_points_y = 100
alpha = 0.1
delta = 0.2
B = 1000
n_points_test = 500
nIter = 8
nb_splits = 30

datasets_id = [
    361072, 361073, 361074, 361076, 361077, 361279, 361078, 361079,
    361080, 361081, 361082, 361280, 361084, 361085,
    361086, 361087, 361088
]
nb_conf_methods = 3
all_coverages = np.zeros((len(datasets_id), nb_splits, nb_conf_methods, len(models)))
all_lengths = np.zeros((len(datasets_id), nb_splits, nb_conf_methods, len(models)))

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

        pval0_raw = get_null_pvals_conformal(B, n_cal, len(y_test))
        template = get_template_conformal(B, n_cal, len(y_test))

        calibrated_thr = sa.calibrate_jer(delta, template, pval0_raw, k_max=int(n_test/2))

        alpha_discretized = int(np.floor(alpha * n_test))
        alpha_hat_kopi = calibrated_thr[alpha_discretized]

        lambda_dkw_fn = IterLambda_Tau(n_cal, n_test, nIter=nIter)
        alpha_hat_ulysse = alpha - lambda_dkw_fn(delta)
        print(alpha_hat_kopi, alpha_hat_ulysse)
        full_residuals = np.zeros((len(models), n_cal))
        trained_models = []

        for i, model in enumerate(models):
            residuals, trained_model = compute_residuals(model, X_train, y_train, X_cal, y_cal)
            full_residuals[i] = residuals
            trained_models.append(trained_model)

        n_cal = len(y_cal)
        quantile_scp_list_kopi = []
        quantile_scp_list_ulysse = []
        quantile_scp_list_vanilla = []

        for m in range(len(models)):
            quantile_scp_list_kopi.append(
                np.quantile(
                    full_residuals[m], np.ceil((1 - alpha_hat_kopi) * (n_cal + 1)) / n_cal
                )
            )
            quantile_scp_list_ulysse.append(
                np.quantile(
                    full_residuals[m], np.ceil((1 - alpha_hat_ulysse) * (n_cal + 1)) / n_cal
                )
            )
            quantile_scp_list_vanilla.append(
                np.quantile(
                    full_residuals[m], np.ceil((1 - alpha) * (n_cal + 1)) / n_cal
                )
            )
        # %%

        interv_len_scp = np.zeros((nb_conf_methods, len(models)))
        coverages = np.zeros((nb_conf_methods, len(models)))
        for m in range(len(models)):
            y_pred_test = trained_models[m].predict(X_test)
            coverages[0, m] = np.sum(
                ((y_pred_test - quantile_scp_list_kopi[m]) <= y_test)
                * (y_test <= (y_pred_test + quantile_scp_list_kopi[m]))
            ) / len(y_test)
            interv_len_scp[0, m] = 2 * quantile_scp_list_kopi[m]
            coverages[1, m] = np.sum(
                ((y_pred_test - quantile_scp_list_ulysse[m]) <= y_test)
                * (y_test <= (y_pred_test + quantile_scp_list_ulysse[m]))
            ) / len(y_test)
            interv_len_scp[1, m] = 2 * quantile_scp_list_ulysse[m]

            coverages[2, m] = np.sum(
                ((y_pred_test - quantile_scp_list_vanilla[m]) <= y_test)
                * (y_test <= (y_pred_test + quantile_scp_list_vanilla[m]))
            ) / len(y_test)
            interv_len_scp[2, m] = 2 * quantile_scp_list_vanilla[m]
        all_coverages[id][spl] = coverages
        all_lengths[id][spl] = interv_len_scp

np.save("results/coverages_openml_fcp_control_multiple_splits.npy", all_coverages)
np.save("results/lengths_openml_fcp_control_mutiple_splits.npy", all_lengths)
# %%
