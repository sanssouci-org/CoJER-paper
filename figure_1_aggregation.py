#%%
import numpy as np
import matplotlib.pyplot as plt

delta = 0.1

# Charger les données depuis le fichier .npy
# data_ = np.load(f'results/coverages_openml_fcp_all_agg_methods_delta{delta}_noagg.npy')
data_ = np.load(f"results/coverages_openml_fcp_all_agg_methods_delta{delta}_reduc.npy")


# Paramètres
alpha = 0.1  # Valeur de base pour alpha
n_test = 100
alpha_corrected = alpha + (1 / n_test)  # Ajustement pour la discrétisation
colors = ['#2ca02c', '#ff7f0e', '#1f77b4']

# Noms des méthodes
method_names = ['CoJER', 'CV+', 'Bonferroni']

# Calculer les dimensions de data_
nb_datasets, nb_splits, nb_methods = data_.shape

# Initialiser une matrice pour stocker les proportions
proportions = np.zeros((nb_datasets, nb_methods))

# Calculer les proportions pour chaque méthode
for i in range(nb_methods):
    proportions[:, i] = np.sum(data_[:, :, i] >= (1 - alpha_corrected), axis=1) / nb_splits

# Créer la figure avec un figsize plus petit (4, 2.5)
fig, ax = plt.subplots(figsize=(4, 2.5))

# Générer le boxplot pour chaque méthode avec boîtes visibles, sans outliers ni remplissage
bp = ax.boxplot(
    [proportions[:, i] for i in range(nb_methods)],
    patch_artist=False,  # Désactiver le remplissage
    positions=np.arange(nb_methods),
    showfliers=False,  # Supprimer les outliers
    medianprops=dict(color='orange', linewidth=1.5),  # Lignes des médianes en orange
    boxprops=dict(color='black', linewidth=1.5),  # Boîtes en noir
    whiskerprops=dict(color='black', linewidth=1),  # Moustaches en noir
    capprops=dict(color='black', linewidth=1)  # Petits traits en noir
)

# Scatter plot pour chaque point de données
for method_idx in range(nb_methods):
    # Obtenir les points de données pour la méthode actuelle
    y = proportions[:, method_idx]
    
    # S'assurer qu'il y a des points à scatter
    if len(y) > 0:
        # Calculer les positions x pour chaque scatter point (autour du centre de chaque boxplot)
        x = np.random.normal(method_idx, 0.05, size=len(y))
        
        # Scatter plot avec les couleurs spécifiques de la méthode
        ax.scatter(x, y, alpha=0.75, marker='v', color=colors[method_idx])

# Limiter l'axe des ordonnées entre 0.3 et 1.05
ax.set_ylim(0.3, 1.03)

# Ajouter une ligne rouge en pointillés au niveau 1 - alpha de base (sans correction de discrétisation)
ax.axhline(y=1 - alpha, color='red', linestyle='--', linewidth=1.5)

# Ajouter des labels sur les axes et un titre
ax.set_xticks(np.arange(nb_methods))
ax.set_xticklabels(method_names)  # Utiliser les noms des méthodes ['CoJER', 'Cross-Conformal', 'Bonferroni']

# Labels des axes et titre
ax.set_ylabel('FCP event control')
ax.set_title('FCP control')

# Enregistrer le graphique
# plt.savefig('results/fig_FCP_control_v2_agg_noaggkopi.pdf', bbox_inches='tight')

plt.savefig('results/fig_FCP_control_v2_agg_fixed.pdf', bbox_inches='tight')

# Afficher le graphique
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# Paramètres
datasets_id = [
    361072, 361073, 361074, 361076, 361077, 361279, 361078, 361079,
    361080, 361081, 361082, 361280, 361084, 361085,
    361086, 361087, 361088
]  # Liste des IDs de datasets
splits = range(20)  # 20 splits
methods = ['hmean', 'amean', 'intersect']  # Les méthodes disponibles
colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
delta = 0.1  # Exemple de valeur pour delta (peut être ajusté)

# Labels des méthodes (avec modification pour utiliser CV+ au lieu de Cross-Conformal)
method_names = ['CoJER', 'CV+', 'Bonferroni']

# Initialiser une liste pour stocker les longueurs d'intervalles moyennes
interval_lengths = {method: [] for method in methods}

# Charger les fichiers pour chaque dataset, split, et méthode
for dataset_id in datasets_id:
    # Stocker les moyennes d'intervalles pour ce dataset
    mean_lengths_per_dataset = {method: [] for method in methods}
    
    for spl in splits:
        for method in methods:
            # Charger les fichiers .npy pour chaque méthode en utilisant l'ID du dataset
            # file_pattern = f"results/intervals_length_{dataset_id}_spl{spl}_delta{delta}_noagg_fcp_{method}.npy"
            file_pattern = f"results/intervals_length_{dataset_id}_spl{spl}_delta{delta}_reduc_fcp_{method}.npy"
            interval_data = np.load(file_pattern)
            
            # Ajouter les données à la liste correspondante
            mean_lengths_per_dataset[method].append(interval_data)
    
    # Moyennage sur les splits pour chaque méthode
    for method in methods:
        interval_lengths[method].append(np.mean(mean_lengths_per_dataset[method]))

# Convertir les longueurs en format numpy array pour chaque méthode
mean_lengths = {method: np.array(interval_lengths[method]) for method in methods}

# Normalisation relative par rapport à la méthode ayant l'intervalle le plus court pour chaque dataset
relative_lengths = {method: [] for method in methods}
for i in range(len(datasets_id)):
    # Trouver la méthode avec l'intervalle le plus court pour ce dataset
    min_length = min(mean_lengths[method][i] for method in methods)
    
    # Calculer la longueur relative pour chaque méthode
    for method in methods:
        relative_lengths[method].append(mean_lengths[method][i] / min_length)

# Créer la figure boxplot avec un figsize plus petit (4, 2.5)
fig, ax = plt.subplots(figsize=(4, 2.5))

# Générer le boxplot pour chaque méthode avec boîtes et médianes visibles, sans outliers ni remplissage
bp = ax.boxplot(
    [relative_lengths[method] for method in methods], 
    patch_artist=False,  # Désactiver le remplissage
    positions=np.arange(len(methods)), 
    showfliers=False,  # Supprimer les outliers
    medianprops=dict(color='orange', linewidth=1.5),  # Lignes des médianes en orange
    boxprops=dict(color='black', linewidth=1.5),  # Boîtes en noir
    whiskerprops=dict(color='black', linewidth=1),  # Moustaches en noir
    capprops=dict(color='black', linewidth=1)  # Petits traits en noir
)

# Scatter plot pour chaque point de données
for method_idx, method in enumerate(methods):
    # Obtenir les points de données pour la méthode actuelle (en relatif)
    y = relative_lengths[method]
    
    # Calculer les positions x pour chaque scatter point (autour du centre de chaque boxplot)
    x = np.random.normal(method_idx, 0.05, size=len(y))
    
    # Plot scatter avec les couleurs spécifiques de la méthode
    ax.scatter(x, y, alpha=0.75, marker='v', color=colors[method_idx], label=method_names[method_idx])

# Limiter l'axe des ordonnées entre 0.9 et 5.5
ax.set_ylim(0.9, 3.5)

# Ajouter des labels sur les axes et un titre
ax.set_xticks(np.arange(len(methods)))
ax.set_xticklabels(method_names)  # Utiliser les noms des méthodes ['CoJER', 'CV+', 'Bonferroni']

# Labels des axes et titre
ax.set_ylabel('Relative length (vs best)')
ax.set_title('Interval Lengths')

# Ajouter une légende pour les méthodes
ax.legend(loc='upper right')

# Enregistrer la figure au format PDF
# plt.savefig('results/fig_relative_interval_lengths_v2_agg_nogg_kopi.pdf', bbox_inches='tight')

plt.savefig('results/fig_relative_interval_lengths_v2_agg_fixed.pdf', bbox_inches='tight')
# Afficher le graphique
plt.show()




# %%
