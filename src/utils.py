import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.manifold import MDS
from scipy.spatial.distance import cdist
import matplotlib as mpl
from scipy.stats import norm
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, MetaEstimatorMixin
import math


def plot_Shepard(mds_model, plot=True):
    """Affiche le diagramme de Shepard et retourne un couple contenant les
    dissimilarités originales et les distances apprises par le
    modèle.
    """

    assert isinstance(mds_model, MDS)

    # Inter-distances apprises
    dist = cdist(mds_model.embedding_, mds_model.embedding_)
    idxs = np.tril_indices_from(dist, k=-1)
    dist_mds = dist[idxs]

    # Inter-distances d'origine
    dist = mds_model.dissimilarity_matrix_
    dist_orig = dist[idxs]

    dists = np.column_stack((dist_orig, dist_mds))

    if plot:
        f, ax = plt.subplots()
        range = [dists.min(), dists.max()]
        ax.plot(range, range, 'r--')
        ax.scatter(*dists.T)
        ax.set_xlabel('Dissimilarités')
        ax.set_ylabel('Distances')

    return (*dists.T,)

# Taken from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    default_kwargs = dict(leaf_font_size=10)
    default_kwargs.update(kwargs or {})

    dendrogram(linkage_matrix, **default_kwargs)

def scatterplot_pca(
    columns=None, hue=None, style=None, data=None, pc1=1, pc2=2, **kwargs
):
    """
    Utilise `sns.scatterplot` en appliquant d'abord une ACP si besoin
    pour réduire la dimension.
    """

    # Select columns (should be numeric)
    data_quant = data if columns is None else data[columns]
    data_quant = data_quant.drop(
        columns=[e for e in [hue, style] if e is not None], errors="ignore"
    )

    # Reduce to two dimensions
    if data_quant.shape[1] == 2:
        data_pca = data_quant
        pca = None
    else:
        n_components = max(pc1, pc2)
        pca = PCA(n_components=n_components)
        data_pca = pca.fit_transform(data_quant)
        data_pca = pd.DataFrame(
            data_pca[:, [pc1 - 1, pc2 - 1]], columns=[f"PC{pc1}", f"PC{pc2}"]
        )

    # Keep name, force categorical data for hue and steal index to
    # avoid unwanted alignment
    if isinstance(hue, pd.Series):
        if not hue.name:
            hue.name = "hue"
        hue_name = hue.name
    elif isinstance(hue, str):
        hue_name = hue
        hue = data[hue]
    elif isinstance(hue, np.ndarray):
        hue = pd.Series(hue, name="class")
        hue_name = "class"

    hue = hue.astype("category")
    hue.index = data_pca.index
    hue.name = hue_name

    if isinstance(style, pd.Series):
        if not style.name:
            style.name = "style"
        style_name = style.name
    elif isinstance(style, str):
        style_name = style
        style = data[style]
    elif isinstance(style, np.ndarray):
        style = pd.Series(style, name="style")
        style_name = "style"

    sp_kwargs = {}
    full_data = data_pca
    if hue is not None:
        full_data = pd.concat((full_data, hue), axis=1)
        sp_kwargs["hue"] = hue_name
    if style is not None:
        full_data = pd.concat((full_data, style), axis=1)
        sp_kwargs["style"] = style_name

    x, y = data_pca.columns
    ax = sns.scatterplot(x=x, y=y, data=full_data, **sp_kwargs)

    return ax, pca


def plot_clustering(data, clus1, clus2=None, ax=None, **kwargs):
    """Affiche les données `data` dans le premier plan principal.
    """

    if ax is None:
        ax = plt.gca()

    other_kwargs = {e: kwargs.pop(e) for e in ["centers", "covars"] if e in kwargs}

    ax, pca = scatterplot_pca(data=data, hue=clus1, style=clus2, ax=ax, **kwargs)

    if "centers" in other_kwargs and "covars" in other_kwargs:
        # Hack to get colors
        # TODO use legend_out = True
        levels = [str(l) for l in np.unique(clus1)]
        hdls, labels = ax.get_legend_handles_labels()
        colors = [
            artist.get_facecolor().ravel()
            for artist, label in zip(hdls, labels)
            if label in levels
        ]
        colors = colors[: len(levels)]

        if data.shape[1] == 2:
            centers_2D = other_kwargs["centers"]
            covars_2D = other_kwargs["covars"]
        else:
            centers_2D = pca.transform(other_kwargs["centers"])
            covars_2D = [
                pca.components_ @ c @ pca.components_.T for c in other_kwargs["covars"]
            ]

        p = 0.9
        sig = norm.ppf(p ** (1 / 2))

        for covar_2D, center_2D, color in zip(covars_2D, centers_2D, colors):
            v, w = linalg.eigh(covar_2D)
            v = 2.0 * sig * np.sqrt(v)

            u = w[0] / linalg.norm(w[0])
            if u[0] == 0:
                angle = np.pi / 2
            else:
                angle = np.arctan(u[1] / u[0])

            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(center_2D, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

    return ax, pca


def add_decision_boundary(model, levels=None, resolution=1000, ax=None, label=None, color=None):
    """Trace une frontière de décision sur une figure existante.

    La fonction utilise `model` pour prédire un score ou une classe
    sur une grille de taille `resolution`x`resolution`. Une (ou
    plusieurs frontières) sont ensuite tracées d'après le paramètre
    `levels` qui fixe la valeur des lignes de niveaux recherchées.
    """
    if ax is None:
        ax = plt.gca()
    if callable(model):
        if levels is None:
            levels = [0]

        def predict(X):
            return model(X)
    else:
        n_classes = len(model.classes_)
        if n_classes == 2:
            if hasattr(model, "decision_function"):
                if levels is None:
                    levels = [0]

                def predict(X):
                    return model.decision_function(X)
            elif hasattr(model, "predict_proba"):
                if levels is None:
                    levels = [.5]

                def predict(X):
                    pred = model.predict_proba(X)
                    if pred.shape[1] > 1:
                        return pred[:, 0]
                    else:
                        return pred
            elif hasattr(model, "predict"):
                if levels is None:
                    levels = [.5]

                def predict(X):
                    return model.predict(X)
            else:
                raise Exception("Modèle pas reconnu")
        else:
            levels = np.arange(n_classes - 1) + .5

            def predict(X):
                pred = model.predict(X)
                _, idxs = np.unique(pred, return_inverse=True)
                return idxs
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = predict(xy).reshape(XX.shape)
    color = "red" if color is None else color
    sns.lineplot([0], [0], label=label, ax=ax, color=color, linestyle="dashed")
    ax.contour(
        XX,
        YY,
        Z,
        levels=levels,
        colors=[color],
        linestyles="dashed",
        antialiased=True,
    )

def inertia_factory(feat_metric, sample_metric):
    def feat_inner(u, v):
        return (u[None, :] @ feat_metric @ v[:, None]).item()

    def inertia_point(X, p=None):
        """Calcule l'inertie de `X` par rapport au point `p`. Si le point `p`
        n'est pas fourni prendre le centre de gravité.
        """

        if p is None:
            # Calcul du centre de gravité pondéré par `sample_metric`
            p = np.sum(sample_metric @ X, axis=0)

        # Linéarisation au cas où...
        p = p.squeeze()

        # Centrer par rapport à `p`
        Xp = X - p

        # Utiliser la formule du cours avec la trace.
        inertia = sum(np.diag(Xp.T @ sample_metric @ Xp @ feat_metric))

        return inertia

    def inertia_axe(X, v, p=None):
        """Calcule l'inertie de `X` par rapport à la droite pointée par `p` et
        de direction `v`. Si `p` n'est pas fourni, on prend la droite
        qui passe par l'origine (p=0).
        """

        if p is None:
            p = np.zeros(2)

        # Linéarisation au cas où...
        p = p.squeeze()

        def dist_axe_point(v, p, q):
            """Distance d'un point `q` à une droite repérée par un
            vecteur `v` et un point `p`.
            """

            # Normalisation du vecteur `v`
            v = v.ravel()
            v = v / math.sqrt(feat_inner(v, v))

            # Calculer le vecteur perpendiculaire à la droite
            # d'origine q et dont l'extrémité appartient à la droite.
            q_projq = (p - q) - feat_inner(p - q, v) * v

            # Retourner la norme de `q_projq`
            return feat_inner(q_projq, q_projq)

        # Matrice de toutes les distances au carré à la droite
        dists = [dist_axe_point(v, p, q) for q in X]

        # Somme pondérée en fonction de `sample_metric`
        inertia = sum(np.diag(sample_metric) * np.array(dists))

        return inertia

    return feat_inner, inertia_point, inertia_axe

