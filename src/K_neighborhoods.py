from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import check_X_y
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import TransformerMixin, BaseEstimator, ClusterMixin
from sklearn.utils import check_X_y



def knn_cross_validation(X, y, n_folds, n_neighbors_list):
    """Génère les couples nombre de voisins et précisions correspondantes."""

    # Conversion en tableau numpy si on fournit des DataFrame par exemple
    X, y = check_X_y(X, y)

    def models_accuracies(train_index, val_index, n_neighbors_list):
        """Précision de tous les modèles pour un jeu de données fixé."""

        # Création de `X_train`, `y_train`, `X_val` et `y_val`
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_val = X[val_index, :]
        y_val = y[val_index]

        # Calcul des précisions pour chaque nombre de voisins présent
        # dans `n_neighbors`
        n = len(train_index)
        for n_neighbors in n_neighbors_list:
            yield (
                n_neighbors,
                accuracy(X_train, y_train, X_val, y_val, n_neighbors),
                n / n_neighbors
            )

    # Définition de `n_folds` jeu de données avec `KFold`
    kf = KFold(n_splits=n_folds, shuffle=True).split(X)

    # Calcul et retour des précisions avec `models_accuracies` pour
    # chaque jeu de données défini par `KFold`.
    for train_index, test_index in kf:
        yield from models_accuracies(train_index, test_index, n_neighbors_list)


def knn_cross_validation2(X, y, n_folds, n_neighbors_list):
    n = (n_folds - 1) / n_folds * len(y)
    for n_neighbors in n_neighbors_list:
        cls = KNeighborsClassifier(n_neighbors=n_neighbors)
        for err in cross_val_score(cls, X, y, cv=n_folds):
            yield (n_neighbors, err, n / n_neighbors)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.utils import check_X_y


def accuracy(X_train, y_train, X_val, y_val, n_neighbors):
    """Précision d'un modèle Knn pour un jeu de données
    d'apprentissage et de validation fournis."""

    # Définition, apprentissage et prédiction par la méthode des
    # plus proches voisins avec `n_neighbors` voisins
    cls = KNeighborsClassifier(n_neighbors=n_neighbors)
    cls.fit(X_train, y_train)
    pred = cls.predict(X_val)

    # Calcul de la précision avec `accuracy_score`
    acc = accuracy_score(pred, y_val)

    return acc


def knn_simple_validation(X_train, y_train, X_val, y_val, n_neighbors_list):
    """Génère les couples nombres de voisins et précision
    correspondante sur l'ensemble de validation."""

    # Calcul des précisions pour tous les nombres de voisins présents
    # dans `n_neighbors_list`
    n = X_train.shape[0]
    for n_neighbors in n_neighbors_list:
        yield (
            n_neighbors,
            accuracy(X_train, y_train, X_val, y_val, n_neighbors),
            n / n_neighbors
        )


def knn_multiple_validation(X, y, n_splits, train_size, n_neighbors_list):
    """Génère les couples nombre de voisins et précisions correspondantes."""

    # Conversion en tableau numpy si on fournit des DataFrame par exemple
    X, y = check_X_y(X, y)

    def models_accuracies(train_index, val_index, n_neighbors_list):
        """Précision de tous les modèles pour un jeu de données fixé."""

        # Création de `X_train`, `y_train`, `X_val` et `y_val`
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_val = X[val_index, :]
        y_val = y[val_index]

        # Calcul des précisions pour chaque nombre de voisins présent
        # dans `n_neighbors`
        n = len(train_index)
        for n_neighbors in n_neighbors_list:
            yield (
                n_neighbors,
                accuracy(X_train, y_train, X_val, y_val, n_neighbors),
                n / n_neighbors
            )

    # Définition de `n_splits` jeu de données avec `ShuffleSplit`
    ms = ShuffleSplit(n_splits=n_splits, train_size=train_size).split(X)

    # Calcul et retour des précisions avec `models_accuracies` pour
    # chaque jeu de données défini par `ShuffleSplit`.
    for train_index, test_index in ms:
        yield from models_accuracies(train_index, test_index, n_neighbors_list)

class NearestPrototypes(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self, n_prototypes_list=[3, 3], n_neighbors=5):
        # Définir une assertion pour contrôler que `n_prototypes_list`
        # et `n_neighbors` ont des valeurs cohérentes.
        assert(sum(n_prototypes_list) >= n_neighbors)

        self.n_prototypes_list = n_prototypes_list
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # Validation des entrées
        X, y = check_X_y(X, y)

        labels = np.unique(y)
        self.classes_ = labels
        assert(len(labels) == len(self.n_prototypes_list))
        assert(len(y) >= sum(self.n_prototypes_list))

        def prototypes(X, label, n_prototypes):
            """Sélectionne les individus d'étiquette `label` dans `X` et lance un
            algorithme des k-means pour calculer `n_prototypes`
            prototypes.
            """

            # Sélection du jeu de données d'étiquette `label`
            Xk = X[y == label, :]

            # Création d'un objet de classe `KMeans` avec le bon nombre
            # de prototypes
            cls = KMeans(n_clusters=n_prototypes)

            # Apprentissage des prototypes
            cls.fit(Xk)

            return cls.cluster_centers_

        # Concaténation de tous les prototypes pour toutes les
        # étiquettes et le nombre de prototypes correspondants.
        self.prototypes_ = np.concatenate([
            prototypes(X, label, n_prototypes)
            for n_prototypes, label in zip(self.n_prototypes_list, labels)
        ])


        # Création des étiquettes pour tous les prototypes construits
        # précédemment. On pourra utiliser `np.repeat`.
        self.labels_ = np.repeat(labels, self.n_prototypes_list)


        # Création d'un objet KNeighborsClassifier
        self.nearest_prototypes_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)


        # Apprentissage du Knn sur les prototypes et leur étiquette
        self.nearest_prototypes_.fit(self.prototypes_, self.labels_)

    def predict(self, X):
        # Prédire les étiquettes en utilisant self.nearest_prototypes_
        return self.nearest_prototypes_.predict(X)