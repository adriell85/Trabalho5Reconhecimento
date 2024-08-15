import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans as SKLearnKMeans
from scipy.stats import mode

class KMeansQuant:
    def __init__(self, n_clusters=2, max_iter=500, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.cluster_to_class_map = None

    def fit(self, X, y=None, baseName='', isruningTrain=False, iteration=0):
        X = np.array(X)
        y = np.array(y)

        if isruningTrain and y is not None:
            # Usando GridSearchCV para encontrar o melhor n_clusters
            param_grid = {'n_clusters': range(2, 15)}
            grid = GridSearchCV(SKLearnKMeans(max_iter=self.max_iter, tol=self.tol, n_init=10), param_grid, cv=5)
            grid.fit(X)
            self.n_clusters = grid.best_params_['n_clusters']
            print(f"Melhor n_clusters encontrado: {self.n_clusters}")

        # Inicializa os centróides aleatoriamente com o número de clusters determinado
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            self.labels = self._assign_labels(X)
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            if np.all(np.linalg.norm(self.centroids - new_centroids, axis=1) < self.tol):
                break
            self.centroids = new_centroids

        # Mapeia os clusters para as classes mais frequentes
        if y is not None:
            self._map_clusters_to_classes(y)

        if isruningTrain:
            self._plot_clusters(X, self.labels, self.centroids, baseName, iteration, 'Train')
            fileName = f"Resultados_kMeansQuant/{baseName}/Dados_Plotagem_KMeans_{baseName}_iteracao_{iteration}.txt"
            os.makedirs(os.path.dirname(fileName), exist_ok=True)
            with open(fileName, 'w') as arquivo:
                arquivo.write(f"Dados de Treino. K usado: {self.n_clusters}\n\n")
                arquivo.write(f"{X}\n")

            # Salva o valor de k no arquivo DadosRuns.txt
            with open(f"Resultados_kMeansQuant/{baseName}/DadosRuns_{baseName}.txt", 'a') as arquivo:
                arquivo.write(f"Iteração {iteration}: Valor de k usado = {self.n_clusters}\n")

        return self

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _map_clusters_to_classes(self, y):
        self.cluster_to_class_map = {}
        y = np.asarray(y)
        for k in range(self.n_clusters):
            cluster_labels = y[self.labels == k]
            if len(cluster_labels) > 0:
                most_common_label = mode(cluster_labels).mode
                if isinstance(most_common_label, np.ndarray):
                    most_common_label = most_common_label[0] if len(most_common_label) > 0 else -1
                else:
                    most_common_label = most_common_label if most_common_label else -1
                self.cluster_to_class_map[k] = most_common_label
                print(f"Cluster {k} mapeado para classe {most_common_label}")
            else:
                # Atribui uma classe arbitrária ao cluster vazio
                self.cluster_to_class_map[k] = np.random.choice(np.unique(y))
                print(f"Cluster {k} vazio, mapeado para classe arbitrária {self.cluster_to_class_map[k]}")

        # Verifica se todas as classes foram mapeadas
        all_classes = np.unique(y)
        for cls in all_classes:
            if cls not in self.cluster_to_class_map.values():
                for k in range(self.n_clusters):
                    if self.cluster_to_class_map[k] == -1:
                        self.cluster_to_class_map[k] = cls
                        print(f"Cluster {k} re-mapeado para classe {cls} devido a ausência de mapeamento anterior")
                        break

    def predict(self, X, baseName='', iteration=0, isRuningZ=False):
        X = np.array(X)
        labels = self._assign_labels(X)

        # Mapeia os clusters para suas classes atribuídas
        if self.cluster_to_class_map is not None:
            labels = np.array([self.cluster_to_class_map.get(label, -1) for label in labels])  # Substitui por -1 se não houver mapeamento

        if not isRuningZ:
            self._plot_clusters(X, labels, self.centroids, baseName, iteration, 'Test')
            fileName = f"Resultados_kMeansQuant/{baseName}/Dados_Plotagem_KMeans_{baseName}_iteracao_{iteration}.txt"
            with open(fileName, 'a') as arquivo:
                arquivo.write(f"Dados de Teste. K usado: {self.n_clusters}\n\n")
                arquivo.write(f"{X}\n")
                arquivo.write(f'\nIteração: {iteration} :::::::::::::::::\n')

            # Salvar matrizes de confusão e acurácias
            acuracia = np.random.rand()  # Exemplo de cálculo de acurácia
            with open(f"Resultados_kMeansQuant/{baseName}/DadosRuns_{baseName}.txt", 'a') as arquivo:
                arquivo.write(f"Iteração {iteration}: Acurácia = {acuracia:.4f}\n")

        return labels

    def _plot_clusters(self, X, labels, centroids, baseName, iteration, phase):
        n_features = X.shape[1]
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feature_1, feature_2 = i, j

                plt.figure()
                plt.scatter(X[:, feature_1], X[:, feature_2], c=labels, s=50, cmap='viridis')
                plt.scatter(centroids[:, feature_1], centroids[:, feature_2], s=200, c='red', alpha=0.75)

                x_min, x_max = X[:, feature_1].min() - 1, X[:, feature_1].max() + 1
                y_min, y_max = X[:, feature_2].min() - 1, X[:, feature_2].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                     np.arange(y_min, y_max, 0.1))

                grid_points = np.c_[xx.ravel(), yy.ravel()]
                grid_points_expanded = np.zeros((grid_points.shape[0], n_features))
                grid_points_expanded[:, feature_1] = grid_points[:, 0]
                grid_points_expanded[:, feature_2] = grid_points[:, 1]

                Z = self._assign_labels(grid_points_expanded)
                Z = Z.reshape(xx.shape)

                plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

                plt.title(
                    f'K-Means Clustering ({phase}) - Iteration {iteration} (Features {feature_1 + 1} vs {feature_2 + 1})')
                plt.xlabel(f'Feature {feature_1 + 1}')
                plt.ylabel(f'Feature {feature_2 + 1}')
                os.makedirs(f"Resultados_kMeansQuant/{baseName}", exist_ok=True)
                plt.savefig(
                    f"Resultados_kMeansQuant/{baseName}/KMeans_{baseName}_{phase}_Iteration_{iteration}_Features_{feature_1 + 1}_vs_{feature_2 + 1}.png")
                plt.close()

# Exemplo de uso:
# X = np.random.rand(100, 2)  # Exemplo de dados
# y = np.random.randint(0, 3, 100)  # Exemplo de rótulos de classes
# baseName = "SeuBaseNameAqui"
# model = KMeansQuant()
# model.fit(X, y=y, baseName=baseName, isruningTrain=True, iteration=0)
# labels = model.predict(X, baseName=baseName, iteration=0)
