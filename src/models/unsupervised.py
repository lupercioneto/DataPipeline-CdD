import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv('../../data/processed/country_vaccinations_processed.csv')
colunas_numericas = ['total_vaccinations_per_hundred', 'people_vaccinated_per_hundred', 
                     'people_fully_vaccinated_per_hundred', 'daily_vaccinations_per_million']
X = df[colunas_numericas].fillna(0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variância PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"Variância PC2: {pca.explained_variance_ratio_[1]:.4f}")

# K-Means Cotovelo 
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o')
plt.title('Método do Cotovelo (Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inércia')
plt.grid(True)
plt.savefig('../../plots/curva_cotovelo_kmeans.png')
plt.show()

# Ajuste Final 
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6, edgecolors='w')
plt.title('Projeção PCA dos Dados de Vacinação (Clusters K-Means)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.savefig('../../plots/pca_clusters_kmeans.png')
plt.show()