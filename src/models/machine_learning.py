import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('../../data/processed/country_vaccinations_processed.csv')

mediana_diaria = df['daily_vaccinations_per_million'].median()
df['is_fast_vax'] = (df['daily_vaccinations_per_million'] > mediana_diaria).astype(int)

X = df[['total_vaccinations_per_hundred', 'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred']].fillna(0)
y = df['is_fast_vax']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
pipe_knn = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])

grid_lr = GridSearchCV(pipe_lr, {'lr__C': [0.1, 1.0, 10.0]}, cv=5, scoring='f1')
grid_knn = GridSearchCV(pipe_knn, {'knn__n_neighbors': [3, 5, 7]}, cv=5, scoring='f1')

grid_lr.fit(X_train, y_train)
grid_knn.fit(X_train, y_train)

for nome, modelo in zip(['Regressão Logística', 'KNN'], [grid_lr, grid_knn]):
    y_pred = modelo.predict(X_test)
    print(f"\n--- {nome} ---")
    print(f"Melhores Hiperparâmetros: {modelo.best_params_}")
    print(f"Matriz de Confusão:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precisão: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall (Sensibilidade): {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")