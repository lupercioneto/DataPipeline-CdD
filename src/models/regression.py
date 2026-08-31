import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('../../data/processed/country_vaccinations_processed.csv')

df_reg = df[['total_vaccinations', 'daily_vaccinations', 'people_fully_vaccinated']].dropna()
X = df_reg[['daily_vaccinations', 'people_fully_vaccinated']]
y = df_reg['total_vaccinations']

modelo_reg = LinearRegression()
modelo_reg.fit(X, y)
y_pred = modelo_reg.predict(X)

r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"B0 (Intercept): {modelo_reg.intercept_:.2f}")
print(f"B1 (Daily Vaccinations): {modelo_reg.coef_[0]:.2f}")
print(f"B2 (People Fully Vaccinated): {modelo_reg.coef_[1]:.2f}")
print(f"R2: {r2:.4f} | RMSE: {rmse:.2f}")