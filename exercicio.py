import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from scipy import stats


# as váriaveis independentes são: id, nome, genero, 
# atividades extracurriculares, aplicação de carreira
# dias que faltou, horas estudadas e aspiração de carreira

# as dependentes são as notas 

df = pd.read_csv("student-scores.csv")

X = df[['absence_days','weekly_self_study_hours']]
y = df['physics_score']

x_transformer = Pipeline(steps=[
 ('imputer', SimpleImputer(strategy='mean')),
 ('scaler', StandardScaler())
])


preprocessor = ColumnTransformer(transformers=[
 ('x', x_transformer, X.columns),
])

X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
 ('preprocessor', preprocessor),
 ('regressor', LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

residuals = y_test - y_pred

results_df = pd.DataFrame({
 'y_true': y_test,
 'y_pred': y_pred,
 'residuals': residuals
})


r2 = r2_score(y_test, y_pred)

regressor = model.named_steps['regressor']
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
coef_df = pd.DataFrame({
 'feature': feature_names,
 'coefficient': regressor.coef_
}).sort_values(by='coefficient', ascending=False)

print("\nCoeficientes do Modelo:\n")
print(coef_df)
#Coeficientes do Modelo:
#
#                     feature  coefficient
#1  x__weekly_self_study_hours     2.246976
#0             x__absence_days    -1.190915
print(f"\nIntercepto: {regressor.intercept_:.3f}") 
# Intercepto: 81.177

# relação de x e 2.246976
for col in X_test.columns:
 plt.figure(figsize=(5,4))
 sns.scatterplot(x=X_test[col], y=y_test)
 plt.xlabel(col)
 plt.ylabel('Score')
 plt.title(f'Relação entre {col} e Score')
 plt.show()

# margem de erros
plt.figure(figsize=(6,4))
plt.plot(residuals.values, marker='o', linestyle='')
plt.axhline(0, color='red', linestyle='--')
plt.title('Independência dos resíduos')
plt.xlabel('Ordem das observações')
plt.ylabel('Resíduo')
plt.show()

# q-q distribuição normal de x e y em seus percentis
plt.figure(figsize=(6,5))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Gráfico Q-Q dos resíduos (normalidade)')
plt.show()

#homoscedasticidade se ele se espalha em x
plt.figure(figsize=(6,5))
sns.scatterplot(x='y_pred', y='residuals',
data=results_df)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores ajustados (ŷ)')
plt.ylabel('Resíduos')
plt.title('Verificação de homoscedasticidade')
plt.show()

#linearidade
plt.figure(figsize=(6,5))
sns.scatterplot(x='y_pred', y='y_true', data=results_df)
plt.plot([results_df['y_pred'].min(), results_df['y_pred'].max()],
 [results_df['y_pred'].min(), results_df['y_pred'].max()],
 'r--', label='Linha ideal')
plt.xlabel('Valores previstos')
plt.ylabel('Valores reais')
plt.title('Verificação de linearidade')
plt.legend()
plt.show()

print(f"R² = {r2:.3f}")
# R² = 0.030