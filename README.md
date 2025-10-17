# Análise de Dados de Desempenho Estudantil

<p align="center">
  <img src="https://media0.giphy.com/media/v1.Y2lkPTZjMDliOTUyeDBrNDF2cjlrYXE2OTFxamsxYThjM2kzMGV1dGE1NW0xdjh3aGQ1YyZlcD12MV9naWZzX3NlYXJjaCZjdD1n/1oBwBVLGoLteCP2kyD/source.gif" alt="Model in Action">
</p>

Este projeto utiliza um modelo de regressão linear para prever as notas dos estudantes (`physics_score`) com base em variáveis independentes como **dias de ausência** (`absence_days`) e **horas de estudo semanal** (`weekly_self_study_hours`). Utilizamos técnicas de pré-processamento, como imputação de valores faltantes e normalização, para garantir a qualidade dos dados.

## Descrição do Projeto

O objetivo deste projeto é construir um modelo preditivo para estimar as **notas de física** dos alunos, com base em informações como horas de estudo, faltas, e outras variáveis relacionadas ao desempenho acadêmico.

## Visão Geral do Modelo

1. **Pré-processamento**:
   - Imputação de dados ausentes.
   - Normalização das variáveis.
   
2. **Modelo**:
   - Utilizamos uma **regressão linear** para prever as notas de física.

3. **Avaliação**:
   - Calculamos o **R² (coeficiente de determinação)** para avaliar a performance do modelo.
   - Realizamos uma análise gráfica para verificar a qualidade do modelo (resíduos, linearidade, normalidade, etc.).

## Estrutura do Código

### 1. Carregamento dos Dados

Os dados são carregados a partir de um arquivo CSV chamado `student-scores.csv`, que contém as informações dos estudantes.

```python
df = pd.read_csv("student-scores.csv")
```
2. Pré-processamento dos Dados
As variáveis independentes são transformadas utilizando um pipeline que inclui imputação de valores ausentes e normalização.

```python
x_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('x', x_transformer, X.columns),
])
```

3. Divisão dos Dados
Dividimos os dados em conjuntos de treinamento e teste utilizando a função train_test_split:

```python

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

4. Treinamento e Avaliação do Modelo
O modelo é treinado utilizando regressão linear e a performance é avaliada com o cálculo do R² e gráficos de resíduos.

```python

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
```

5. Visualizações
Diversas visualizações são geradas para verificar a qualidade do modelo:

Gráficos de resíduos

Gráfico Q-Q para normalidade dos resíduos

Verificação de homoscedasticidade

Verificação de linearidade

6. Análise Final
O R² final do modelo foi de aproximadamente 0.030, o que indica que o modelo explica apenas uma pequena variação nas notas dos alunos. Precisariamos aumentar a quantidade de váriaveis ou aplicar alguma transformação para termos dados mais padronizados e que atendam as suposições, por enquanto eles fogem do padrão e desviam de nossos gráficos.

Como Rodar o Código

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
```

Conclusão
Este projeto apresenta uma análise exploratória e a construção de um modelo de regressão linear simples para prever o desempenho dos estudantes. Com base nos resultados, o modelo não conseguiu explicar uma grande parte da variação nas notas, o que sugere que outras variáveis ou abordagens poderiam ser exploradas para melhorar as previsões.
