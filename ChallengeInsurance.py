import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import TrainningModel
import Graphics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Lendo o arquivo de dados
dados = pd.read_csv(".venv/database/insurance.csv")
print(dados.head())

# Pré processamento da base

# Verificando se temos dados nulos no arquivo de dados
msno.matrix(dados)
print(dados.isnull().sum())

# Normalizando as colunas que eram strings em número
encoder = LabelEncoder()
dados['sex'] = encoder.fit_transform(dados['sex'])
dados['smoker'] = encoder.fit_transform(dados['smoker'])
dados['region'] = encoder.fit_transform(dados['region'])

# Gerando gráficos com os dados
Graphics.generate(dados, plt, pd)

# Normalizando os dados utilizando min max scaler
x = dados.drop(columns=["charges"])
y = dados[["charges"]]

scaler = MinMaxScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)

# Treinamento
TrainningModel.trainModel(x_scaled, y, plt)





