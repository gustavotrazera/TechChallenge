import pandas as pd
import missingno as msno

dados = pd.read_csv("../TechChallenge/insurance.csv")
print(dados.head())


msno.matrix(dados)

print(dados.isnull().sum())
