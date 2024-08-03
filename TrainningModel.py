from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def trainModel(dados, plt):

    x = dados.drop(columns=['sex'])
    y = dados[['charges']]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print("Total base de treino: ", len(x_train))
    print("Total base de teste: ", len(y_test))

    # Normalizando os dados utilizando min max scaler
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train_min_max_scaled = scaler.transform(x_train)
    x_test_min_max_scaled = scaler.transform(x_test)

    # Usando linear Regression
    modelo_classificador = LinearRegression()
    modelo_classificador.fit(x_train_min_max_scaled, y_train)
    previsoes = modelo_classificador.predict(x_test_min_max_scaled)

    # Avaliando o desempenho do modelo
    erro_medio_quadratico = mean_squared_error(y_test, previsoes)
    erro_absoluto_medio = mean_absolute_error(y_test, previsoes)
    r_quadrado = r2_score(y_test, previsoes)

    print(f'Erro Médio Quadrático: {erro_medio_quadratico}')
    print(f'Erro Absoluto Médio: {erro_absoluto_medio}')
    print(f'R² (coeficiente de determinação): {r_quadrado}')

    # Visualizando as previsões
    plt.scatter(x_test_min_max_scaled, y_test, label='Real')
    plt.scatter(x_test_min_max_scaled, previsoes, label='Previsto', color='red')
    plt.xlabel('Age ')
    plt.ylabel('Charges')
    plt.title('Previsões do Modelo de Regressão Linear')
    plt.legend()
    plt.show()