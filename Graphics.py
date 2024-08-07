
def generate(dados, plt):

    dados.plot.scatter(x="age", y="charges")
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title('Relação entre Idade e Custo de plano de saúde')
    plt.show()

    categorias = ["Não fumante", "Fumante"]
    plt.pie(dados["smoker"].value_counts(), labels=categorias, autopct="%0.0f%%", explode=[0, 0.1], colors=("g", "r"))
    plt.show()

    categorias = ["Mulher", "Homem"]
    plt.pie(dados["sex"].value_counts(), labels=categorias, autopct="%0.0f%%", explode=[0, 0.1], colors=("g", "r"))
    plt.show()

    plt.hist(dados["children"])
    plt.title("Quantidade de filhos")
    plt.show()

    dados.plot.scatter(x="bmi", y="charges")
    plt.show()