# ============================================================
# 014_KNN.py
# Classificação utilizando K-Nearest Neighbors (KNN)
# ============================================================

from pathlib import Path
from utilidades import ler_arquivo

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

RANDOM_STATE = 42

EXTENSOES_PERMITIDAS = [
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet"
]

# Valores padrão do KNeighborsClassifier
DEFAULT_N_NEIGHBORS = 5
DEFAULT_WEIGHTS = "uniform"
DEFAULT_ALGORITHM = "auto"
DEFAULT_LEAF_SIZE = 30
DEFAULT_P = 2
DEFAULT_METRIC = "minkowski"


# ============================================================
# CAMINHOS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent
PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"

PASTA_SAIDA.mkdir(exist_ok=True)

# ============================================================
# SELEÇÃO DE ARQUIVO
# ============================================================

def selecionar_arquivo(mensagem="Selecione o arquivo:"):
    """Exibe os arquivos disponíveis na pasta ENTRADA."""

    arquivos = sorted(
        [
            arquivo
            for arquivo in PASTA_ENTRADA.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in EXTENSOES_PERMITIDAS
        ]
    )

    if not arquivos:

        print(
            "\nNenhum arquivo compatível foi encontrado "
            "na pasta ENTRADA."
        )

        return None

    print("\n" + "=" * 60)
    print(mensagem)
    print("=" * 60)

    for i, arquivo in enumerate(arquivos, start=1):

        print(
            f"{i} - {arquivo.name}"
        )

    while True:

        try:

            opcao = int(
                input(
                    "\nDigite o número do arquivo: "
                )
            )

            if 1 <= opcao <= len(arquivos):

                return arquivos[opcao - 1]

            print("Opção inválida.")

        except ValueError:

            print(
                "Digite apenas um número."
            )


# ============================================================
# SELEÇÃO DE COLUNA
# ============================================================

def selecionar_coluna(
    df,
    mensagem="Selecione uma coluna:"
):
    """Permite selecionar uma coluna pelo número."""

    print("\n" + "=" * 60)
    print(mensagem)
    print("=" * 60)

    for i, coluna in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i} - {coluna} ({df[coluna].dtype})"
        )

    while True:

        try:

            opcao = int(
                input(
                    "\nDigite o número da coluna: "
                )
            )

            if 1 <= opcao <= len(df.columns):

                return df.columns[opcao - 1]

            print("Opção inválida.")

        except ValueError:

            print(
                "Digite apenas um número."
            )


# ============================================================
# CONFIGURAÇÃO: N_NEIGHBORS
# ============================================================

def configurar_n_neighbors():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: n_neighbors")
    print("=" * 60)

    print(
        f"\nValor atual/padrão: {DEFAULT_N_NEIGHBORS}"
    )

    print("\nValores comuns:")

    print("1 - 3")
    print("2 - 5")
    print("3 - 7")
    print("4 - 9")
    print("5 - Outro valor")

    while True:

        opcao = input(
            "\nEscolha uma opção: "
        ).strip()

        valores = {
            "1": 3,
            "2": 5,
            "3": 7,
            "4": 9
        }

        if opcao in valores:

            valor = valores[opcao]

            print(
                f"\nn_neighbors definido como: {valor}"
            )

            return valor

        elif opcao == "5":

            while True:

                try:

                    valor = int(
                        input(
                            "\nDigite o número de vizinhos: "
                        )
                    )

                    if valor >= 1:

                        print(
                            f"\nn_neighbors definido como: "
                            f"{valor}"
                        )

                        return valor

                    print(
                        "O valor deve ser maior ou igual a 1."
                    )

                except ValueError:

                    print(
                        "Digite um número inteiro."
                    )

        else:

            print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO: WEIGHTS
# ============================================================

def configurar_weights():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: weights")
    print("=" * 60)

    print(
        f"\nValor atual/padrão: '{DEFAULT_WEIGHTS}'"
    )

    print("\nOpções disponíveis:")

    print(
        "1 - uniform "
        "(todos os vizinhos possuem o mesmo peso)"
    )

    print(
        "2 - distance "
        "(vizinhos mais próximos possuem maior peso)"
    )

    while True:

        opcao = input(
            "\nEscolha uma opção: "
        ).strip()

        if opcao == "1":

            return "uniform"

        elif opcao == "2":

            return "distance"

        else:

            print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO: ALGORITHM
# ============================================================

def configurar_algorithm():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: algorithm")
    print("=" * 60)

    print(
        f"\nValor atual/padrão: '{DEFAULT_ALGORITHM}'"
    )

    print("\nOpções disponíveis:")

    print(
        "1 - auto "
        "(escolha automática)"
    )

    print(
        "2 - ball_tree "
        "(utiliza Ball Tree)"
    )

    print(
        "3 - kd_tree "
        "(utiliza KD Tree)"
    )

    print(
        "4 - brute "
        "(busca por força bruta)"
    )

    while True:

        opcao = input(
            "\nEscolha uma opção: "
        ).strip()

        valores = {
            "1": "auto",
            "2": "ball_tree",
            "3": "kd_tree",
            "4": "brute"
        }

        if opcao in valores:

            valor = valores[opcao]

            print(
                f"\nalgorithm definido como: {valor}"
            )

            return valor

        print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO: LEAF_SIZE
# ============================================================

def configurar_leaf_size():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: leaf_size")
    print("=" * 60)

    print(
        f"\nValor atual/padrão: {DEFAULT_LEAF_SIZE}"
    )

    print("\nValores comuns:")

    print("1 - 10")
    print("2 - 20")
    print("3 - 30")
    print("4 - 50")
    print("5 - Outro valor")

    while True:

        opcao = input(
            "\nEscolha uma opção: "
        ).strip()

        valores = {
            "1": 10,
            "2": 20,
            "3": 30,
            "4": 50
        }

        if opcao in valores:

            valor = valores[opcao]

            print(
                f"\nleaf_size definido como: {valor}"
            )

            return valor

        elif opcao == "5":

            while True:

                try:

                    valor = int(
                        input(
                            "\nDigite o tamanho da folha: "
                        )
                    )

                    if valor >= 1:

                        print(
                            f"\nleaf_size definido como: "
                            f"{valor}"
                        )

                        return valor

                    print(
                        "O valor deve ser maior ou igual a 1."
                    )

                except ValueError:

                    print(
                        "Digite um número inteiro."
                    )

        else:

            print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO: P
# ============================================================

def configurar_p():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: p")
    print("=" * 60)

    print(
        f"\nValor atual/padrão: {DEFAULT_P}"
    )

    print(
        "\np controla a potência utilizada na "
        "distância Minkowski."
    )

    print("\nValores comuns:")

    print(
        "1 - 1 "
        "(distância Manhattan)"
    )

    print(
        "2 - 2 "
        "(distância Euclidiana)"
    )

    print(
        "3 - 3"
    )

    print(
        "4 - Outro valor"
    )

    while True:

        opcao = input(
            "\nEscolha uma opção: "
        ).strip()

        if opcao == "1":

            return 1

        elif opcao == "2":

            return 2

        elif opcao == "3":

            return 3

        elif opcao == "4":

            while True:

                try:

                    valor = float(
                        input(
                            "\nDigite um valor maior ou "
                            "igual a 1: "
                        )
                    )

                    if valor >= 1:

                        return valor

                    print(
                        "O valor deve ser maior ou igual a 1."
                    )

                except ValueError:

                    print(
                        "Digite um número válido."
                    )

        else:

            print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO: METRIC
# ============================================================

def configurar_metric():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: metric")
    print("=" * 60)

    print(
        f"\nValor atual/padrão: '{DEFAULT_METRIC}'"
    )

    print("\nOpções comuns:")

    print(
        "1 - minkowski "
        "(distância Minkowski)"
    )

    print(
        "2 - euclidean "
        "(distância Euclidiana)"
    )

    print(
        "3 - manhattan "
        "(distância Manhattan)"
    )

    print(
        "4 - chebyshev "
        "(distância Chebyshev)"
    )

    print(
        "5 - Outro valor"
    )

    while True:

        opcao = input(
            "\nEscolha uma opção: "
        ).strip()

        valores = {
            "1": "minkowski",
            "2": "euclidean",
            "3": "manhattan",
            "4": "chebyshev"
        }

        if opcao in valores:

            valor = valores[opcao]

            print(
                f"\nmetric definido como: {valor}"
            )

            return valor

        elif opcao == "5":

            valor = input(
                "\nDigite o nome da métrica: "
            ).strip()

            if valor:

                return valor

            print(
                "A métrica não pode ficar vazia."
            )

        else:

            print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

def configurar_modelo():

    n_neighbors = DEFAULT_N_NEIGHBORS
    weights = DEFAULT_WEIGHTS
    algorithm = DEFAULT_ALGORITHM
    leaf_size = DEFAULT_LEAF_SIZE
    p = DEFAULT_P
    metric = DEFAULT_METRIC

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DO KNN")
    print("=" * 60)

    print("\nModelo: KNeighborsClassifier")

    print("\nParâmetros atuais:")

    print(
        f"n_neighbors = {n_neighbors}"
    )

    print(
        f"weights     = '{weights}'"
    )

    print(
        f"algorithm   = '{algorithm}'"
    )

    print(
        f"leaf_size   = {leaf_size}"
    )

    print(
        f"p           = {p}"
    )

    print(
        f"metric      = '{metric}'"
    )

    print("\nDeseja utilizar essa configuração?")

    print(
        "1 - Sim, utilizar padrão"
    )

    print(
        "2 - Não, modificar parâmetros"
    )

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao == "1":

            return {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "algorithm": algorithm,
                "leaf_size": leaf_size,
                "p": p,
                "metric": metric
            }

        elif opcao == "2":

            break

        else:

            print("Opção inválida.")

    while True:

        print("\n" + "=" * 60)
        print("CONFIGURAÇÃO MANUAL")
        print("=" * 60)

        print("\nParâmetros atuais:")

        print(
            f"1 - n_neighbors = {n_neighbors}"
        )

        print(
            f"2 - weights     = '{weights}'"
        )

        print(
            f"3 - algorithm   = '{algorithm}'"
        )

        print(
            f"4 - leaf_size   = {leaf_size}"
        )

        print(
            f"5 - p           = {p}"
        )

        print(
            f"6 - metric      = '{metric}'"
        )

        print(
            "7 - Finalizar configuração"
        )

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao == "1":

            n_neighbors = configurar_n_neighbors()

        elif opcao == "2":

            weights = configurar_weights()

        elif opcao == "3":

            algorithm = configurar_algorithm()

        elif opcao == "4":

            leaf_size = configurar_leaf_size()

        elif opcao == "5":

            p = configurar_p()

        elif opcao == "6":

            metric = configurar_metric()

        elif opcao == "7":

            break

        else:

            print("Opção inválida.")

    # --------------------------------------------------------
    # VALIDAÇÃO DA RELAÇÃO ENTRE P E METRIC
    # --------------------------------------------------------

    if metric != "minkowski":

        print("\n" + "=" * 60)
        print("ATENÇÃO")
        print("=" * 60)

        print(
            "\nO parâmetro p é utilizado diretamente pela "
            "métrica Minkowski."
        )

        print(
            f"Métrica escolhida: '{metric}'"
        )

        print(
            f"p configurado: {p}"
        )

        print(
            "\nO valor de p será mantido na configuração, "
            "mas pode não ter efeito para a métrica escolhida."
        )

    # --------------------------------------------------------
    # RESUMO
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO FINAL")
    print("=" * 60)

    print(
        f"\nn_neighbors = {n_neighbors}"
    )

    print(
        f"weights     = '{weights}'"
    )

    print(
        f"algorithm   = '{algorithm}'"
    )

    print(
        f"leaf_size   = {leaf_size}"
    )

    print(
        f"p           = {p}"
    )

    print(
        f"metric      = '{metric}'"
    )

    return {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "algorithm": algorithm,
        "leaf_size": leaf_size,
        "p": p,
        "metric": metric
    }


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def criar_modelo(configuracao):

    return KNeighborsClassifier(
        n_neighbors=configuracao["n_neighbors"],
        weights=configuracao["weights"],
        algorithm=configuracao["algorithm"],
        leaf_size=configuracao["leaf_size"],
        p=configuracao["p"],
        metric=configuracao["metric"]
    )


# ============================================================
# VALIDAÇÃO DO DATASET
# ============================================================

def validar_dataset(
    df,
    coluna_alvo
):

    problemas = []

    # --------------------------------------------------------
    # VALORES AUSENTES NO ALVO
    # --------------------------------------------------------

    if df[coluna_alvo].isna().any():

        problemas.append(
            f"A coluna alvo '{coluna_alvo}' "
            "possui valores ausentes."
        )

    # --------------------------------------------------------
    # PREDITORES
    # --------------------------------------------------------

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    # --------------------------------------------------------
    # VALORES AUSENTES NOS PREDITORES
    # --------------------------------------------------------

    colunas_com_nan = [
        coluna
        for coluna in colunas_preditoras
        if df[coluna].isna().any()
    ]

    if colunas_com_nan:

        problemas.append(
            "Existem valores ausentes nas variáveis "
            "preditoras: "
            + ", ".join(colunas_com_nan)
        )

    # --------------------------------------------------------
    # PREDITORES NÃO NUMÉRICOS
    # --------------------------------------------------------

    colunas_nao_numericas = []

    for coluna in colunas_preditoras:

        if not pd.api.types.is_numeric_dtype(
            df[coluna]
        ):

            colunas_nao_numericas.append(
                coluna
            )

    if colunas_nao_numericas:

        problemas.append(
            "As seguintes variáveis preditoras não são "
            "numéricas: "
            + ", ".join(colunas_nao_numericas)
            + ". O programa não realiza codificação "
              "automática das variáveis preditoras."
        )

    # --------------------------------------------------------
    # QUANTIDADE DE CLASSES
    # --------------------------------------------------------

    quantidade_classes = (
        df[coluna_alvo].nunique()
    )

    if quantidade_classes < 2:

        problemas.append(
            "A variável alvo precisa possuir pelo menos "
            "2 classes."
        )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    if problemas:

        print("\n" + "=" * 60)
        print("ERROS DE VALIDAÇÃO")
        print("=" * 60)

        for problema in problemas:

            print(
                f"\n- {problema}"
            )

        return False

    return True


# ============================================================
# PREPARAÇÃO DA VARIÁVEL ALVO
# ============================================================

def preparar_alvo(
    df,
    coluna_alvo
):

    serie = df[coluna_alvo]

    # --------------------------------------------------------
    # ALVO NUMÉRICO
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(
        serie
    ):

        print(
            "\nA variável alvo já é numérica."
        )

        return (
            serie.copy(),
            None
        )

    # --------------------------------------------------------
    # ALVO CATEGÓRICO
    # --------------------------------------------------------

    print(
        "\nA variável alvo é categórica."
    )

    categorias = sorted(
        serie.unique(),
        key=lambda x: str(x).lower()
    )

    mapeamento = {
        categoria: indice
        for indice, categoria
        in enumerate(categorias)
    }

    print(
        "\nMapeamento realizado:"
    )

    for categoria, numero in mapeamento.items():

        print(
            f"  {categoria} -> {numero}"
        )

    serie_numerica = serie.map(
        mapeamento
    )

    return (
        serie_numerica,
        mapeamento
    )


# ============================================================
# PREPARAÇÃO DOS PREDITORES
# ============================================================

def preparar_preditores(
    df,
    coluna_alvo
):

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    return df[
        colunas_preditoras
    ].copy()


# ============================================================
# MÉTRICAS
# ============================================================

def calcular_metricas(
    y_real,
    y_pred
):

    return {
        "accuracy": accuracy_score(
            y_real,
            y_pred
        ),

        "precision": precision_score(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "recall": recall_score(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "f1": f1_score(
            y_real,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


# ============================================================
# AVALIAÇÃO TEXTUAL
# ============================================================

def gerar_avaliacao_textual(
    y_real,
    y_pred
):

    metricas = calcular_metricas(
        y_real,
        y_pred
    )

    matriz = confusion_matrix(
        y_real,
        y_pred
    )

    relatorio = classification_report(
        y_real,
        y_pred,
        zero_division=0
    )

    texto = ""

    texto += "MÉTRICAS\n"

    texto += "-" * 60 + "\n"

    texto += (
        f"Acurácia : "
        f"{metricas['accuracy']:.4f}\n"
    )

    texto += (
        f"Precisão : "
        f"{metricas['precision']:.4f}\n"
    )

    texto += (
        f"Recall   : "
        f"{metricas['recall']:.4f}\n"
    )

    texto += (
        f"F1-score : "
        f"{metricas['f1']:.4f}\n"
    )

    texto += (
        "\nMATRIZ DE CONFUSÃO\n"
    )

    texto += "-" * 60 + "\n"

    texto += str(matriz)

    texto += (
        "\n\nCLASSIFICATION REPORT\n"
    )

    texto += "-" * 60 + "\n"

    texto += relatorio

    return (
        texto,
        metricas
    )


# ============================================================
# AUTO TESTE
# ============================================================

def executar_auto_teste(
    X,
    y,
    configuracao
):

    print("\n" + "=" * 60)
    print("EXECUTANDO AUTO TESTE")
    print("=" * 60)

    # --------------------------------------------------------
    # VALIDAÇÃO DO NÚMERO DE VIZINHOS
    # --------------------------------------------------------

    if configuracao["n_neighbors"] > len(X):

        print(
            "\nErro: n_neighbors não pode ser maior "
            "que a quantidade de registros disponíveis."
        )

        return None

    modelo = criar_modelo(
        configuracao
    )

    modelo.fit(
        X,
        y
    )

    previsoes = modelo.predict(
        X
    )

    texto_avaliacao, _ = (
        gerar_avaliacao_textual(
            y,
            previsoes
        )
    )

    texto = ""

    texto += (
        "MÉTODO: AUTO TESTE\n"
    )

    texto += "=" * 60 + "\n\n"

    texto += (
        "O modelo foi treinado e testado "
        "utilizando o mesmo conjunto de dados.\n"
    )

    texto += (
        "ATENÇÃO: as métricas tendem a ser "
        "otimistas, pois os dados utilizados "
        "no teste também participaram do treinamento.\n\n"
    )

    texto += texto_avaliacao

    print(
        "\nAuto teste concluído."
    )

    return texto


# ============================================================
# HOLD OUT
# ============================================================

def executar_holdout(
    X_treino,
    y_treino,
    X_teste,
    y_teste,
    configuracao
):

    print("\n" + "=" * 60)
    print("EXECUTANDO HOLD OUT")
    print("=" * 60)

    if configuracao["n_neighbors"] > len(
        X_treino
    ):

        print(
            "\nErro: n_neighbors não pode ser maior "
            "que a quantidade de registros do conjunto "
            "de treinamento."
        )

        return None

    modelo = criar_modelo(
        configuracao
    )

    modelo.fit(
        X_treino,
        y_treino
    )

    previsoes = modelo.predict(
        X_teste
    )

    texto_avaliacao, _ = (
        gerar_avaliacao_textual(
            y_teste,
            previsoes
        )
    )

    texto = ""

    texto += (
        "MÉTODO: HOLD OUT\n"
    )

    texto += "=" * 60 + "\n\n"

    texto += (
        f"Quantidade de dados de treinamento: "
        f"{len(X_treino)}\n"
    )

    texto += (
        f"Quantidade de dados de teste: "
        f"{len(X_teste)}\n\n"
    )

    texto += texto_avaliacao

    print(
        "\nHold Out concluído."
    )

    return texto


# ============================================================
# CROSS-VALIDATION
# ============================================================

def executar_cross_validation(
    X,
    y,
    quantidade_folds,
    configuracao
):

    print("\n" + "=" * 60)
    print("EXECUTANDO CROSS-VALIDATION")
    print("=" * 60)

    cv = StratifiedKFold(
        n_splits=quantidade_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados = []

    texto = ""

    texto += (
        "MÉTODO: CROSS-VALIDATION\n"
    )

    texto += "=" * 60 + "\n\n"

    for numero_fold, (
        indices_treino,
        indices_teste
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(
            f"\nExecutando Fold "
            f"{numero_fold}/"
            f"{quantidade_folds}..."
        )

        X_treino = X.iloc[
            indices_treino
        ]

        X_teste = X.iloc[
            indices_teste
        ]

        y_treino = y.iloc[
            indices_treino
        ]

        y_teste = y.iloc[
            indices_teste
        ]

        # ----------------------------------------------------
        # VALIDAÇÃO DO NÚMERO DE VIZINHOS
        # ----------------------------------------------------

        if configuracao["n_neighbors"] > len(
            X_treino
        ):

            print(
                "\nErro: n_neighbors é maior que "
                "a quantidade de registros disponíveis "
                "no treinamento deste fold."
            )

            return None

        # ----------------------------------------------------
        # NOVO MODELO PARA CADA FOLD
        # ----------------------------------------------------

        modelo = criar_modelo(
            configuracao
        )

        modelo.fit(
            X_treino,
            y_treino
        )

        previsoes = modelo.predict(
            X_teste
        )

        texto_avaliacao, metricas = (
            gerar_avaliacao_textual(
                y_teste,
                previsoes
            )
        )

        resultados.append(
            metricas
        )

        texto += (
            f"\nFOLD {numero_fold}\n"
        )

        texto += (
            "-" * 60
            + "\n"
        )

        texto += texto_avaliacao

    # --------------------------------------------------------
    # MÉDIAS
    # --------------------------------------------------------

    accuracy = [
        resultado["accuracy"]
        for resultado in resultados
    ]

    precision = [
        resultado["precision"]
        for resultado in resultados
    ]

    recall = [
        resultado["recall"]
        for resultado in resultados
    ]

    f1 = [
        resultado["f1"]
        for resultado in resultados
    ]

    texto += "\n\n"

    texto += (
        "RESULTADO AGREGADO\n"
    )

    texto += "=" * 60 + "\n\n"

    texto += (
        f"Acurácia média : "
        f"{np.mean(accuracy):.4f}\n"
    )

    texto += (
        f"Acurácia desvio: "
        f"{np.std(accuracy):.4f}\n\n"
    )

    texto += (
        f"Precisão média : "
        f"{np.mean(precision):.4f}\n"
    )

    texto += (
        f"Precisão desvio: "
        f"{np.std(precision):.4f}\n\n"
    )

    texto += (
        f"Recall médio   : "
        f"{np.mean(recall):.4f}\n"
    )

    texto += (
        f"Recall desvio  : "
        f"{np.std(recall):.4f}\n\n"
    )

    texto += (
        f"F1 médio       : "
        f"{np.mean(f1):.4f}\n"
    )

    texto += (
        f"F1 desvio      : "
        f"{np.std(f1):.4f}\n"
    )

    print(
        "\nCross-validation concluída."
    )

    return texto


# ============================================================
# SELEÇÃO DO MÉTODO DE AVALIAÇÃO
# ============================================================

def selecionar_metodo():

    print("\n" + "=" * 60)
    print("MÉTODO DE AVALIAÇÃO")
    print("=" * 60)

    print(
        "\n1 - Auto Teste"
    )

    print(
        "2 - Hold Out"
    )

    print(
        "3 - Cross-validation"
    )

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao in ["1", "2", "3"]:

            return opcao

        print(
            "Opção inválida."
        )


# ============================================================
# CONFIGURAÇÃO DO HOLD OUT
# ============================================================

def configurar_holdout():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DO HOLD OUT")
    print("=" * 60)

    print(
        "\nOs dados de treino e teste já estão separados?"
    )

    print(
        "1 - Sim"
    )

    print(
        "2 - Não"
    )

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao in ["1", "2"]:

            return opcao

        print(
            "Opção inválida."
        )


# ============================================================
# PERCENTUAL DE TREINAMENTO
# ============================================================

def solicitar_percentual_treino():

    while True:

        entrada = input(
            "\nDigite o percentual destinado ao "
            "treinamento (ex.: 80): "
        ).strip()

        try:

            percentual = float(
                entrada
            )

            if 1 < percentual < 100:

                return percentual

            print(
                "Digite um percentual maior que 1 "
                "e menor que 100."
            )

        except ValueError:

            print(
                "Digite um número válido."
            )


# ============================================================
# QUANTIDADE DE FOLDS
# ============================================================

def solicitar_folds(y):

    quantidade_minima = (
        y.value_counts().min()
    )

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DA CROSS-VALIDATION")
    print("=" * 60)

    print(
        f"\nA classe com menor quantidade de "
        f"registros possui {quantidade_minima} registros."
    )

    print(
        "O número de folds não pode ser maior "
        "que essa quantidade."
    )

    while True:

        try:

            folds = int(
                input(
                    "\nDigite o número de folds: "
                )
            )

            if (
                2 <= folds
                <= quantidade_minima
            ):

                return folds

            print(
                f"Digite um valor entre 2 e "
                f"{quantidade_minima}."
            )

        except ValueError:

            print(
                "Digite um número inteiro."
            )


# ============================================================
# RELATÓRIO
# ============================================================

def criar_relatorio(
    arquivo,
    coluna_alvo,
    metodo,
    configuracao,
    texto_resultado,
    mapeamento_alvo=None
):

    caminho_saida = (
        PASTA_SAIDA
        / f"014_KNN_{arquivo.stem}.txt"
    )

    with open(
        caminho_saida,
        "w",
        encoding="utf-8"
    ) as relatorio:

        relatorio.write(
            "RELATÓRIO - KNN\n"
        )

        relatorio.write(
            "=" * 60 + "\n\n"
        )

        relatorio.write(
            f"Arquivo: {arquivo.name}\n"
        )

        relatorio.write(
            f"Variável alvo: {coluna_alvo}\n"
        )

        relatorio.write(
            f"Método de avaliação: {metodo}\n\n"
        )

        # ----------------------------------------------------
        # CONFIGURAÇÃO DO MODELO
        # ----------------------------------------------------

        relatorio.write(
            "CONFIGURAÇÃO DO MODELO\n"
        )

        relatorio.write(
            "-" * 60 + "\n"
        )

        relatorio.write(
            "Modelo: KNeighborsClassifier\n"
        )

        relatorio.write(
            f"n_neighbors: "
            f"{configuracao['n_neighbors']}\n"
        )

        relatorio.write(
            f"weights: "
            f"{configuracao['weights']}\n"
        )

        relatorio.write(
            f"algorithm: "
            f"{configuracao['algorithm']}\n"
        )

        relatorio.write(
            f"leaf_size: "
            f"{configuracao['leaf_size']}\n"
        )

        relatorio.write(
            f"p: "
            f"{configuracao['p']}\n"
        )

        relatorio.write(
            f"metric: "
            f"{configuracao['metric']}\n"
        )

        relatorio.write("\n")

        # ----------------------------------------------------
        # MAPEAMENTO DO ALVO
        # ----------------------------------------------------

        if mapeamento_alvo is not None:

            relatorio.write(
                "MAPEAMENTO DA VARIÁVEL ALVO\n"
            )

            relatorio.write(
                "-" * 60 + "\n"
            )

            for categoria, numero in (
                mapeamento_alvo.items()
            ):

                relatorio.write(
                    f"{categoria} -> {numero}\n"
                )

            relatorio.write("\n")

        # ----------------------------------------------------
        # RESULTADOS
        # ----------------------------------------------------

        if texto_resultado is not None:

            relatorio.write(
                texto_resultado
            )

    return caminho_saida


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("KNN - CLASSIFICAÇÃO")
    print("=" * 60)

    # --------------------------------------------------------
    # MÉTODO DE AVALIAÇÃO
    # --------------------------------------------------------

    metodo = selecionar_metodo()

    # --------------------------------------------------------
    # DATASET DE TREINO
    # --------------------------------------------------------

    arquivo_treino = selecionar_arquivo(
        "Selecione o dataset:"
    )

    if arquivo_treino is None:

        return

    df_treino = ler_arquivo(
        arquivo_treino
    )

    if df_treino is None:

        return

    if df_treino.empty:

        print(
            "\nO dataset está vazio."
        )

        return

    # --------------------------------------------------------
    # HOLD OUT
    # --------------------------------------------------------

    arquivo_teste = None
    df_teste = None

    if metodo == "2":

        opcao_holdout = configurar_holdout()

        if opcao_holdout == "1":

            arquivo_teste = selecionar_arquivo(
                "Selecione o dataset de teste:"
            )

            if arquivo_teste is None:

                return

            df_teste = ler_arquivo(
                arquivo_teste
            )

            if df_teste is None:

                return

            if df_teste.empty:

                print(
                    "\nO dataset de teste está vazio."
                )

                return

    # --------------------------------------------------------
    # VARIÁVEL ALVO
    # --------------------------------------------------------

    coluna_alvo = selecionar_coluna(
        df_treino,
        "Selecione a variável alvo:"
    )

    # --------------------------------------------------------
    # VALIDAÇÃO DO TREINO
    # --------------------------------------------------------

    if not validar_dataset(
        df_treino,
        coluna_alvo
    ):

        return

    # --------------------------------------------------------
    # VALIDAÇÃO DO TESTE
    # --------------------------------------------------------

    if df_teste is not None:

        if coluna_alvo not in df_teste.columns:

            print(
                f"\nErro: a variável alvo "
                f"'{coluna_alvo}' não existe "
                "no dataset de teste."
            )

            return

        colunas_treino = set(
            df_treino.columns
        )

        colunas_teste = set(
            df_teste.columns
        )

        if colunas_treino != colunas_teste:

            print(
                "\nErro: os datasets de treino "
                "e teste não possuem as mesmas colunas."
            )

            apenas_treino = (
                colunas_treino
                - colunas_teste
            )

            apenas_teste = (
                colunas_teste
                - colunas_treino
            )

            if apenas_treino:

                print(
                    "\nColunas existentes apenas "
                    "no treino:"
                )

                for coluna in apenas_treino:

                    print(
                        f"- {coluna}"
                    )

            if apenas_teste:

                print(
                    "\nColunas existentes apenas "
                    "no teste:"
                )

                for coluna in apenas_teste:

                    print(
                        f"- {coluna}"
                    )

            return

        if not validar_dataset(
            df_teste,
            coluna_alvo
        ):

            return

    # --------------------------------------------------------
    # PREPARAÇÃO DO ALVO
    # --------------------------------------------------------

    y_treino, mapeamento_alvo = (
        preparar_alvo(
            df_treino,
            coluna_alvo
        )
    )

    # --------------------------------------------------------
    # PREDITORES
    # --------------------------------------------------------

    X_treino = preparar_preditores(
        df_treino,
        coluna_alvo
    )

    # --------------------------------------------------------
    # CONFIGURAÇÃO DO MODELO
    # --------------------------------------------------------

    configuracao = configurar_modelo()

    # --------------------------------------------------------
    # AUTO TESTE
    # --------------------------------------------------------

    if metodo == "1":

        texto_resultado = (
            executar_auto_teste(
                X_treino,
                y_treino,
                configuracao
            )
        )

        nome_metodo = "Auto Teste"

    # --------------------------------------------------------
    # HOLD OUT
    # --------------------------------------------------------

    elif metodo == "2":

        if df_teste is not None:

            # -----------------------------------------------
            # DATASETS SEPARADOS
            # -----------------------------------------------

            if mapeamento_alvo is not None:

                y_teste = (
                    df_teste[coluna_alvo]
                    .map(mapeamento_alvo)
                )

                if y_teste.isna().any():

                    print(
                        "\nErro: o dataset de teste "
                        "possui categorias na variável "
                        "alvo que não existem no dataset "
                        "de treinamento."
                    )

                    return

            else:

                y_teste = (
                    df_teste[coluna_alvo]
                    .copy()
                )

            X_teste = preparar_preditores(
                df_teste,
                coluna_alvo
            )

            texto_resultado = (
                executar_holdout(
                    X_treino,
                    y_treino,
                    X_teste,
                    y_teste,
                    configuracao
                )
            )

        else:

            # -----------------------------------------------
            # DIVISÃO AUTOMÁTICA
            # -----------------------------------------------

            percentual_treino = (
                solicitar_percentual_treino()
            )

            tamanho_teste = (
                1 - percentual_treino / 100
            )

            (
                X_treino_dividido,
                X_teste,
                y_treino_dividido,
                y_teste
            ) = train_test_split(
                X_treino,
                y_treino,
                test_size=tamanho_teste,
                random_state=RANDOM_STATE,
                stratify=y_treino
            )

            texto_resultado = (
                executar_holdout(
                    X_treino_dividido,
                    y_treino_dividido,
                    X_teste,
                    y_teste,
                    configuracao
                )
            )

        nome_metodo = "Hold Out"

    # --------------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------------

    else:

        quantidade_folds = (
            solicitar_folds(
                y_treino
            )
        )

        texto_resultado = (
            executar_cross_validation(
                X_treino,
                y_treino,
                quantidade_folds,
                configuracao
            )
        )

        nome_metodo = (
            f"Cross-validation "
            f"({quantidade_folds} folds)"
        )

    # --------------------------------------------------------
    # ERRO DURANTE A EXECUÇÃO
    # --------------------------------------------------------

    if texto_resultado is None:

        print(
            "\nO processamento não pôde ser concluído."
        )

        return

    # --------------------------------------------------------
    # SALVAMENTO
    # --------------------------------------------------------

    caminho_saida = criar_relatorio(
        arquivo_treino,
        coluna_alvo,
        nome_metodo,
        configuracao,
        texto_resultado,
        mapeamento_alvo
    )

    # --------------------------------------------------------
    # FINALIZAÇÃO
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("=" * 60)

    print(
        f"\nRelatório salvo em:\n"
        f"{caminho_saida}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()