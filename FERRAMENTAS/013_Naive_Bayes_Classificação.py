# ============================================================
# 013_Naive_Bayes.py
# Classificação utilizando Gaussian Naive Bayes
# ============================================================

from pathlib import Path
from utilidades import ler_arquivo
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
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

DEFAULT_VAR_SMOOTHING = 1e-9
DEFAULT_PRIORS = None


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
        print("\nNenhum arquivo compatível foi encontrado na pasta ENTRADA.")
        return None

    print("\n" + "=" * 60)
    print(mensagem)
    print("=" * 60)

    for i, arquivo in enumerate(arquivos, start=1):
        print(f"{i} - {arquivo.name}")

    while True:
        try:
            opcao = int(input("\nDigite o número do arquivo: "))

            if 1 <= opcao <= len(arquivos):
                return arquivos[opcao - 1]

            print("Opção inválida.")

        except ValueError:
            print("Digite apenas um número.")


# ============================================================
# SELEÇÃO DE COLUNA
# ============================================================

def selecionar_coluna(df, mensagem="Selecione uma coluna:"):
    """Permite selecionar uma coluna pelo número."""

    print("\n" + "=" * 60)
    print(mensagem)
    print("=" * 60)

    for i, coluna in enumerate(df.columns, start=1):
        print(f"{i} - {coluna} ({df[coluna].dtype})")

    while True:
        try:
            opcao = int(input("\nDigite o número da coluna: "))

            if 1 <= opcao <= len(df.columns):
                return df.columns[opcao - 1]

            print("Opção inválida.")

        except ValueError:
            print("Digite apenas um número.")


# ============================================================
# CONFIGURAÇÃO DO VAR_SMOOTHING
# ============================================================

def configurar_var_smoothing():
    """Configura o parâmetro var_smoothing."""

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: var_smoothing")
    print("=" * 60)

    print("\nValor atual/padrão: 1e-09")

    print("\nValores comuns:")
    print("1 - 1e-08")
    print("2 - 1e-07")
    print("3 - 1e-06")
    print("4 - 1e-05")
    print("5 - Outro valor")

    while True:
        opcao = input("\nEscolha uma opção: ").strip()

        valores = {
            "1": 1e-8,
            "2": 1e-7,
            "3": 1e-6,
            "4": 1e-5
        }

        if opcao in valores:
            valor = valores[opcao]

            print(f"\nvar_smoothing definido como: {valor}")
            return valor

        elif opcao == "5":
            while True:
                entrada = input(
                    "\nDigite um valor positivo para var_smoothing: "
                ).strip()

                try:
                    valor = float(entrada)

                    if valor > 0:
                        print(
                            f"\nvar_smoothing definido como: {valor}"
                        )
                        return valor

                    print("O valor deve ser maior que zero.")

                except ValueError:
                    print("Digite um número válido.")

        else:
            print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO DOS PRIORS
# ============================================================

def configurar_priors(classes):
    """Configura manualmente as probabilidades a priori das classes."""

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO: priors")
    print("=" * 60)

    print("\nValor atual/padrão: None")

    print(
        "\nQuando priors = None, o modelo calcula automaticamente "
        "as probabilidades das classes com base nos dados de treinamento."
    )

    print("\nClasses encontradas:")

    for i, classe in enumerate(classes, start=1):
        print(f"{i} - {classe}")

    print("\nDeseja definir as probabilidades manualmente?")
    print("1 - Não, utilizar None")
    print("2 - Sim, definir manualmente")

    while True:
        opcao = input("\nDigite a opção: ").strip()

        if opcao == "1":
            return None

        elif opcao == "2":
            break

        else:
            print("Opção inválida.")

    print("\n" + "-" * 60)
    print("Definição das probabilidades")
    print("-" * 60)

    print(
        "\nA soma das probabilidades deve ser igual a 1.0."
    )

    while True:

        priors = []

        for classe in classes:

            while True:
                entrada = input(
                    f"\nProbabilidade da classe '{classe}': "
                ).strip()

                try:
                    probabilidade = float(entrada)

                    if 0 <= probabilidade <= 1:
                        priors.append(probabilidade)
                        break

                    print(
                        "A probabilidade deve estar entre 0 e 1."
                    )

                except ValueError:
                    print("Digite um número válido.")

        soma = sum(priors)

        print("\nProbabilidades informadas:")

        for classe, probabilidade in zip(classes, priors):
            print(f"  {classe}: {probabilidade}")

        print(f"\nSoma: {soma:.10f}")

        if np.isclose(soma, 1.0):
            print("\nProbabilidades válidas.")
            return priors

        print(
            "\nA soma das probabilidades precisa ser igual a 1.0."
        )
        print("Informe os valores novamente.")


# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

def configurar_modelo(classes):
    """
    Permite utilizar os parâmetros padrão do GaussianNB
    ou modificá-los manualmente.
    """

    var_smoothing = DEFAULT_VAR_SMOOTHING
    priors = DEFAULT_PRIORS

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DO NAIVE BAYES")
    print("=" * 60)

    print("\nModelo: GaussianNB")

    print("\nParâmetros atuais:")

    print(f"priors        = {priors}")
    print(f"var_smoothing = {var_smoothing}")

    print("\nDeseja utilizar essa configuração?")
    print("1 - Sim, utilizar padrão")
    print("2 - Não, modificar parâmetros")

    while True:
        opcao = input("\nDigite a opção: ").strip()

        if opcao == "1":
            return {
                "priors": priors,
                "var_smoothing": var_smoothing
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

        print(f"1 - var_smoothing = {var_smoothing}")
        print(f"2 - priors        = {priors}")
        print("3 - Finalizar configuração")

        opcao = input("\nDigite a opção: ").strip()

        if opcao == "1":

            var_smoothing = configurar_var_smoothing()

        elif opcao == "2":

            priors = configurar_priors(classes)

        elif opcao == "3":
            break

        else:
            print("Opção inválida.")

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO FINAL")
    print("=" * 60)

    print(f"\npriors        = {priors}")
    print(f"var_smoothing = {var_smoothing}")

    return {
        "priors": priors,
        "var_smoothing": var_smoothing
    }


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def criar_modelo(configuracao):
    """Cria uma nova instância do GaussianNB."""

    return GaussianNB(
        priors=configuracao["priors"],
        var_smoothing=configuracao["var_smoothing"]
    )


# ============================================================
# VALIDAÇÃO DO DATASET
# ============================================================

def validar_dataset(df, coluna_alvo):
    """
    Valida se o dataset possui as características necessárias
    para o Gaussian Naive Bayes.
    """

    problemas = []

    # --------------------------------------------------------
    # Valores ausentes
    # --------------------------------------------------------

    if df[coluna_alvo].isna().any():
        problemas.append(
            f"A coluna alvo '{coluna_alvo}' possui valores ausentes."
        )

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    colunas_com_nan = [
        coluna
        for coluna in colunas_preditoras
        if df[coluna].isna().any()
    ]

    if colunas_com_nan:
        problemas.append(
            "Existem valores ausentes nas variáveis preditoras: "
            + ", ".join(colunas_com_nan)
        )

    # --------------------------------------------------------
    # Verificação dos preditores
    # --------------------------------------------------------

    colunas_nao_numericas = []

    for coluna in colunas_preditoras:

        if not pd.api.types.is_numeric_dtype(df[coluna]):
            colunas_nao_numericas.append(coluna)

    if colunas_nao_numericas:
        problemas.append(
            "As seguintes variáveis preditoras não são numéricas: "
            + ", ".join(colunas_nao_numericas)
            + ". O programa não realiza codificação automática "
              "das variáveis preditoras."
        )

    # --------------------------------------------------------
    # Quantidade de classes
    # --------------------------------------------------------

    quantidade_classes = df[coluna_alvo].nunique()

    if quantidade_classes < 2:
        problemas.append(
            "A variável alvo precisa possuir pelo menos 2 classes."
        )

    if problemas:

        print("\n" + "=" * 60)
        print("ERROS DE VALIDAÇÃO")
        print("=" * 60)

        for problema in problemas:
            print(f"\n- {problema}")

        return False

    return True


# ============================================================
# PREPARAÇÃO DA VARIÁVEL ALVO
# ============================================================

def preparar_alvo(df, coluna_alvo):
    """
    Converte automaticamente uma variável alvo categórica
    para valores numéricos.
    """

    serie = df[coluna_alvo]

    if pd.api.types.is_numeric_dtype(serie):

        print(
            "\nA variável alvo já é numérica."
        )

        return serie.copy(), None

    print("\nA variável alvo é categórica.")

    categorias = sorted(
        serie.unique(),
        key=lambda x: str(x).lower()
    )

    mapeamento = {
        categoria: indice
        for indice, categoria in enumerate(categorias)
    }

    print("\nMapeamento realizado:")

    for categoria, numero in mapeamento.items():
        print(f"  {categoria} -> {numero}")

    serie_numerica = serie.map(mapeamento)

    return serie_numerica, mapeamento


# ============================================================
# PREPARAÇÃO DOS PREDITORES
# ============================================================

def preparar_preditores(df, coluna_alvo):
    """Retorna somente as variáveis preditoras numéricas."""

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    return df[colunas_preditoras].copy()


# ============================================================
# AVALIAÇÃO DO MODELO
# ============================================================

def calcular_metricas(y_real, y_pred):

    return {
        "accuracy": accuracy_score(y_real, y_pred),
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
# FORMATAÇÃO DA AVALIAÇÃO
# ============================================================

def gerar_avaliacao_textual(y_real, y_pred):

    metricas = calcular_metricas(y_real, y_pred)

    matriz = confusion_matrix(y_real, y_pred)

    relatorio = classification_report(
        y_real,
        y_pred,
        zero_division=0
    )

    texto = ""

    texto += "MÉTRICAS\n"
    texto += "-" * 60 + "\n"

    texto += (
        f"Acurácia : {metricas['accuracy']:.4f}\n"
    )

    texto += (
        f"Precisão : {metricas['precision']:.4f}\n"
    )

    texto += (
        f"Recall   : {metricas['recall']:.4f}\n"
    )

    texto += (
        f"F1-score : {metricas['f1']:.4f}\n"
    )

    texto += "\nMATRIZ DE CONFUSÃO\n"
    texto += "-" * 60 + "\n"

    texto += str(matriz)

    texto += "\n\nCLASSIFICATION REPORT\n"
    texto += "-" * 60 + "\n"

    texto += relatorio

    return texto, metricas


# ============================================================
# AUTO TESTE
# ============================================================

def executar_auto_teste(X, y, configuracao):

    print("\n" + "=" * 60)
    print("EXECUTANDO AUTO TESTE")
    print("=" * 60)

    modelo = criar_modelo(configuracao)

    modelo.fit(X, y)

    previsoes = modelo.predict(X)

    texto_avaliacao, metricas = gerar_avaliacao_textual(
        y,
        previsoes
    )

    texto = ""

    texto += "MÉTODO: AUTO TESTE\n"
    texto += "=" * 60 + "\n\n"

    texto += (
        "O modelo foi treinado e testado utilizando o mesmo conjunto "
        "de dados.\n"
    )

    texto += (
        "ATENÇÃO: as métricas tendem a ser otimistas, pois os dados "
        "utilizados no teste também participaram do treinamento.\n\n"
    )

    texto += texto_avaliacao

    print("\nAuto teste concluído.")

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

    modelo = criar_modelo(configuracao)

    modelo.fit(
        X_treino,
        y_treino
    )

    previsoes = modelo.predict(X_teste)

    texto_avaliacao, _ = gerar_avaliacao_textual(
        y_teste,
        previsoes
    )

    texto = ""

    texto += "MÉTODO: HOLD OUT\n"
    texto += "=" * 60 + "\n\n"

    texto += (
        f"Quantidade de dados de treinamento: {len(X_treino)}\n"
    )

    texto += (
        f"Quantidade de dados de teste: {len(X_teste)}\n\n"
    )

    texto += texto_avaliacao

    print("\nHold Out concluído.")

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

    texto += "MÉTODO: CROSS-VALIDATION\n"
    texto += "=" * 60 + "\n\n"

    for numero_fold, (indices_treino, indices_teste) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(
            f"\nExecutando Fold {numero_fold}/"
            f"{quantidade_folds}..."
        )

        X_treino = X.iloc[indices_treino]
        X_teste = X.iloc[indices_teste]

        y_treino = y.iloc[indices_treino]
        y_teste = y.iloc[indices_teste]

        # Cada fold recebe uma nova instância do modelo.
        modelo = criar_modelo(configuracao)

        modelo.fit(
            X_treino,
            y_treino
        )

        previsoes = modelo.predict(X_teste)

        texto_avaliacao, metricas = gerar_avaliacao_textual(
            y_teste,
            previsoes
        )

        resultados.append(metricas)

        texto += (
            f"\nFOLD {numero_fold}\n"
            + "-" * 60
            + "\n"
        )

        texto += texto_avaliacao

    # --------------------------------------------------------
    # MÉDIAS E DESVIOS
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
    texto += "RESULTADO AGREGADO\n"
    texto += "=" * 60 + "\n\n"

    texto += (
        f"Acurácia média : {np.mean(accuracy):.4f}\n"
        f"Acurácia desvio: {np.std(accuracy):.4f}\n\n"
    )

    texto += (
        f"Precisão média : {np.mean(precision):.4f}\n"
        f"Precisão desvio: {np.std(precision):.4f}\n\n"
    )

    texto += (
        f"Recall médio   : {np.mean(recall):.4f}\n"
        f"Recall desvio  : {np.std(recall):.4f}\n\n"
    )

    texto += (
        f"F1 médio       : {np.mean(f1):.4f}\n"
        f"F1 desvio      : {np.std(f1):.4f}\n"
    )

    print("\nCross-validation concluída.")

    return texto


# ============================================================
# CONFIGURAÇÃO DO MÉTODO DE AVALIAÇÃO
# ============================================================

def selecionar_metodo():

    print("\n" + "=" * 60)
    print("MÉTODO DE AVALIAÇÃO")
    print("=" * 60)

    print("\n1 - Auto Teste")
    print("2 - Hold Out")
    print("3 - Cross-validation")

    while True:

        opcao = input("\nDigite a opção: ").strip()

        if opcao in ["1", "2", "3"]:
            return opcao

        print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO DO HOLD OUT
# ============================================================

def configurar_holdout():

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DO HOLD OUT")
    print("=" * 60)

    print("\nOs dados de treino e teste já estão separados?")
    print("1 - Sim")
    print("2 - Não")

    while True:

        opcao = input("\nDigite a opção: ").strip()

        if opcao in ["1", "2"]:
            return opcao

        print("Opção inválida.")


# ============================================================
# PERCENTUAL DE TREINAMENTO
# ============================================================

def solicitar_percentual_treino():

    while True:

        entrada = input(
            "\nDigite o percentual destinado ao treinamento "
            "(ex.: 80): "
        ).strip()

        try:

            percentual = float(entrada)

            if 1 < percentual < 100:
                return percentual

            print(
                "Digite um percentual maior que 1 e menor que 100."
            )

        except ValueError:
            print("Digite um número válido.")


# ============================================================
# QUANTIDADE DE FOLDS
# ============================================================

def solicitar_folds(y):

    quantidade_minima = y.value_counts().min()

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DA CROSS-VALIDATION")
    print("=" * 60)

    print(
        f"\nA classe com menor quantidade de registros possui "
        f"{quantidade_minima} registros."
    )

    print(
        "O número de folds não pode ser maior que essa quantidade."
    )

    while True:

        try:

            folds = int(
                input("\nDigite o número de folds: ")
            )

            if 2 <= folds <= quantidade_minima:
                return folds

            print(
                f"Digite um valor entre 2 e {quantidade_minima}."
            )

        except ValueError:
            print("Digite um número inteiro.")


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
        / f"013_Naive_Bayes_{arquivo.stem}.txt"
    )

    with open(
        caminho_saida,
        "w",
        encoding="utf-8"
    ) as relatorio:

        relatorio.write(
            "RELATÓRIO - NAIVE BAYES\n"
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
            "Modelo: GaussianNB\n"
        )

        relatorio.write(
            f"priors: {configuracao['priors']}\n"
        )

        relatorio.write(
            f"var_smoothing: "
            f"{configuracao['var_smoothing']}\n"
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

            for categoria, numero in mapeamento_alvo.items():

                relatorio.write(
                    f"{categoria} -> {numero}\n"
                )

            relatorio.write("\n")

        # ----------------------------------------------------
        # RESULTADOS
        # ----------------------------------------------------

        relatorio.write(
            texto_resultado
        )

    return caminho_saida


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("NAIVE BAYES - CLASSIFICAÇÃO")
    print("=" * 60)

    # --------------------------------------------------------
    # MÉTODO DE AVALIAÇÃO
    # --------------------------------------------------------

    metodo = selecionar_metodo()

    # --------------------------------------------------------
    # CARREGAMENTO DO DATASET
    # --------------------------------------------------------

    arquivo_treino = selecionar_arquivo(
        "Selecione o dataset:"
    )

    if arquivo_treino is None:
        return

    df_treino = ler_arquivo(arquivo_treino)

    if df_treino is None:
        return

    if df_treino.empty:
        print("\nO dataset está vazio.")
        return

    # --------------------------------------------------------
    # HOLD OUT COM DATASETS SEPARADOS
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

            df_teste = ler_arquivo(arquivo_teste)

            if df_teste is None:
                return

            if df_teste.empty:
                print("\nO dataset de teste está vazio.")
                return

    # --------------------------------------------------------
    # ESCOLHA DA VARIÁVEL ALVO
    # --------------------------------------------------------

    coluna_alvo = selecionar_coluna(
        df_treino,
        "Selecione a variável alvo:"
    )

    # --------------------------------------------------------
    # VALIDAÇÃO DO DATASET DE TREINO
    # --------------------------------------------------------

    if not validar_dataset(
        df_treino,
        coluna_alvo
    ):
        return

    # --------------------------------------------------------
    # VALIDAÇÃO DO DATASET DE TESTE
    # --------------------------------------------------------

    if df_teste is not None:

        if coluna_alvo not in df_teste.columns:

            print(
                f"\nErro: a variável alvo '{coluna_alvo}' "
                "não existe no dataset de teste."
            )

            return

        colunas_treino = set(df_treino.columns)
        colunas_teste = set(df_teste.columns)

        if colunas_treino != colunas_teste:

            print(
                "\nErro: os datasets de treino e teste "
                "não possuem as mesmas colunas."
            )

            apenas_treino = colunas_treino - colunas_teste
            apenas_teste = colunas_teste - colunas_treino

            if apenas_treino:
                print(
                    "\nColunas existentes apenas no treino:"
                )

                for coluna in apenas_treino:
                    print(f"- {coluna}")

            if apenas_teste:
                print(
                    "\nColunas existentes apenas no teste:"
                )

                for coluna in apenas_teste:
                    print(f"- {coluna}")

            return

        if not validar_dataset(
            df_teste,
            coluna_alvo
        ):
            return

    # --------------------------------------------------------
    # PREPARAÇÃO DO ALVO
    # --------------------------------------------------------

    y_treino, mapeamento_alvo = preparar_alvo(
        df_treino,
        coluna_alvo
    )

    # --------------------------------------------------------
    # PREPARAÇÃO DOS PREDITORES
    # --------------------------------------------------------

    X_treino = preparar_preditores(
        df_treino,
        coluna_alvo
    )

    # --------------------------------------------------------
    # CONFIGURAÇÃO DO MODELO
    # --------------------------------------------------------

    classes = np.sort(
        y_treino.unique()
    )

    configuracao = configurar_modelo(
        classes
    )

    # --------------------------------------------------------
    # AUTO TESTE
    # --------------------------------------------------------

    if metodo == "1":

        texto_resultado = executar_auto_teste(
            X_treino,
            y_treino,
            configuracao
        )

        nome_metodo = "Auto Teste"

    # --------------------------------------------------------
    # HOLD OUT
    # --------------------------------------------------------

    elif metodo == "2":

        if df_teste is not None:

            # -----------------------------------------------
            # DATASET DE TESTE SEPARADO
            # -----------------------------------------------

            if mapeamento_alvo is not None:

                y_teste = df_teste[coluna_alvo].map(
                    mapeamento_alvo
                )

                if y_teste.isna().any():

                    print(
                        "\nErro: o dataset de teste possui "
                        "categorias na variável alvo que não "
                        "existem no dataset de treinamento."
                    )

                    return

            else:

                y_teste = df_teste[coluna_alvo].copy()

            X_teste = preparar_preditores(
                df_teste,
                coluna_alvo
            )

            texto_resultado = executar_holdout(
                X_treino,
                y_treino,
                X_teste,
                y_teste,
                configuracao
            )

        else:

            # -----------------------------------------------
            # DIVISÃO AUTOMÁTICA
            # -----------------------------------------------

            percentual_treino = solicitar_percentual_treino()

            tamanho_teste = (
                1 - percentual_treino / 100
            )

            X_treino_dividido, X_teste, \
                y_treino_dividido, y_teste = train_test_split(
                    X_treino,
                    y_treino,
                    test_size=tamanho_teste,
                    random_state=RANDOM_STATE,
                    stratify=y_treino
                )

            texto_resultado = executar_holdout(
                X_treino_dividido,
                y_treino_dividido,
                X_teste,
                y_teste,
                configuracao
            )

        nome_metodo = "Hold Out"

    # --------------------------------------------------------
    # CROSS-VALIDATION
    # --------------------------------------------------------

    else:

        quantidade_folds = solicitar_folds(
            y_treino
        )

        texto_resultado = executar_cross_validation(
            X_treino,
            y_treino,
            quantidade_folds,
            configuracao
        )

        nome_metodo = (
            f"Cross-validation ({quantidade_folds} folds)"
        )

    # --------------------------------------------------------
    # SALVAMENTO DO RELATÓRIO
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
        f"\nRelatório salvo em:\n{caminho_saida}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()