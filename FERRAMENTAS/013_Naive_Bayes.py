from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURAÇÃO DOS DIRETÓRIOS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"

PASTA_SAIDA.mkdir(exist_ok=True)


# ============================================================
# CONFIGURAÇÕES DO MODELO
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LISTAR ARQUIVOS
# ============================================================

def listar_arquivos():

    extensoes = {
        ".csv",
        ".xlsx",
        ".xls",
        ".json",
        ".parquet"
    }

    arquivos = sorted(
        [
            arquivo
            for arquivo in PASTA_ENTRADA.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in extensoes
        ]
    )

    return arquivos


# ============================================================
# SELECIONAR ARQUIVO
# ============================================================

def selecionar_arquivo():

    arquivos = listar_arquivos()

    if not arquivos:

        print(
            "\nNenhum arquivo compatível "
            "encontrado em ENTRADA."
        )

        return None

    print("\n" + "=" * 60)
    print("ARQUIVOS DISPONÍVEIS")
    print("=" * 60)

    for i, arquivo in enumerate(
        arquivos,
        start=1
    ):

        print(
            f"{i} - {arquivo.name}"
        )

    while True:

        try:

            opcao = int(
                input(
                    "\nDigite o número do arquivo: "
                ).strip()
            )

            if 1 <= opcao <= len(arquivos):

                return arquivos[opcao - 1]

            print(
                "Erro: escolha um número da lista."
            )

        except ValueError:

            print(
                "Erro: digite apenas um número."
            )


# ============================================================
# CARREGAR ARQUIVO
# ============================================================

def carregar_arquivo(caminho):

    extensao = caminho.suffix.lower()

    try:

        if extensao == ".csv":

            return pd.read_csv(caminho)

        elif extensao == ".xlsx":

            return pd.read_excel(caminho)

        elif extensao == ".xls":

            return pd.read_excel(caminho)

        elif extensao == ".json":

            return pd.read_json(caminho)

        elif extensao == ".parquet":

            return pd.read_parquet(caminho)

        else:

            print(
                "\nFormato de arquivo não suportado."
            )

            return None

    except Exception as erro:

        print(
            "\nErro ao carregar o arquivo:"
        )

        print(erro)

        return None


# ============================================================
# MOSTRAR COLUNAS
# ============================================================

def mostrar_colunas(df):

    print("\n" + "=" * 60)
    print("COLUNAS DISPONÍVEIS")
    print("=" * 60)

    for i, coluna in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i} - {coluna} "
            f"[{df[coluna].dtype}]"
        )


# ============================================================
# SELECIONAR COLUNA
# ============================================================

def selecionar_coluna(df, mensagem):

    while True:

        try:

            opcao = int(
                input(mensagem).strip()
            )

            if 1 <= opcao <= len(df.columns):

                return df.columns[opcao - 1]

            print(
                "Erro: escolha um número da lista."
            )

        except ValueError:

            print(
                "Erro: digite apenas um número."
            )


# ============================================================
# VERIFICAR SE A COLUNA É NUMÉRICA
# ============================================================

def eh_numerica(serie):

    if pd.api.types.is_bool_dtype(serie):
        return False

    return pd.api.types.is_numeric_dtype(serie)


# ============================================================
# CONVERTER VALORES DA VARIÁVEL-ALVO
# ============================================================

def preparar_variavel_alvo(serie, dicionario=None):

    serie_sem_nulos = serie.dropna()

    if len(serie_sem_nulos) == 0:

        raise ValueError(
            "A variável-alvo não possui valores válidos."
        )

    # --------------------------------------------------------
    # VARIÁVEL-ALVO NUMÉRICA
    # --------------------------------------------------------

    if eh_numerica(serie):

        resultado = pd.to_numeric(
            serie,
            errors="coerce"
        )

        return resultado, None

    # --------------------------------------------------------
    # VARIÁVEL-ALVO CATEGÓRICA
    # --------------------------------------------------------

    if dicionario is None:

        categorias = sorted(
            serie_sem_nulos.unique(),
            key=lambda x: str(x).casefold()
        )

        dicionario = {
            categoria: numero
            for numero, categoria
            in enumerate(categorias)
        }

    resultado = serie.map(dicionario)

    resultado = pd.to_numeric(
        resultado,
        errors="coerce"
    )

    return resultado, dicionario


# ============================================================
# VALIDAR VARIÁVEL-ALVO
# ============================================================

def validar_variavel_alvo(serie, nome_coluna):

    if serie.isna().any():

        quantidade = int(
            serie.isna().sum()
        )

        print("\n" + "=" * 60)
        print("ERRO - VALORES AUSENTES")
        print("=" * 60)

        print(
            f"\nA variável-alvo '{nome_coluna}' "
            f"possui {quantidade} valor(es) ausente(s)."
        )

        print(
            "\nO modelo não pode ser treinado "
            "com uma variável-alvo sem classe definida."
        )

        return False

    quantidade_classes = serie.nunique()

    if quantidade_classes < 2:

        print("\n" + "=" * 60)
        print("ERRO - CLASSES INSUFICIENTES")
        print("=" * 60)

        print(
            f"\nA variável-alvo '{nome_coluna}' "
            f"possui apenas {quantidade_classes} classe."
        )

        print(
            "\nA classificação exige pelo menos "
            "duas classes."
        )

        return False

    return True


# ============================================================
# VALIDAR VARIÁVEIS PREDITORAS
# ============================================================

def validar_preditoras(df, coluna_alvo):

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    if not colunas_preditoras:

        print("\n" + "=" * 60)
        print("ERRO - SEM VARIÁVEIS PREDITORAS")
        print("=" * 60)

        print(
            "\nO dataset não possui nenhuma coluna "
            "para utilizar no treinamento."
        )

        return False

    colunas_invalidas = []

    for coluna in colunas_preditoras:

        if not eh_numerica(df[coluna]):

            colunas_invalidas.append(
                coluna
            )

    if colunas_invalidas:

        print("\n" + "=" * 60)
        print("ERRO - COLUNAS INCOMPATÍVEIS")
        print("=" * 60)

        print(
            "\nAs seguintes colunas não são compatíveis "
            "com o treinamento:"
        )

        for coluna in colunas_invalidas:

            print(
                f"\n- {coluna}"
                f"\n  Tipo: {df[coluna].dtype}"
            )

        print(
            "\nTodas as variáveis utilizadas para "
            "treinamento devem ser numéricas."
        )

        print(
            "Apenas a variável-alvo pode ser categórica."
        )

        return False

    # --------------------------------------------------------
    # VERIFICAR VALORES AUSENTES
    # --------------------------------------------------------

    colunas_com_nulos = []

    for coluna in colunas_preditoras:

        quantidade = int(
            df[coluna].isna().sum()
        )

        if quantidade > 0:

            colunas_com_nulos.append(
                (coluna, quantidade)
            )

    if colunas_com_nulos:

        print("\n" + "=" * 60)
        print("ERRO - VALORES AUSENTES")
        print("=" * 60)

        print(
            "\nAs seguintes variáveis possuem "
            "valores ausentes:"
        )

        for coluna, quantidade in colunas_com_nulos:

            print(
                f"\n- {coluna}: "
                f"{quantidade} valor(es)"
            )

        print(
            "\nO Naive Bayes não será executado "
            "enquanto houver valores ausentes "
            "nas variáveis preditoras."
        )

        print(
            "\nUtilize anteriormente a ferramenta "
            "003_Tratamento_Valores_Ausentes.py."
        )

        return False

    return True


# ============================================================
# PREPARAR X
# ============================================================

def preparar_preditoras(df, coluna_alvo):

    colunas = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    return df[colunas].copy()


# ============================================================
# CALCULAR MÉTRICAS
# ============================================================

def calcular_metricas(y_real, y_pred):

    metricas = {

        "acuracia": accuracy_score(
            y_real,
            y_pred
        ),

        "precisao": precision_score(
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

    matriz = confusion_matrix(
        y_real,
        y_pred
    )

    relatorio = classification_report(
        y_real,
        y_pred,
        zero_division=0
    )

    return metricas, matriz, relatorio


# ============================================================
# FORMATAR MÉTRICAS
# ============================================================

def formatar_metricas(metricas):

    linhas = []

    linhas.append(
        f"Acurácia:  {metricas['acuracia']:.4f}"
    )

    linhas.append(
        f"Precisão:  {metricas['precisao']:.4f}"
    )

    linhas.append(
        f"Recall:    {metricas['recall']:.4f}"
    )

    linhas.append(
        f"F1-score:  {metricas['f1']:.4f}"
    )

    return "\n".join(linhas)


# ============================================================
# FORMATAR MATRIZ DE CONFUSÃO
# ============================================================

def formatar_matriz(matriz):

    return np.array2string(
        matriz,
        separator=" "
    )


# ============================================================
# AUTO TESTE
# ============================================================

def executar_auto_teste(
    df,
    coluna_alvo,
    y,
    mapeamento_alvo
):

    X = preparar_preditoras(
        df,
        coluna_alvo
    )

    modelo = GaussianNB()

    modelo.fit(
        X,
        y
    )

    y_pred = modelo.predict(X)

    metricas, matriz, relatorio = (
        calcular_metricas(
            y,
            y_pred
        )
    )

    linhas = []

    linhas.append(
        "MÉTODO: AUTO TESTE"
    )

    linhas.append(
        "\nO mesmo dataset foi utilizado "
        "para treinamento e teste."
    )

    linhas.append(
        "As métricas representam o desempenho "
        "sobre os próprios dados utilizados "
        "no treinamento."
    )

    linhas.append(
        "\n" + "-" * 60
    )

    linhas.append(
        "MÉTRICAS"
    )

    linhas.append(
        "-" * 60
    )

    linhas.append(
        formatar_metricas(metricas)
    )

    linhas.append(
        "\n" + "-" * 60
    )

    linhas.append(
        "MATRIZ DE CONFUSÃO"
    )

    linhas.append(
        "-" * 60
    )

    linhas.append(
        formatar_matriz(matriz)
    )

    linhas.append(
        "\n" + "-" * 60
    )

    linhas.append(
        "RELATÓRIO DE CLASSIFICAÇÃO"
    )

    linhas.append(
        "-" * 60
    )

    linhas.append(
        relatorio
    )

    return "\n".join(linhas)


# ============================================================
# HOLD OUT
# ============================================================

def executar_hold_out(
    df_treino,
    df_teste,
    coluna_alvo,
    percentual_treino=None,
    mapeamento_alvo=None
):

    # --------------------------------------------------------
    # PREPARAR VARIÁVEL-ALVO
    # --------------------------------------------------------

    y_treino, mapeamento = (
        preparar_variavel_alvo(
            df_treino[coluna_alvo],
            mapeamento_alvo
        )
    )

    # --------------------------------------------------------
    # CASO COM DATASET DE TESTE SEPARADO
    # --------------------------------------------------------

    if df_teste is not None:

        y_teste = df_teste[
            coluna_alvo
        ].map(mapeamento) if mapeamento else (
            pd.to_numeric(
                df_teste[coluna_alvo],
                errors="coerce"
            )
            if eh_numerica(
                df_treino[coluna_alvo]
            )
            else None
        )

        if y_teste is None:

            raise ValueError(
                "Não foi possível converter "
                "a variável-alvo do dataset de teste."
            )

        X_treino = preparar_preditoras(
            df_treino,
            coluna_alvo
        )

        X_teste = preparar_preditoras(
            df_teste,
            coluna_alvo
        )

        # ----------------------------------------------------
        # VERIFICAR MESMAS COLUNAS
        # ----------------------------------------------------

        if list(X_treino.columns) != list(
            X_teste.columns
        ):

            raise ValueError(
                "Os datasets de treinamento e teste "
                "não possuem as mesmas variáveis preditoras."
            )

        descricao_divisao = (
            "Dataset de treinamento e dataset "
            "de teste fornecidos separadamente."
        )

    # --------------------------------------------------------
    # GERAR TREINO E TESTE
    # --------------------------------------------------------

    else:

        X = preparar_preditoras(
            df_treino,
            coluna_alvo
        )

        if percentual_treino is None:

            raise ValueError(
                "Percentual de treinamento não informado."
            )

        X_treino, X_teste, y_treino, y_teste = (
            train_test_split(
                X,
                y_treino,
                test_size=(
                    1 - percentual_treino
                ),
                random_state=RANDOM_STATE,
                stratify=y_treino
            )
        )

        descricao_divisao = (
            f"Dataset dividido com "
            f"{percentual_treino * 100:.0f}% "
            f"para treinamento e "
            f"{(1 - percentual_treino) * 100:.0f}% "
            f"para teste."
        )

    # --------------------------------------------------------
    # VERIFICAR VALORES DO TESTE
    # --------------------------------------------------------

    if pd.Series(y_teste).isna().any():

        raise ValueError(
            "A variável-alvo do dataset de teste "
            "possui valores ausentes ou categorias "
            "que não existem no treinamento."
        )

    # --------------------------------------------------------
    # TREINAMENTO
    # --------------------------------------------------------

    modelo = GaussianNB()

    modelo.fit(
        X_treino,
        y_treino
    )

    # --------------------------------------------------------
    # PREVISÃO
    # --------------------------------------------------------

    y_pred = modelo.predict(
        X_teste
    )

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------

    metricas, matriz, relatorio = (
        calcular_metricas(
            y_teste,
            y_pred
        )
    )

    linhas = []

    linhas.append(
        "MÉTODO: HOLD OUT"
    )

    linhas.append(
        f"\n{descricao_divisao}"
    )

    linhas.append(
        f"Linhas de treinamento: "
        f"{len(X_treino)}"
    )

    linhas.append(
        f"Linhas de teste: "
        f"{len(X_teste)}"
    )

    linhas.append(
        "\n" + "-" * 60
    )

    linhas.append(
        "MÉTRICAS"
    )

    linhas.append(
        "-" * 60
    )

    linhas.append(
        formatar_metricas(metricas)
    )

    linhas.append(
        "\n" + "-" * 60
    )

    linhas.append(
        "MATRIZ DE CONFUSÃO"
    )

    linhas.append(
        "-" * 60
    )

    linhas.append(
        formatar_matriz(matriz)
    )

    linhas.append(
        "\n" + "-" * 60
    )

    linhas.append(
        "RELATÓRIO DE CLASSIFICAÇÃO"
    )

    linhas.append(
        "-" * 60
    )

    linhas.append(
        relatorio
    )

    return "\n".join(linhas)


# ============================================================
# CROSS-VALIDATION
# ============================================================

def executar_cross_validation(
    df,
    coluna_alvo,
    y,
    quantidade_folds
):

    X = preparar_preditoras(
        df,
        coluna_alvo
    )

    classes = y.unique()

    menor_classe = (
        y.value_counts().min()
    )

    if quantidade_folds > menor_classe:

        raise ValueError(
            "A quantidade de folds não pode ser "
            "maior que a quantidade de observações "
            "da menor classe."
        )

    kfold = StratifiedKFold(
        n_splits=quantidade_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados = []

    linhas = []

    linhas.append(
        "MÉTODO: CROSS-VALIDATION"
    )

    linhas.append(
        f"\nQuantidade de folds: "
        f"{quantidade_folds}"
    )

    linhas.append(
        "\nFoi utilizada validação cruzada "
        "estratificada."
    )

    # --------------------------------------------------------
    # CADA FOLD
    # --------------------------------------------------------

    for numero_fold, (
        indices_treino,
        indices_teste
    ) in enumerate(
        kfold.split(X, y),
        start=1
    ):

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

        modelo = GaussianNB()

        modelo.fit(
            X_treino,
            y_treino
        )

        y_pred = modelo.predict(
            X_teste
        )

        metricas, matriz, relatorio = (
            calcular_metricas(
                y_teste,
                y_pred
            )
        )

        resultados.append(
            metricas
        )

        linhas.append(
            "\n" + "=" * 60
        )

        linhas.append(
            f"FOLD {numero_fold}"
        )

        linhas.append(
            "=" * 60
        )

        linhas.append(
            f"Treinamento: "
            f"{len(X_treino)} linhas"
        )

        linhas.append(
            f"Teste: "
            f"{len(X_teste)} linhas"
        )

        linhas.append(
            "\nMÉTRICAS"
        )

        linhas.append(
            formatar_metricas(
                metricas
            )
        )

        linhas.append(
            "\nMATRIZ DE CONFUSÃO"
        )

        linhas.append(
            formatar_matriz(
                matriz
            )
        )

        linhas.append(
            "\nRELATÓRIO DE CLASSIFICAÇÃO"
        )

        linhas.append(
            relatorio
        )

    # --------------------------------------------------------
    # MÉDIAS FINAIS
    # --------------------------------------------------------

    linhas.append(
        "\n" + "=" * 60
    )

    linhas.append(
        "RESULTADO FINAL"
    )

    linhas.append(
        "=" * 60
    )

    nomes_metricas = [
        "acuracia",
        "precisao",
        "recall",
        "f1"
    ]

    for nome in nomes_metricas:

        valores = [
            resultado[nome]
            for resultado in resultados
        ]

        media = np.mean(
            valores
        )

        desvio = np.std(
            valores
        )

        nome_formatado = {
            "acuracia": "Acurácia",
            "precisao": "Precisão",
            "recall": "Recall",
            "f1": "F1-score"
        }[nome]

        linhas.append(
            f"{nome_formatado}: "
            f"{media:.4f} "
            f"(± {desvio:.4f})"
        )

    return "\n".join(linhas)


# ============================================================
# ESCOLHER MÉTODO DE AVALIAÇÃO
# ============================================================

def selecionar_metodo():

    print("\n" + "=" * 60)
    print("MÉTODO DE AVALIAÇÃO")
    print("=" * 60)

    print("1 - Auto teste")
    print("2 - Hold Out")
    print("3 - Cross-Validation")

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao in {"1", "2", "3"}:

            return opcao

        print(
            "Erro: escolha uma opção de 1 a 3."
        )


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("013 - NAIVE BAYES")
    print("=" * 60)

    print(
        "\nModelo utilizado: Gaussian Naive Bayes"
    )

    # --------------------------------------------------------
    # ESCOLHER MÉTODO
    # --------------------------------------------------------

    metodo = selecionar_metodo()

    # --------------------------------------------------------
    # HOLD OUT
    # --------------------------------------------------------

    if metodo == "2":

        print("\n" + "=" * 60)
        print("HOLD OUT")
        print("=" * 60)

        while True:

            opcao_datasets = input(
                "\nVocê já possui os dados de "
                "treinamento e teste separados?\n"
                "1 - Sim\n"
                "2 - Não\n\n"
                "Digite a opção: "
            ).strip()

            if opcao_datasets in {"1", "2"}:
                break

            print(
                "Erro: escolha 1 ou 2."
            )

        # ----------------------------------------------------
        # DATASETS SEPARADOS
        # ----------------------------------------------------

        if opcao_datasets == "1":

            print(
                "\nSelecione o dataset de TREINAMENTO:"
            )

            caminho_treino = selecionar_arquivo()

            if caminho_treino is None:
                return

            df_treino = carregar_arquivo(
                caminho_treino
            )

            if df_treino is None:
                return

            print(
                "\nSelecione o dataset de TESTE:"
            )

            caminho_teste = selecionar_arquivo()

            if caminho_teste is None:
                return

            df_teste = carregar_arquivo(
                caminho_teste
            )

            if df_teste is None:
                return

        # ----------------------------------------------------
        # UM DATASET
        # ----------------------------------------------------

        else:

            caminho_treino = selecionar_arquivo()

            if caminho_treino is None:
                return

            df_treino = carregar_arquivo(
                caminho_treino
            )

            if df_treino is None:
                return

            df_teste = None

    # --------------------------------------------------------
    # AUTO TESTE / CROSS-VALIDATION
    # --------------------------------------------------------

    else:

        caminho_treino = selecionar_arquivo()

        if caminho_treino is None:
            return

        df_treino = carregar_arquivo(
            caminho_treino
        )

        if df_treino is None:
            return

        df_teste = None

    # --------------------------------------------------------
    # INFORMAÇÕES DO DATASET
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET")
    print("=" * 60)

    print(
        f"Arquivo: "
        f"{caminho_treino.name}"
    )

    print(
        f"Linhas: "
        f"{len(df_treino)}"
    )

    print(
        f"Colunas: "
        f"{len(df_treino.columns)}"
    )

    # --------------------------------------------------------
    # VARIÁVEL-ALVO
    # --------------------------------------------------------

    mostrar_colunas(
        df_treino
    )

    coluna_alvo = selecionar_coluna(
        df_treino,
        "\nDigite o número da variável-alvo: "
    )

    print(
        f"\nVariável-alvo selecionada: "
        f"{coluna_alvo}"
    )

    print(
        f"Tipo original: "
        f"{df_treino[coluna_alvo].dtype}"
    )

    # --------------------------------------------------------
    # VALIDAR PREDITORAS
    # --------------------------------------------------------

    if not validar_preditoras(
        df_treino,
        coluna_alvo
    ):

        return

    # --------------------------------------------------------
    # PREPARAR VARIÁVEL-ALVO
    # --------------------------------------------------------

    try:

        y, mapeamento = (
            preparar_variavel_alvo(
                df_treino[coluna_alvo]
            )
        )

    except ValueError as erro:

        print(
            f"\nErro ao preparar a variável-alvo:"
            f"\n{erro}"
        )

        return

    # --------------------------------------------------------
    # VALIDAR VARIÁVEL-ALVO
    # --------------------------------------------------------

    if not validar_variavel_alvo(
        y,
        coluna_alvo
    ):

        return

    # --------------------------------------------------------
    # MOSTRAR MAPEAMENTO DA ALVO
    # --------------------------------------------------------

    if mapeamento is not None:

        print("\n" + "-" * 60)
        print("CONVERSÃO DA VARIÁVEL-ALVO")
        print("-" * 60)

        print(
            "\nA variável-alvo é categórica."
        )

        print(
            "As categorias serão convertidas "
            "automaticamente para números:"
        )

        for categoria, numero in (
            mapeamento.items()
        ):

            print(
                f"{categoria} → {numero}"
            )

    else:

        print(
            "\nA variável-alvo já é numérica "
            "e será utilizada como está."
        )

    # --------------------------------------------------------
    # VALIDAR DATASET DE TESTE
    # --------------------------------------------------------

    if df_teste is not None:

        if coluna_alvo not in df_teste.columns:

            print("\n" + "=" * 60)
            print("ERRO - VARIÁVEL-ALVO")
            print("=" * 60)

            print(
                f"\nA coluna '{coluna_alvo}' "
                "não existe no dataset de teste."
            )

            return

        if not validar_preditoras(
            df_teste,
            coluna_alvo
        ):

            return

        # ----------------------------------------------------
        # VERIFICAR COLUNAS
        # ----------------------------------------------------

        colunas_treino = [
            coluna
            for coluna in df_treino.columns
            if coluna != coluna_alvo
        ]

        colunas_teste = [
            coluna
            for coluna in df_teste.columns
            if coluna != coluna_alvo
        ]

        if set(colunas_treino) != set(
            colunas_teste
        ):

            print("\n" + "=" * 60)
            print("ERRO - DATASETS INCOMPATÍVEIS")
            print("=" * 60)

            print(
                "\nOs datasets de treinamento e "
                "teste precisam possuir as mesmas "
                "variáveis preditoras."
            )

            return

    # --------------------------------------------------------
    # EXECUTAR MÉTODO
    # --------------------------------------------------------

    try:

        if metodo == "1":

            texto_resultado = (
                executar_auto_teste(
                    df_treino,
                    coluna_alvo,
                    y,
                    mapeamento
                )
            )

        elif metodo == "2":

            if df_teste is not None:

                texto_resultado = (
                    executar_hold_out(
                        df_treino,
                        df_teste,
                        coluna_alvo,
                        mapeamento_alvo=mapeamento
                    )
                )

            else:

                print("\n" + "=" * 60)
                print("DIVISÃO DO DATASET")
                print("=" * 60)

                while True:

                    try:

                        porcentagem = float(
                            input(
                                "\nQual percentual deseja "
                                "utilizar para treinamento? "
                                "(ex.: 80): "
                            ).strip()
                            .replace(",", ".")
                        )

                        if (
                            porcentagem > 0
                            and porcentagem < 100
                        ):

                            break

                        print(
                            "Erro: informe um valor "
                            "entre 0 e 100."
                        )

                    except ValueError:

                        print(
                            "Erro: informe uma porcentagem válida."
                        )

                percentual_treino = (
                    porcentagem / 100
                )

                texto_resultado = (
                    executar_hold_out(
                        df_treino,
                        None,
                        coluna_alvo,
                        percentual_treino,
                        mapeamento
                    )
                )

        else:

            print("\n" + "=" * 60)
            print("CROSS-VALIDATION")
            print("=" * 60)

            while True:

                try:

                    quantidade_folds = int(
                        input(
                            "\nQuantas repartições "
                            "(folds) deseja utilizar? "
                        ).strip()
                    )

                    if quantidade_folds >= 2:

                        break

                    print(
                        "Erro: utilize pelo menos "
                        "2 folds."
                    )

                except ValueError:

                    print(
                        "Erro: informe um número inteiro."
                    )

            texto_resultado = (
                executar_cross_validation(
                    df_treino,
                    coluna_alvo,
                    y,
                    quantidade_folds
                )
            )

    except Exception as erro:

        print("\n" + "=" * 60)
        print("ERRO DURANTE O TREINAMENTO")
        print("=" * 60)

        print(
            f"\n{erro}"
        )

        return

    # --------------------------------------------------------
    # MONTAR RELATÓRIO
    # --------------------------------------------------------

    relatorio = []

    relatorio.append(
        "=" * 60
    )

    relatorio.append(
        "NAIVE BAYES - RELATÓRIO"
    )

    relatorio.append(
        "=" * 60
    )

    relatorio.append(
        f"\nModelo: Gaussian Naive Bayes"
    )

    relatorio.append(
        f"Dataset: {caminho_treino.name}"
    )

    relatorio.append(
        f"Variável-alvo: {coluna_alvo}"
    )

    relatorio.append(
        f"Linhas: {len(df_treino)}"
    )

    relatorio.append(
        f"Variáveis preditoras: "
        f"{len(df_treino.columns) - 1}"
    )

    # --------------------------------------------------------
    # MAPEAMENTO DA ALVO
    # --------------------------------------------------------

    if mapeamento is not None:

        relatorio.append(
            "\n" + "-" * 60
        )

        relatorio.append(
            "MAPEAMENTO DA VARIÁVEL-ALVO"
        )

        relatorio.append(
            "-" * 60
        )

        for categoria, numero in (
            mapeamento.items()
        ):

            relatorio.append(
                f"{categoria} → {numero}"
            )

    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    relatorio.append(
        "\n" + "=" * 60
    )

    relatorio.append(
        texto_resultado
    )

    relatorio.append(
        "\n" + "=" * 60
    )

    relatorio.append(
        "FIM DO RELATÓRIO"
    )

    relatorio.append(
        "=" * 60
    )

    texto_final = "\n".join(
        relatorio
    )

    # --------------------------------------------------------
    # EXIBIR RESULTADO
    # --------------------------------------------------------

    print("\n")
    print(texto_final)

    # --------------------------------------------------------
    # SALVAR TXT
    # --------------------------------------------------------

    nome_saida = (
        f"013_Naive_Bayes_"
        f"{caminho_treino.stem}.txt"
    )

    caminho_saida = (
        PASTA_SAIDA / nome_saida
    )

    try:

        with open(
            caminho_saida,
            "w",
            encoding="utf-8"
        ) as arquivo:

            arquivo.write(
                texto_final
            )

    except Exception as erro:

        print(
            "\nErro ao salvar o relatório:"
        )

        print(erro)

        return

    # --------------------------------------------------------
    # FINALIZAÇÃO
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PROCESSO CONCLUÍDO")
    print("=" * 60)

    print(
        f"\nRelatório salvo em:"
        f"\n{caminho_saida}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()