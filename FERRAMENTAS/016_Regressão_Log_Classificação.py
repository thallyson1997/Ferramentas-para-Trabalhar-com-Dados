# ============================================================
# REGRESSÃO LOGÍSTICA - CLASSIFICAÇÃO
# ============================================================

from pathlib import Path
from utilidades import ler_arquivo
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
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
# CONFIGURAÇÕES GERAIS
# ============================================================

RANDOM_STATE = 42

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent
PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"

PASTA_SAIDA.mkdir(exist_ok=True)


# ============================================================
# SELEÇÃO DE ARQUIVO
# ============================================================

def selecionar_arquivo():
    extensoes_suportadas = {
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
            and arquivo.suffix.lower() in extensoes_suportadas
        ]
    )

    if not arquivos:
        raise FileNotFoundError(
            "Nenhum arquivo compatível foi encontrado na pasta ENTRADA."
        )

    print("\n" + "=" * 60)
    print("Selecione o dataset:")
    print("=" * 60)

    for i, arquivo in enumerate(arquivos, start=1):
        print(f"{i} - {arquivo.name}")

    while True:
        try:
            opcao = int(
                input("\nDigite o número do arquivo: ")
            )

            if 1 <= opcao <= len(arquivos):
                caminho = arquivos[opcao - 1]
                return caminho, ler_arquivo(caminho)

            print("Opção inválida.")

        except ValueError:
            print("Digite um número válido.")


# ============================================================
# SELEÇÃO DE COLUNA
# ============================================================

def selecionar_coluna(df, mensagem):
    print("\n" + "=" * 60)
    print(mensagem)
    print("=" * 60)

    for i, coluna in enumerate(df.columns, start=1):
        print(
            f"{i} - {coluna} "
            f"({df[coluna].dtype})"
        )

    while True:
        try:
            opcao = int(
                input("\nDigite o número da coluna: ")
            )

            if 1 <= opcao <= len(df.columns):
                return df.columns[opcao - 1]

            print("Opção inválida.")

        except ValueError:
            print("Digite um número válido.")


# ============================================================
# CONFIGURAÇÃO DO SOLVER
# ============================================================

def configurar_solver(valor_atual):
    print("\nSolver")
    print(f"Valor atual: '{valor_atual}'")

    print("1 - lbfgs")
    print("2 - liblinear")
    print("3 - newton-cg")
    print("4 - newton-cholesky")
    print("5 - sag")
    print("6 - saga")

    while True:
        entrada = input(
            "Digite a opção "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        opcoes = {
            "1": "lbfgs",
            "2": "liblinear",
            "3": "newton-cg",
            "4": "newton-cholesky",
            "5": "sag",
            "6": "saga"
        }

        if entrada in opcoes:
            return opcoes[entrada]

        print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO DO C
# ============================================================

def configurar_c(valor_atual):
    print("\nParâmetro C")
    print(f"Valor atual: {valor_atual}")

    print(
        "Valores comuns: "
        "0.01, 0.1, 1.0, 10.0, 100.0"
    )

    print(
        "Valores menores significam regularização mais forte."
    )

    print(
        "Digite 'inf' para utilizar C infinito "
        "(sem regularização)."
    )

    while True:
        entrada = input(
            "Digite o novo valor de C "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        if entrada.lower() in {
            "inf",
            "infinito",
            "infinity"
        }:
            return np.inf

        try:
            valor = float(entrada)

            if valor > 0:
                return valor

            print(
                "C deve ser maior que zero."
            )

        except ValueError:
            print(
                "Digite um número válido."
            )


# ============================================================
# CONFIGURAÇÃO DO L1_RATIO
# ============================================================

def configurar_l1_ratio(valor_atual):
    print("\nL1 Ratio")
    print(f"Valor atual: {valor_atual}")

    print(
        "0.0 = L2"
    )

    print(
        "1.0 = L1"
    )

    print(
        "Entre 0 e 1 = Elastic Net"
    )

    print(
        "Valores comuns: 0.0, 0.25, 0.5, 0.75, 1.0"
    )

    while True:
        entrada = input(
            "Digite o novo valor de l1_ratio "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        try:
            valor = float(entrada)

            if 0 <= valor <= 1:
                return valor

            print(
                "l1_ratio deve estar entre 0 e 1."
            )

        except ValueError:
            print(
                "Digite um número válido."
            )


# ============================================================
# CONFIGURAÇÃO DO MAX_ITER
# ============================================================

def configurar_max_iter(valor_atual):
    print("\nMax Iter")
    print(f"Valor atual: {valor_atual}")

    print(
        "Valores comuns: "
        "100, 500, 1000, 2000"
    )

    while True:
        entrada = input(
            "Digite o novo valor "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        try:
            valor = int(entrada)

            if valor >= 1:
                return valor

            print(
                "max_iter deve ser maior ou igual a 1."
            )

        except ValueError:
            print(
                "Digite um número inteiro válido."
            )


# ============================================================
# CONFIGURAÇÃO DO TOL
# ============================================================

def configurar_tol(valor_atual):
    print("\nTol")
    print(f"Valor atual: {valor_atual}")

    print(
        "Valores comuns: "
        "1e-4, 1e-5, 1e-6"
    )

    while True:
        entrada = input(
            "Digite o novo valor "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        try:
            valor = float(entrada)

            if valor > 0:
                return valor

            print(
                "tol deve ser maior que zero."
            )

        except ValueError:
            print(
                "Digite um número válido."
            )


# ============================================================
# CONFIGURAÇÃO DO CLASS_WEIGHT
# ============================================================

def configurar_class_weight(valor_atual):
    print("\nClass Weight")
    print(f"Valor atual: {valor_atual}")

    print("1 - None")
    print("2 - balanced")

    while True:
        entrada = input(
            "Digite a opção "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        if entrada == "1":
            return None

        if entrada == "2":
            return "balanced"

        print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

def configurar_modelo():

    parametros = {
        "C": 1.0,
        "l1_ratio": 0.0,
        "solver": "lbfgs",
        "max_iter": 100,
        "tol": 1e-4,
        "class_weight": None
    }

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DA REGRESSÃO LOGÍSTICA")
    print("=" * 60)

    print("\nModelo: LogisticRegression")

    print("\nParâmetros atuais:")

    print(
        f"C            = {parametros['C']}"
    )

    print(
        f"l1_ratio     = {parametros['l1_ratio']}"
    )

    print(
        f"solver       = '{parametros['solver']}'"
    )

    print(
        f"max_iter     = {parametros['max_iter']}"
    )

    print(
        f"tol          = {parametros['tol']}"
    )

    print(
        f"class_weight = {parametros['class_weight']}"
    )

    print(
        "\nInterpretação do l1_ratio:"
    )

    print(
        "0.0 = L2"
    )

    print(
        "1.0 = L1"
    )

    print(
        "0.0 < l1_ratio < 1.0 = Elastic Net"
    )

    print(
        "C = inf = sem regularização"
    )

    while True:

        print(
            "\nDeseja utilizar essa configuração?"
        )

        print(
            "1 - Sim, utilizar padrão"
        )

        print(
            "2 - Não, modificar parâmetros"
        )

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao == "1":
            return parametros

        if opcao == "2":
            break

        print(
            "Opção inválida."
        )

    while True:

        print("\n" + "=" * 60)
        print(
            "PARÂMETROS DA REGRESSÃO LOGÍSTICA"
        )
        print("=" * 60)

        print(
            f"1 - C            = {parametros['C']}"
        )

        print(
            f"2 - l1_ratio     = {parametros['l1_ratio']}"
        )

        print(
            f"3 - solver       = '{parametros['solver']}'"
        )

        print(
            f"4 - max_iter     = {parametros['max_iter']}"
        )

        print(
            f"5 - tol          = {parametros['tol']}"
        )

        print(
            f"6 - class_weight = {parametros['class_weight']}"
        )

        print(
            "7 - Finalizar configuração"
        )

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao == "1":

            parametros["C"] = configurar_c(
                parametros["C"]
            )

        elif opcao == "2":

            parametros["l1_ratio"] = (
                configurar_l1_ratio(
                    parametros["l1_ratio"]
                )
            )

        elif opcao == "3":

            parametros["solver"] = (
                configurar_solver(
                    parametros["solver"]
                )
            )

        elif opcao == "4":

            parametros["max_iter"] = (
                configurar_max_iter(
                    parametros["max_iter"]
                )
            )

        elif opcao == "5":

            parametros["tol"] = configurar_tol(
                parametros["tol"]
            )

        elif opcao == "6":

            parametros["class_weight"] = (
                configurar_class_weight(
                    parametros["class_weight"]
                )
            )

        elif opcao == "7":
            break

        else:
            print(
                "Opção inválida."
            )

    print("\n" + "=" * 60)
    print(
        "CONFIGURAÇÃO FINAL DA REGRESSÃO LOGÍSTICA"
    )
    print("=" * 60)

    for chave, valor in parametros.items():

        if isinstance(valor, str):

            print(
                f"{chave:<15} = '{valor}'"
            )

        else:

            print(
                f"{chave:<15} = {valor}"
            )

    return parametros


# ============================================================
# VALIDAÇÃO DA CONFIGURAÇÃO DO MODELO
# ============================================================

def validar_configuracao_modelo(parametros):

    solver = parametros["solver"]
    l1_ratio = parametros["l1_ratio"]

    solvers_validos = {
        "lbfgs",
        "liblinear",
        "newton-cg",
        "newton-cholesky",
        "sag",
        "saga"
    }

    if solver not in solvers_validos:

        raise ValueError(
            f"Solver inválido: {solver}"
        )

    # --------------------------------------------------------
    # Solvers diferentes de SAGA aceitam somente:
    # l1_ratio = 0 (L2)
    # ou
    # l1_ratio = 1 (L1, somente liblinear)
    # --------------------------------------------------------

    if solver in {
        "lbfgs",
        "newton-cg",
        "newton-cholesky",
        "sag"
    }:

        if l1_ratio != 0:

            raise ValueError(
                f"O solver '{solver}' utiliza "
                "l1_ratio=0 (L2). "
                "Para L1 ou Elastic Net, utilize "
                "um solver compatível."
            )

    # --------------------------------------------------------
    # LIBLINEAR
    # --------------------------------------------------------

    elif solver == "liblinear":

        if l1_ratio not in {
            0,
            1
        }:

            raise ValueError(
                "O solver 'liblinear' aceita "
                "l1_ratio=0 (L2) ou "
                "l1_ratio=1 (L1). "
                "Elastic Net requer o solver 'saga'."
            )

    # --------------------------------------------------------
    # SAGA
    # --------------------------------------------------------

    elif solver == "saga":

        if not (
            0 <= l1_ratio <= 1
        ):

            raise ValueError(
                "l1_ratio deve estar entre 0 e 1."
            )

    # --------------------------------------------------------
    # C
    # --------------------------------------------------------

    C = parametros["C"]

    if not (
        np.isinf(C)
        or C > 0
    ):

        raise ValueError(
            "C deve ser maior que zero ou infinito."
        )


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def criar_modelo(parametros):

    validar_configuracao_modelo(
        parametros
    )

    argumentos = {
        "C": parametros["C"],
        "l1_ratio": parametros["l1_ratio"],
        "solver": parametros["solver"],
        "max_iter": parametros["max_iter"],
        "tol": parametros["tol"],
        "class_weight": parametros["class_weight"],
        "random_state": RANDOM_STATE
    }

    return LogisticRegression(
        **argumentos
    )


# ============================================================
# VALIDAÇÃO DO DATASET
# ============================================================

def validar_dataset(df, coluna_alvo):

    problemas = []

    if coluna_alvo not in df.columns:

        problemas.append(
            f"A variável alvo '{coluna_alvo}' "
            "não existe no dataset."
        )

        return problemas

    if df.empty:

        problemas.append(
            "O dataset está vazio."
        )

    if df[coluna_alvo].isna().any():

        quantidade = (
            df[coluna_alvo].isna().sum()
        )

        problemas.append(
            f"A variável alvo possui "
            f"{quantidade} valor(es) ausente(s)."
        )

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    for coluna in colunas_preditoras:

        if df[coluna].isna().any():

            quantidade = (
                df[coluna].isna().sum()
            )

            problemas.append(
                f"O preditor '{coluna}' possui "
                f"{quantidade} valor(es) ausente(s)."
            )

        if not pd.api.types.is_numeric_dtype(
            df[coluna]
        ):

            problemas.append(
                f"O preditor '{coluna}' possui "
                f"tipo '{df[coluna].dtype}' "
                "e não é numérico."
            )

    if not problemas:

        quantidade_classes = (
            df[coluna_alvo].nunique()
        )

        if quantidade_classes < 2:

            problemas.append(
                "A variável alvo precisa possuir "
                "pelo menos 2 classes diferentes."
            )

    return problemas


# ============================================================
# PREPARAÇÃO DA VARIÁVEL ALVO
# ============================================================

def preparar_alvo(
    df,
    coluna_alvo,
    mapeamento=None
):

    serie = df[coluna_alvo]

    if pd.api.types.is_numeric_dtype(
        serie
    ):

        return serie.copy(), None

    categorias = (
        pd.Series(serie)
        .dropna()
        .unique()
        .tolist()
    )

    if mapeamento is None:

        categorias = sorted(
            categorias,
            key=lambda valor: str(valor)
        )

        mapeamento = {
            categoria: indice
            for indice, categoria
            in enumerate(categorias)
        }

    valores_desconhecidos = [
        valor
        for valor in serie.dropna().unique()
        if valor not in mapeamento
    ]

    if valores_desconhecidos:

        raise ValueError(
            "Foram encontradas categorias no alvo "
            "que não existem no mapeamento de "
            f"treinamento: {valores_desconhecidos}"
        )

    alvo = serie.map(
        mapeamento
    )

    if alvo.isna().any():

        raise ValueError(
            "Não foi possível converter todos "
            "os valores da variável alvo."
        )

    alvo = alvo.astype(int)

    return alvo, mapeamento


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

    X = df[
        colunas_preditoras
    ].copy()

    for coluna in colunas_preditoras:

        if not pd.api.types.is_numeric_dtype(
            X[coluna]
        ):

            raise ValueError(
                f"O preditor '{coluna}' "
                "não é numérico."
            )

    return X, colunas_preditoras


# ============================================================
# CÁLCULO DAS MÉTRICAS
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

    matriz = confusion_matrix(
        y_real,
        y_pred
    )

    relatorio = classification_report(
        y_real,
        y_pred,
        zero_division=0
    )

    return matriz, relatorio


# ============================================================
# AUTO TESTE
# ============================================================

def executar_auto_teste(
    df,
    coluna_alvo,
    parametros
):

    print("\n" + "=" * 60)
    print(
        "EXECUTANDO AUTO TESTE"
    )
    print("=" * 60)

    X, colunas_preditoras = (
        preparar_preditores(
            df,
            coluna_alvo
        )
    )

    y, mapeamento = preparar_alvo(
        df,
        coluna_alvo
    )

    modelo = criar_modelo(
        parametros
    )

    modelo.fit(
        X,
        y
    )

    y_pred = modelo.predict(
        X
    )

    metricas = calcular_metricas(
        y,
        y_pred
    )

    matriz, relatorio = (
        gerar_avaliacao_textual(
            y,
            y_pred
        )
    )

    print(
        "\nAuto Teste concluído."
    )

    return {
        "metodo": "Auto Teste",
        "metricas": metricas,
        "matriz": matriz,
        "relatorio": relatorio,
        "mapeamento": mapeamento,
        "colunas_preditoras": colunas_preditoras
    }


# ============================================================
# HOLD OUT
# ============================================================

def executar_holdout(
    df_treino,
    df_teste,
    coluna_alvo,
    parametros,
    percentual_treino=None
):

    print("\n" + "=" * 60)
    print(
        "EXECUTANDO HOLD OUT"
    )
    print("=" * 60)

    # ========================================================
    # DATASET ÚNICO
    # ========================================================

    if df_teste is None:

        X, colunas_preditoras = (
            preparar_preditores(
                df_treino,
                coluna_alvo
            )
        )

        y, mapeamento = (
            preparar_alvo(
                df_treino,
                coluna_alvo
            )
        )

        X_treino, X_teste, y_treino, y_teste = (
            train_test_split(
                X,
                y,
                test_size=1 - percentual_treino,
                random_state=RANDOM_STATE,
                stratify=y
            )
        )

    # ========================================================
    # DATASETS SEPARADOS
    # ========================================================

    else:

        X_treino, colunas_preditoras_treino = (
            preparar_preditores(
                df_treino,
                coluna_alvo
            )
        )

        y_treino, mapeamento = (
            preparar_alvo(
                df_treino,
                coluna_alvo
            )
        )

        colunas_preditoras_teste = [
            coluna
            for coluna in df_teste.columns
            if coluna != coluna_alvo
        ]

        if set(
            colunas_preditoras_treino
        ) != set(
            colunas_preditoras_teste
        ):

            raise ValueError(
                "Os datasets de treino e teste "
                "não possuem as mesmas "
                "variáveis preditoras."
            )

        # ----------------------------------------------------
        # Mantém exatamente a mesma ordem
        # das colunas utilizadas no treinamento.
        # ----------------------------------------------------

        X_teste = df_teste[
            colunas_preditoras_treino
        ].copy()

        for coluna in (
            colunas_preditoras_treino
        ):

            if not pd.api.types.is_numeric_dtype(
                X_teste[coluna]
            ):

                raise ValueError(
                    f"O preditor '{coluna}' "
                    "no dataset de teste "
                    "não é numérico."
                )

            if X_teste[coluna].isna().any():

                raise ValueError(
                    f"O preditor '{coluna}' "
                    "no dataset de teste "
                    "possui valores ausentes."
                )

        y_teste, _ = preparar_alvo(
            df_teste,
            coluna_alvo,
            mapeamento
        )

        classes_treino = set(
            y_treino.unique()
        )

        classes_teste = set(
            y_teste.unique()
        )

        classes_desconhecidas = (
            classes_teste - classes_treino
        )

        if classes_desconhecidas:

            raise ValueError(
                "O dataset de teste possui "
                "classes que não existem no "
                "dataset de treinamento: "
                f"{sorted(classes_desconhecidas)}"
            )

        colunas_preditoras = (
            colunas_preditoras_treino
        )

    # ========================================================
    # TREINAMENTO
    # ========================================================

    modelo = criar_modelo(
        parametros
    )

    modelo.fit(
        X_treino,
        y_treino
    )

    # ========================================================
    # PREDIÇÃO
    # ========================================================

    y_pred = modelo.predict(
        X_teste
    )

    # ========================================================
    # MÉTRICAS
    # ========================================================

    metricas = calcular_metricas(
        y_teste,
        y_pred
    )

    matriz, relatorio = (
        gerar_avaliacao_textual(
            y_teste,
            y_pred
        )
    )

    print(
        "\nHold Out concluído."
    )

    return {
        "metodo": "Hold Out",
        "metricas": metricas,
        "matriz": matriz,
        "relatorio": relatorio,
        "mapeamento": mapeamento,
        "colunas_preditoras": colunas_preditoras
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def executar_cross_validation(
    df,
    coluna_alvo,
    parametros,
    quantidade_folds
):

    print("\n" + "=" * 60)
    print(
        "EXECUTANDO CROSS-VALIDATION"
    )
    print("=" * 60)

    X, colunas_preditoras = (
        preparar_preditores(
            df,
            coluna_alvo
        )
    )

    y, mapeamento = (
        preparar_alvo(
            df,
            coluna_alvo
        )
    )

    menor_classe = (
        y.value_counts().min()
    )

    if quantidade_folds > menor_classe:

        raise ValueError(
            f"A quantidade de folds "
            f"({quantidade_folds}) não pode "
            "ser maior que a quantidade "
            "de observações da menor classe "
            f"({menor_classe})."
        )

    cross_validation = (
        StratifiedKFold(
            n_splits=quantidade_folds,
            shuffle=True,
            random_state=RANDOM_STATE
        )
    )

    resultados_folds = []

    for numero_fold, (
        indices_treino,
        indices_teste
    ) in enumerate(
        cross_validation.split(
            X,
            y
        ),
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

        modelo = criar_modelo(
            parametros
        )

        modelo.fit(
            X_treino,
            y_treino
        )

        y_pred = modelo.predict(
            X_teste
        )

        metricas = calcular_metricas(
            y_teste,
            y_pred
        )

        matriz, relatorio = (
            gerar_avaliacao_textual(
                y_teste,
                y_pred
            )
        )

        resultados_folds.append({
            "fold": numero_fold,
            "metricas": metricas,
            "matriz": matriz,
            "relatorio": relatorio
        })

        print(
            f"Fold {numero_fold}/"
            f"{quantidade_folds} concluído."
        )

    # ========================================================
    # MÉDIA E DESVIO PADRÃO
    # ========================================================

    metricas_finais = {}

    for metrica in [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]:

        valores = [
            resultado["metricas"][metrica]
            for resultado in resultados_folds
        ]

        metricas_finais[metrica] = {
            "media": np.mean(
                valores
            ),

            "desvio_padrao": (
                np.std(
                    valores,
                    ddof=1
                )
                if len(valores) > 1
                else 0.0
            )
        }

    print(
        "\nCross-validation concluído."
    )

    return {
        "metodo": "Cross-validation",
        "quantidade_folds": quantidade_folds,
        "folds": resultados_folds,
        "metricas_finais": metricas_finais,
        "mapeamento": mapeamento,
        "colunas_preditoras": colunas_preditoras
    }


# ============================================================
# SELEÇÃO DO MÉTODO
# ============================================================

def selecionar_metodo():

    print("\n" + "=" * 60)
    print(
        "MÉTODO DE AVALIAÇÃO"
    )
    print("=" * 60)

    print(
        "1 - Auto Teste"
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

        if opcao in {
            "1",
            "2",
            "3"
        }:

            return opcao

        print(
            "Opção inválida."
        )


# ============================================================
# CONFIGURAÇÃO DO HOLD OUT
# ============================================================

def configurar_holdout():

    print("\n" + "=" * 60)
    print(
        "CONFIGURAÇÃO DO HOLD OUT"
    )
    print("=" * 60)

    print(
        "Os dados de treino e teste "
        "já estão separados?"
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

        if opcao in {
            "1",
            "2"
        }:

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
            "\nDigite o percentual destinado "
            "ao treinamento (ex.: 80): "
        ).strip()

        try:

            percentual = float(
                entrada
            )

            if 50 <= percentual < 100:

                return percentual / 100

            print(
                "Digite um percentual entre 50 e 99."
            )

        except ValueError:

            print(
                "Digite um número válido."
            )


# ============================================================
# QUANTIDADE DE FOLDS
# ============================================================

def solicitar_folds():

    while True:

        entrada = input(
            "\nDigite a quantidade de folds "
            "(ex.: 5): "
        ).strip()

        try:

            folds = int(
                entrada
            )

            if folds >= 2:

                return folds

            print(
                "A quantidade de folds deve "
                "ser maior ou igual a 2."
            )

        except ValueError:

            print(
                "Digite um número inteiro válido."
            )


# ============================================================
# RELATÓRIO
# ============================================================

def criar_relatorio(
    caminho_dataset,
    metodo,
    coluna_alvo,
    parametros,
    resultado
):

    linhas = []

    linhas.append(
        "=" * 70
    )

    linhas.append(
        "REGRESSÃO LOGÍSTICA - CLASSIFICAÇÃO"
    )

    linhas.append(
        "=" * 70
    )

    linhas.append("")

    linhas.append(
        f"Dataset: {caminho_dataset.name}"
    )

    linhas.append(
        f"Método: {metodo}"
    )

    linhas.append(
        f"Variável alvo: {coluna_alvo}"
    )

    linhas.append("")

    linhas.append(
        "=" * 70
    )

    linhas.append(
        "CONFIGURAÇÃO DO MODELO"
    )

    linhas.append(
        "=" * 70
    )

    linhas.append(
        "Modelo: LogisticRegression"
    )

    for chave, valor in (
        parametros.items()
    ):

        if isinstance(
            valor,
            str
        ):

            linhas.append(
                f"{chave:<15} = '{valor}'"
            )

        else:

            linhas.append(
                f"{chave:<15} = {valor}"
            )

    linhas.append(
        f"{'random_state':<15} = "
        f"{RANDOM_STATE}"
    )

    linhas.append("")

    linhas.append(
        "Observação:"
    )

    linhas.append(
        "A regularização é configurada "
        "por C e l1_ratio, conforme a "
        "API atual do scikit-learn."
    )

    linhas.append(
        "l1_ratio=0 representa L2; "
        "l1_ratio=1 representa L1; "
        "valores entre 0 e 1 representam "
        "Elastic Net."
    )

    linhas.append(
        "C=infinito representa ausência "
        "de regularização."
    )

    # ========================================================
    # AUTO TESTE / HOLD OUT
    # ========================================================

    if metodo in {
        "Auto Teste",
        "Hold Out"
    }:

        linhas.append("")

        linhas.append(
            "=" * 70
        )

        linhas.append(
            "MÉTRICAS"
        )

        linhas.append(
            "=" * 70
        )

        metricas = resultado[
            "metricas"
        ]

        linhas.append(
            f"Acurácia:           "
            f"{metricas['accuracy']:.4f}"
        )

        linhas.append(
            f"Precisão ponderada: "
            f"{metricas['precision']:.4f}"
        )

        linhas.append(
            f"Recall ponderado:   "
            f"{metricas['recall']:.4f}"
        )

        linhas.append(
            f"F1 ponderado:       "
            f"{metricas['f1']:.4f}"
        )

        linhas.append("")

        linhas.append(
            "=" * 70
        )

        linhas.append(
            "MATRIZ DE CONFUSÃO"
        )

        linhas.append(
            "=" * 70
        )

        linhas.append(
            np.array2string(
                resultado["matriz"]
            )
        )

        linhas.append("")

        linhas.append(
            "=" * 70
        )

        linhas.append(
            "CLASSIFICATION REPORT"
        )

        linhas.append(
            "=" * 70
        )

        linhas.append(
            resultado["relatorio"]
        )

        if resultado.get(
            "mapeamento"
        ):

            linhas.append("")

            linhas.append(
                "=" * 70
            )

            linhas.append(
                "MAPEAMENTO DA VARIÁVEL ALVO"
            )

            linhas.append(
                "=" * 70
            )

            for categoria, codigo in (
                resultado["mapeamento"].items()
            ):

                linhas.append(
                    f"{categoria} -> {codigo}"
                )

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    elif metodo == "Cross-validation":

        linhas.append("")

        linhas.append(
            "=" * 70
        )

        linhas.append(
            f"RESULTADOS DOS "
            f"{resultado['quantidade_folds']} FOLDS"
        )

        linhas.append(
            "=" * 70
        )

        for fold in resultado["folds"]:

            linhas.append("")

            linhas.append(
                f"---------------- FOLD "
                f"{fold['fold']} ----------------"
            )

            metricas = fold[
                "metricas"
            ]

            linhas.append(
                f"Acurácia:           "
                f"{metricas['accuracy']:.4f}"
            )

            linhas.append(
                f"Precisão ponderada: "
                f"{metricas['precision']:.4f}"
            )

            linhas.append(
                f"Recall ponderado:   "
                f"{metricas['recall']:.4f}"
            )

            linhas.append(
                f"F1 ponderado:       "
                f"{metricas['f1']:.4f}"
            )

            linhas.append("")

            linhas.append(
                "Matriz de confusão:"
            )

            linhas.append(
                np.array2string(
                    fold["matriz"]
                )
            )

            linhas.append("")

            linhas.append(
                "Classification report:"
            )

            linhas.append(
                fold["relatorio"]
            )

        # ====================================================
        # RESULTADO FINAL
        # ====================================================

        linhas.append("")

        linhas.append(
            "=" * 70
        )

        linhas.append(
            "RESULTADO FINAL DA CROSS-VALIDATION"
        )

        linhas.append(
            "=" * 70
        )

        metricas_finais = (
            resultado["metricas_finais"]
        )

        for metrica, valores in (
            metricas_finais.items()
        ):

            nome = {
                "accuracy": "Acurácia",
                "precision": "Precisão ponderada",
                "recall": "Recall ponderado",
                "f1": "F1 ponderado"
            }[metrica]

            linhas.append(
                f"{nome}: "
                f"{valores['media']:.4f} "
                f"+/- "
                f"{valores['desvio_padrao']:.4f}"
            )

        if resultado.get(
            "mapeamento"
        ):

            linhas.append("")

            linhas.append(
                "=" * 70
            )

            linhas.append(
                "MAPEAMENTO DA VARIÁVEL ALVO"
            )

            linhas.append(
                "=" * 70
            )

            for categoria, codigo in (
                resultado["mapeamento"].items()
            ):

                linhas.append(
                    f"{categoria} -> {codigo}"
                )

    linhas.append("")

    linhas.append(
        "=" * 70
    )

    linhas.append(
        "FIM DO RELATÓRIO"
    )

    linhas.append(
        "=" * 70
    )

    nome_saida = (
        f"016_Regressão_Log_Classificação_"
        f"{caminho_dataset.stem}.txt"
    )

    caminho_saida = (
        PASTA_SAIDA / nome_saida
    )

    with open(
        caminho_saida,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            "\n".join(linhas)
        )

    return caminho_saida


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print(
        "REGRESSÃO LOGÍSTICA - CLASSIFICAÇÃO"
    )
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # MÉTODO
        # ----------------------------------------------------

        metodo = selecionar_metodo()

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        caminho, df = (
            selecionar_arquivo()
        )

        # ----------------------------------------------------
        # CONFIGURAÇÕES DO HOLD OUT
        # ----------------------------------------------------

        df_teste = None
        percentual_treino = None

        # ----------------------------------------------------
        # VARIÁVEL ALVO
        # ----------------------------------------------------

        coluna_alvo = selecionar_coluna(
            df,
            "Selecione a variável alvo:"
        )

        if pd.api.types.is_numeric_dtype(
            df[coluna_alvo]
        ):

            print(
                "\nA variável alvo já é numérica."
            )

        else:

            print(
                "\nA variável alvo é categórica."
            )

            print(
                "As categorias serão convertidas "
                "automaticamente para números."
            )

        # ----------------------------------------------------
        # VALIDAÇÃO
        # ----------------------------------------------------

        problemas = validar_dataset(
            df,
            coluna_alvo
        )

        if problemas:

            print("\n" + "=" * 60)
            print(
                "ERROS ENCONTRADOS"
            )
            print("=" * 60)

            for problema in problemas:

                print(
                    f"- {problema}"
                )

            print(
                "\nO processamento foi interrompido."
            )

            return

        # ----------------------------------------------------
        # CONFIGURAÇÃO DO HOLD OUT
        # ----------------------------------------------------

        if metodo == "2":

            configuracao = (
                configurar_holdout()
            )

            if configuracao == "1":

                print(
                    "\nSelecione o dataset de teste."
                )

                caminho_teste, df_teste = (
                    selecionar_arquivo()
                )

                if coluna_alvo not in (
                    df_teste.columns
                ):

                    print(
                        f"\nA variável alvo "
                        f"'{coluna_alvo}' não existe "
                        "no dataset de teste."
                    )

                    return

                problemas_teste = (
                    validar_dataset(
                        df_teste,
                        coluna_alvo
                    )
                )

                if problemas_teste:

                    print(
                        "\n" + "=" * 60
                    )

                    print(
                        "ERROS NO DATASET DE TESTE"
                    )

                    print(
                        "=" * 60
                    )

                    for problema in (
                        problemas_teste
                    ):

                        print(
                            f"- {problema}"
                        )

                    return

            else:

                percentual_treino = (
                    solicitar_percentual_treino()
                )

        # ----------------------------------------------------
        # CONFIGURAÇÃO DO MODELO
        # ----------------------------------------------------

        parametros = (
            configurar_modelo()
        )

        # ----------------------------------------------------
        # EXECUÇÃO
        # ----------------------------------------------------

        if metodo == "1":

            resultado = (
                executar_auto_teste(
                    df,
                    coluna_alvo,
                    parametros
                )
            )

            nome_metodo = (
                "Auto Teste"
            )

        elif metodo == "2":

            resultado = (
                executar_holdout(
                    df_treino=df,
                    df_teste=df_teste,
                    coluna_alvo=coluna_alvo,
                    parametros=parametros,
                    percentual_treino=percentual_treino
                )
            )

            nome_metodo = (
                "Hold Out"
            )

        else:

            quantidade_folds = (
                solicitar_folds()
            )

            resultado = (
                executar_cross_validation(
                    df,
                    coluna_alvo,
                    parametros,
                    quantidade_folds
                )
            )

            nome_metodo = (
                "Cross-validation"
            )

        # ----------------------------------------------------
        # RELATÓRIO
        # ----------------------------------------------------

        caminho_saida = (
            criar_relatorio(
                caminho,
                nome_metodo,
                coluna_alvo,
                parametros,
                resultado
            )
        )

        print("\n" + "=" * 60)
        print(
            "PROCESSAMENTO CONCLUÍDO"
        )
        print("=" * 60)

        print(
            "\nRelatório salvo em:"
        )

        print(
            caminho_saida
        )

    except Exception as erro:

        print("\n" + "=" * 60)
        print(
            "ERRO DURANTE O PROCESSAMENTO"
        )
        print("=" * 60)

        print(
            f"\n{erro}"
        )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()