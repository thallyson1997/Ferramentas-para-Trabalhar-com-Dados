# ============================================================
# SVM - CLASSIFICAÇÃO
# ============================================================

from pathlib import Path
from utilidades import ler_arquivo
import pandas as pd
import numpy as np

from sklearn.svm import SVC
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
            opcao = int(input("\nDigite o número do arquivo: "))

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
            opcao = int(input("\nDigite o número da coluna: "))

            if 1 <= opcao <= len(df.columns):
                return df.columns[opcao - 1]

            print("Opção inválida.")

        except ValueError:
            print("Digite um número válido.")


# ============================================================
# CONFIGURAÇÃO DO PARÂMETRO C
# ============================================================

def configurar_c(valor_atual):
    print("\nParâmetro C")
    print(f"Valor atual: {valor_atual}")
    print("Valores comuns: 0.1, 1.0, 10.0, 100.0")

    while True:
        entrada = input(
            "Digite o novo valor de C "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        try:
            valor = float(entrada)

            if valor > 0:
                return valor

            print("C deve ser maior que zero.")

        except ValueError:
            print("Digite um número válido.")


# ============================================================
# CONFIGURAÇÃO DO KERNEL
# ============================================================

def configurar_kernel(valor_atual):
    print("\nKernel")
    print(f"Valor atual: '{valor_atual}'")
    print("1 - rbf")
    print("2 - linear")
    print("3 - poly")
    print("4 - sigmoid")

    while True:
        entrada = input(
            "Digite a opção "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        opcoes = {
            "1": "rbf",
            "2": "linear",
            "3": "poly",
            "4": "sigmoid"
        }

        if entrada in opcoes:
            return opcoes[entrada]

        print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO DO DEGREE
# ============================================================

def configurar_degree(valor_atual):
    print("\nDegree")
    print(f"Valor atual: {valor_atual}")
    print("Valores comuns: 2, 3, 4, 5")
    print("Esse parâmetro é utilizado principalmente pelo kernel 'poly'.")

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

            print("Degree deve ser um inteiro maior ou igual a 1.")

        except ValueError:
            print("Digite um número inteiro válido.")


# ============================================================
# CONFIGURAÇÃO DO GAMMA
# ============================================================

def configurar_gamma(valor_atual):
    print("\nGamma")
    print(f"Valor atual: '{valor_atual}'")
    print("Opções comuns: 'scale', 'auto' ou um valor numérico.")

    while True:
        entrada = input(
            "Digite o novo valor "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        entrada_lower = entrada.lower()

        if entrada_lower in {"scale", "auto"}:
            return entrada_lower

        try:
            valor = float(entrada)

            if valor > 0:
                return valor

            print("Gamma deve ser maior que zero.")

        except ValueError:
            print(
                "Digite 'scale', 'auto' ou um valor numérico válido."
            )


# ============================================================
# CONFIGURAÇÃO DO COEF0
# ============================================================

def configurar_coef0(valor_atual):
    print("\nCoef0")
    print(f"Valor atual: {valor_atual}")
    print("Esse parâmetro é utilizado principalmente pelos kernels 'poly' e 'sigmoid'.")

    while True:
        entrada = input(
            "Digite o novo valor "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        try:
            return float(entrada)

        except ValueError:
            print("Digite um número válido.")


# ============================================================
# CONFIGURAÇÃO DO SHRINKING
# ============================================================

def configurar_shrinking(valor_atual):
    print("\nShrinking")
    print(f"Valor atual: {valor_atual}")
    print("1 - True")
    print("2 - False")

    while True:
        entrada = input(
            "Digite a opção "
            "(pressione ENTER para manter): "
        ).strip()

        if entrada == "":
            return valor_atual

        if entrada == "1":
            return True

        if entrada == "2":
            return False

        print("Opção inválida.")


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
        "kernel": "rbf",
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
        "shrinking": True,
        "class_weight": None
    }

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DO SVM")
    print("=" * 60)

    print("\nModelo: SVC")

    print("\nParâmetros atuais:")
    print(f"C            = {parametros['C']}")
    print(f"kernel       = '{parametros['kernel']}'")
    print(f"degree       = {parametros['degree']}")
    print(f"gamma        = '{parametros['gamma']}'")
    print(f"coef0        = {parametros['coef0']}")
    print(f"shrinking    = {parametros['shrinking']}")
    print(f"class_weight = {parametros['class_weight']}")

    print(
        "\nObservação: o parâmetro 'probability' não é utilizado "
        "nesta versão para evitar o FutureWarning do scikit-learn."
    )

    while True:
        print("\nDeseja utilizar essa configuração?")
        print("1 - Sim, utilizar padrão")
        print("2 - Não, modificar parâmetros")

        opcao = input("\nDigite a opção: ").strip()

        if opcao == "1":
            return parametros

        if opcao == "2":
            break

        print("Opção inválida.")

    while True:
        print("\n" + "=" * 60)
        print("PARÂMETROS DO SVM")
        print("=" * 60)

        print(f"1 - C            = {parametros['C']}")
        print(f"2 - kernel       = '{parametros['kernel']}'")
        print(f"3 - degree       = {parametros['degree']}")
        print(f"4 - gamma        = '{parametros['gamma']}'")
        print(f"5 - coef0        = {parametros['coef0']}")
        print(f"6 - shrinking    = {parametros['shrinking']}")
        print(f"7 - class_weight = {parametros['class_weight']}")
        print("8 - Finalizar configuração")

        opcao = input("\nDigite a opção: ").strip()

        if opcao == "1":
            parametros["C"] = configurar_c(parametros["C"])

        elif opcao == "2":
            parametros["kernel"] = configurar_kernel(
                parametros["kernel"]
            )

        elif opcao == "3":
            parametros["degree"] = configurar_degree(
                parametros["degree"]
            )

        elif opcao == "4":
            parametros["gamma"] = configurar_gamma(
                parametros["gamma"]
            )

        elif opcao == "5":
            parametros["coef0"] = configurar_coef0(
                parametros["coef0"]
            )

        elif opcao == "6":
            parametros["shrinking"] = configurar_shrinking(
                parametros["shrinking"]
            )

        elif opcao == "7":
            parametros["class_weight"] = configurar_class_weight(
                parametros["class_weight"]
            )

        elif opcao == "8":
            break

        else:
            print("Opção inválida.")

    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO FINAL DO SVM")
    print("=" * 60)

    for chave, valor in parametros.items():
        if isinstance(valor, str):
            print(f"{chave:<15} = '{valor}'")
        else:
            print(f"{chave:<15} = {valor}")

    return parametros


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def criar_modelo(parametros):
    return SVC(
        C=parametros["C"],
        kernel=parametros["kernel"],
        degree=parametros["degree"],
        gamma=parametros["gamma"],
        coef0=parametros["coef0"],
        shrinking=parametros["shrinking"],
        class_weight=parametros["class_weight"],
        random_state=RANDOM_STATE
    )


# ============================================================
# VALIDAÇÃO DO DATASET
# ============================================================

def validar_dataset(df, coluna_alvo):
    problemas = []

    if coluna_alvo not in df.columns:
        problemas.append(
            f"A variável alvo '{coluna_alvo}' não existe no dataset."
        )
        return problemas

    if df.empty:
        problemas.append("O dataset está vazio.")

    if df[coluna_alvo].isna().any():
        quantidade = df[coluna_alvo].isna().sum()

        problemas.append(
            f"A variável alvo possui {quantidade} valor(es) ausente(s)."
        )

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    for coluna in colunas_preditoras:

        if df[coluna].isna().any():
            quantidade = df[coluna].isna().sum()

            problemas.append(
                f"O preditor '{coluna}' possui "
                f"{quantidade} valor(es) ausente(s)."
            )

        if not pd.api.types.is_numeric_dtype(df[coluna]):
            problemas.append(
                f"O preditor '{coluna}' possui tipo "
                f"'{df[coluna].dtype}' e não é numérico."
            )

    if not problemas:

        quantidade_classes = df[coluna_alvo].nunique()

        if quantidade_classes < 2:
            problemas.append(
                "A variável alvo precisa possuir pelo menos "
                "2 classes diferentes."
            )

    return problemas


# ============================================================
# PREPARAÇÃO DA VARIÁVEL ALVO
# ============================================================

def preparar_alvo(df, coluna_alvo, mapeamento=None):
    serie = df[coluna_alvo]

    if pd.api.types.is_numeric_dtype(serie):
        return serie.copy(), None

    categorias = pd.Series(serie).dropna().unique().tolist()

    if mapeamento is None:
        categorias = sorted(
            categorias,
            key=lambda valor: str(valor)
        )

        mapeamento = {
            categoria: indice
            for indice, categoria in enumerate(categorias)
        }

    valores_desconhecidos = [
        valor
        for valor in serie.dropna().unique()
        if valor not in mapeamento
    ]

    if valores_desconhecidos:
        raise ValueError(
            "Foram encontradas categorias no alvo que não existem "
            f"no mapeamento de treinamento: {valores_desconhecidos}"
        )

    alvo = serie.map(mapeamento)

    if alvo.isna().any():
        raise ValueError(
            "Não foi possível converter todos os valores "
            "da variável alvo."
        )

    alvo = alvo.astype(int)

    return alvo, mapeamento


# ============================================================
# PREPARAÇÃO DOS PREDITORES
# ============================================================

def preparar_preditores(df, coluna_alvo):
    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != coluna_alvo
    ]

    X = df[colunas_preditoras].copy()

    for coluna in colunas_preditoras:
        if not pd.api.types.is_numeric_dtype(X[coluna]):
            raise ValueError(
                f"O preditor '{coluna}' não é numérico."
            )

    return X, colunas_preditoras


# ============================================================
# CÁLCULO DAS MÉTRICAS
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
# AVALIAÇÃO TEXTUAL
# ============================================================

def gerar_avaliacao_textual(y_real, y_pred):
    matriz = confusion_matrix(y_real, y_pred)

    relatorio = classification_report(
        y_real,
        y_pred,
        zero_division=0
    )

    return matriz, relatorio


# ============================================================
# AUTO TESTE
# ============================================================

def executar_auto_teste(df, coluna_alvo, parametros):
    print("\n" + "=" * 60)
    print("EXECUTANDO AUTO TESTE")
    print("=" * 60)

    X, colunas_preditoras = preparar_preditores(
        df,
        coluna_alvo
    )

    y, mapeamento = preparar_alvo(
        df,
        coluna_alvo
    )

    modelo = criar_modelo(parametros)

    modelo.fit(X, y)

    y_pred = modelo.predict(X)

    metricas = calcular_metricas(y, y_pred)

    matriz, relatorio = gerar_avaliacao_textual(
        y,
        y_pred
    )

    print("\nAuto Teste concluído.")

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
    print("EXECUTANDO HOLD OUT")
    print("=" * 60)

    # --------------------------------------------------------
    # CASO 1: DATASET ÚNICO
    # --------------------------------------------------------

    if df_teste is None:

        X, colunas_preditoras = preparar_preditores(
            df_treino,
            coluna_alvo
        )

        y, mapeamento = preparar_alvo(
            df_treino,
            coluna_alvo
        )

        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X,
            y,
            test_size=1 - percentual_treino,
            random_state=RANDOM_STATE,
            stratify=y
        )

    # --------------------------------------------------------
    # CASO 2: DATASETS SEPARADOS
    # --------------------------------------------------------

    else:

        X_treino, colunas_preditoras_treino = preparar_preditores(
            df_treino,
            coluna_alvo
        )

        y_treino, mapeamento = preparar_alvo(
            df_treino,
            coluna_alvo
        )

        # ----------------------------------------------------
        # Verificação das colunas
        # ----------------------------------------------------

        colunas_preditoras_teste = [
            coluna
            for coluna in df_teste.columns
            if coluna != coluna_alvo
        ]

        if set(colunas_preditoras_treino) != set(
            colunas_preditoras_teste
        ):
            raise ValueError(
                "Os datasets de treino e teste não possuem "
                "as mesmas variáveis preditoras."
            )

        # ----------------------------------------------------
        # IMPORTANTE:
        # O teste passa a usar exatamente a mesma ordem
        # de colunas utilizada durante o treinamento.
        # ----------------------------------------------------

        X_teste = df_teste[
            colunas_preditoras_treino
        ].copy()

        for coluna in colunas_preditoras_treino:
            if not pd.api.types.is_numeric_dtype(
                X_teste[coluna]
            ):
                raise ValueError(
                    f"O preditor '{coluna}' no dataset de teste "
                    "não é numérico."
                )

        # ----------------------------------------------------
        # Preparação do alvo do teste
        # ----------------------------------------------------

        y_teste, _ = preparar_alvo(
            df_teste,
            coluna_alvo,
            mapeamento
        )

        colunas_preditoras = colunas_preditoras_treino

    # ========================================================
    # TREINAMENTO
    # ========================================================

    modelo = criar_modelo(parametros)

    modelo.fit(
        X_treino,
        y_treino
    )

    # ========================================================
    # PREDIÇÃO
    # ========================================================

    y_pred = modelo.predict(X_teste)

    # ========================================================
    # MÉTRICAS
    # ========================================================

    metricas = calcular_metricas(
        y_teste,
        y_pred
    )

    matriz, relatorio = gerar_avaliacao_textual(
        y_teste,
        y_pred
    )

    print("\nHold Out concluído.")

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
    print("EXECUTANDO CROSS-VALIDATION")
    print("=" * 60)

    X, colunas_preditoras = preparar_preditores(
        df,
        coluna_alvo
    )

    y, mapeamento = preparar_alvo(
        df,
        coluna_alvo
    )

    cross_validation = StratifiedKFold(
        n_splits=quantidade_folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados_folds = []

    for numero_fold, (
        indices_treino,
        indices_teste
    ) in enumerate(
        cross_validation.split(X, y),
        start=1
    ):

        X_treino = X.iloc[indices_treino]
        X_teste = X.iloc[indices_teste]

        y_treino = y.iloc[indices_treino]
        y_teste = y.iloc[indices_teste]

        modelo = criar_modelo(parametros)

        modelo.fit(
            X_treino,
            y_treino
        )

        y_pred = modelo.predict(X_teste)

        metricas = calcular_metricas(
            y_teste,
            y_pred
        )

        matriz, relatorio = gerar_avaliacao_textual(
            y_teste,
            y_pred
        )

        resultados_folds.append({
            "fold": numero_fold,
            "metricas": metricas,
            "matriz": matriz,
            "relatorio": relatorio
        })

        print(
            f"Fold {numero_fold}/{quantidade_folds} concluído."
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
            "media": np.mean(valores),
            "desvio_padrao": np.std(
                valores,
                ddof=1
            ) if len(valores) > 1 else 0.0
        }

    print("\nCross-validation concluído.")

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
    print("MÉTODO DE AVALIAÇÃO")
    print("=" * 60)

    print("1 - Auto Teste")
    print("2 - Hold Out")
    print("3 - Cross-validation")

    while True:
        opcao = input("\nDigite a opção: ").strip()

        if opcao in {"1", "2", "3"}:
            return opcao

        print("Opção inválida.")


# ============================================================
# CONFIGURAÇÃO DO HOLD OUT
# ============================================================

def configurar_holdout():
    print("\n" + "=" * 60)
    print("CONFIGURAÇÃO DO HOLD OUT")
    print("=" * 60)

    print("Os dados de treino e teste já estão separados?")
    print("1 - Sim")
    print("2 - Não")

    while True:
        opcao = input("\nDigite a opção: ").strip()

        if opcao in {"1", "2"}:
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

            if 50 <= percentual < 100:
                return percentual / 100

            print(
                "Digite um percentual entre 50 e 99."
            )

        except ValueError:
            print("Digite um número válido.")


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
            folds = int(entrada)

            if folds >= 2:
                return folds

            print(
                "A quantidade de folds deve ser "
                "maior ou igual a 2."
            )

        except ValueError:
            print("Digite um número inteiro válido.")


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

    linhas.append("=" * 70)
    linhas.append("SVM - CLASSIFICAÇÃO")
    linhas.append("=" * 70)

    linhas.append("")
    linhas.append(f"Dataset: {caminho_dataset.name}")
    linhas.append(f"Método: {metodo}")
    linhas.append(f"Variável alvo: {coluna_alvo}")

    linhas.append("")
    linhas.append("=" * 70)
    linhas.append("CONFIGURAÇÃO DO MODELO")
    linhas.append("=" * 70)

    linhas.append("Modelo: SVC")

    for chave, valor in parametros.items():

        if isinstance(valor, str):
            linhas.append(
                f"{chave:<15} = '{valor}'"
            )
        else:
            linhas.append(
                f"{chave:<15} = {valor}"
            )

    linhas.append(
        f"{'random_state':<15} = {RANDOM_STATE}"
    )

    linhas.append("")
    linhas.append(
        "Observação: probability não é utilizado, "
        "pois o parâmetro foi depreciado nas versões "
        "recentes do scikit-learn."
    )

    # ========================================================
    # AUTO TESTE / HOLD OUT
    # ========================================================

    if metodo in {
        "Auto Teste",
        "Hold Out"
    }:

        linhas.append("")
        linhas.append("=" * 70)
        linhas.append("MÉTRICAS")
        linhas.append("=" * 70)

        metricas = resultado["metricas"]

        linhas.append(
            f"Acurácia:           {metricas['accuracy']:.4f}"
        )

        linhas.append(
            f"Precisão ponderada: {metricas['precision']:.4f}"
        )

        linhas.append(
            f"Recall ponderado:   {metricas['recall']:.4f}"
        )

        linhas.append(
            f"F1 ponderado:       {metricas['f1']:.4f}"
        )

        linhas.append("")
        linhas.append("=" * 70)
        linhas.append("MATRIZ DE CONFUSÃO")
        linhas.append("=" * 70)

        matriz = resultado["matriz"]

        linhas.append(
            np.array2string(matriz)
        )

        linhas.append("")
        linhas.append("=" * 70)
        linhas.append("CLASSIFICATION REPORT")
        linhas.append("=" * 70)

        linhas.append(
            resultado["relatorio"]
        )

        if resultado.get("mapeamento"):
            linhas.append("")
            linhas.append("=" * 70)
            linhas.append("MAPEAMENTO DA VARIÁVEL ALVO")
            linhas.append("=" * 70)

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
        linhas.append("=" * 70)
        linhas.append(
            f"RESULTADOS DOS {resultado['quantidade_folds']} FOLDS"
        )
        linhas.append("=" * 70)

        for fold in resultado["folds"]:

            linhas.append("")
            linhas.append(
                f"---------------- FOLD {fold['fold']} ----------------"
            )

            metricas = fold["metricas"]

            linhas.append(
                f"Acurácia:           {metricas['accuracy']:.4f}"
            )

            linhas.append(
                f"Precisão ponderada: {metricas['precision']:.4f}"
            )

            linhas.append(
                f"Recall ponderado:   {metricas['recall']:.4f}"
            )

            linhas.append(
                f"F1 ponderado:       {metricas['f1']:.4f}"
            )

            linhas.append("")
            linhas.append("Matriz de confusão:")
            linhas.append(
                np.array2string(fold["matriz"])
            )

            linhas.append("")
            linhas.append("Classification report:")
            linhas.append(
                fold["relatorio"]
            )

        # ====================================================
        # RESULTADO FINAL
        # ====================================================

        linhas.append("")
        linhas.append("=" * 70)
        linhas.append("RESULTADO FINAL DA CROSS-VALIDATION")
        linhas.append("=" * 70)

        metricas_finais = resultado["metricas_finais"]

        for metrica, valores in metricas_finais.items():

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

        if resultado.get("mapeamento"):
            linhas.append("")
            linhas.append("=" * 70)
            linhas.append("MAPEAMENTO DA VARIÁVEL ALVO")
            linhas.append("=" * 70)

            for categoria, codigo in (
                resultado["mapeamento"].items()
            ):
                linhas.append(
                    f"{categoria} -> {codigo}"
                )

    linhas.append("")
    linhas.append("=" * 70)
    linhas.append("FIM DO RELATÓRIO")
    linhas.append("=" * 70)

    nome_saida = (
        f"015_SVM_{caminho_dataset.stem}.txt"
    )

    caminho_saida = PASTA_SAIDA / nome_saida

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
    print("SVM - CLASSIFICAÇÃO")
    print("=" * 60)

    try:

        # ----------------------------------------------------
        # MÉTODO
        # ----------------------------------------------------

        metodo = selecionar_metodo()

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        caminho, df = selecionar_arquivo()

        # ----------------------------------------------------
        # HOLD OUT COM DATASETS SEPARADOS
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
        # VALIDAÇÃO DO DATASET PRINCIPAL
        # ----------------------------------------------------

        problemas = validar_dataset(
            df,
            coluna_alvo
        )

        if problemas:

            print("\n" + "=" * 60)
            print("ERROS ENCONTRADOS")
            print("=" * 60)

            for problema in problemas:
                print(f"- {problema}")

            print(
                "\nO processamento foi interrompido."
            )

            return

        # ----------------------------------------------------
        # CONFIGURAÇÃO DO MÉTODO
        # ----------------------------------------------------

        if metodo == "2":

            configuracao = configurar_holdout()

            if configuracao == "1":

                print(
                    "\nSelecione o dataset de teste."
                )

                caminho_teste, df_teste = (
                    selecionar_arquivo()
                )

                problemas_teste = validar_dataset(
                    df_teste,
                    coluna_alvo
                )

                if problemas_teste:

                    print("\n" + "=" * 60)
                    print(
                        "ERROS NO DATASET DE TESTE"
                    )
                    print("=" * 60)

                    for problema in problemas_teste:
                        print(f"- {problema}")

                    return

                if coluna_alvo not in df_teste.columns:
                    print(
                        f"\nA variável alvo '{coluna_alvo}' "
                        "não existe no dataset de teste."
                    )
                    return

            else:

                percentual_treino = (
                    solicitar_percentual_treino()
                )

        # ----------------------------------------------------
        # CONFIGURAÇÃO DO MODELO
        # ----------------------------------------------------

        parametros = configurar_modelo()

        # ----------------------------------------------------
        # EXECUÇÃO
        # ----------------------------------------------------

        if metodo == "1":

            resultado = executar_auto_teste(
                df,
                coluna_alvo,
                parametros
            )

            nome_metodo = "Auto Teste"

        elif metodo == "2":

            resultado = executar_holdout(
                df_treino=df,
                df_teste=df_teste,
                coluna_alvo=coluna_alvo,
                parametros=parametros,
                percentual_treino=percentual_treino
            )

            nome_metodo = "Hold Out"

        else:

            quantidade_folds = solicitar_folds()

            resultado = executar_cross_validation(
                df,
                coluna_alvo,
                parametros,
                quantidade_folds
            )

            nome_metodo = "Cross-validation"

        # ----------------------------------------------------
        # RELATÓRIO
        # ----------------------------------------------------

        caminho_saida = criar_relatorio(
            caminho,
            nome_metodo,
            coluna_alvo,
            parametros,
            resultado
        )

        print("\n" + "=" * 60)
        print("PROCESSAMENTO CONCLUÍDO")
        print("=" * 60)

        print("\nRelatório salvo em:")
        print(caminho_saida)

    except Exception as erro:

        print("\n" + "=" * 60)
        print("ERRO DURANTE O PROCESSAMENTO")
        print("=" * 60)

        print(f"\n{erro}")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()