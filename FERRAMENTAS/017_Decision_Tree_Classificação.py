# ============================================================
# ÁRVORE DE DECISÃO - CLASSIFICAÇÃO
# ============================================================

from pathlib import Path
from utilidades import ler_arquivo
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
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

EXTENSOES_PERMITIDAS = [
    ".csv",
    ".xlsx",
    ".xls",
    ".json",
    ".parquet"
]


# ============================================================
# LEITURA DE ARQUIVOS
# ============================================================

def listar_arquivos():
    """Lista os arquivos compatíveis encontrados em ENTRADA."""

    if not PASTA_ENTRADA.exists():
        return []

    arquivos = sorted(
        [
            arquivo
            for arquivo in PASTA_ENTRADA.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in EXTENSOES_PERMITIDAS
        ]
    )

    return arquivos


def selecionar_arquivo(mensagem):
    """Permite ao usuário selecionar um arquivo."""

    arquivos = listar_arquivos()

    if not arquivos:
        print("\nNenhum arquivo compatível encontrado em ENTRADA.")
        return None

    print("\n" + "=" * 70)
    print(mensagem)
    print("=" * 70)

    for indice, arquivo in enumerate(arquivos, start=1):
        print(f"{indice}. {arquivo.name}")

    while True:

        escolha = input(
            "\nDigite o número do arquivo: "
        ).strip()

        try:

            numero = int(escolha)

            if 1 <= numero <= len(arquivos):
                return arquivos[numero - 1]

            print("Número inválido.")

        except ValueError:

            print(
                "Digite apenas o número correspondente ao arquivo."
            )


# ============================================================
# SELEÇÃO DA VARIÁVEL TARGET
# ============================================================

def selecionar_target(df):
    """Permite selecionar a variável alvo."""

    print("\n" + "=" * 70)
    print("SELEÇÃO DA VARIÁVEL ALVO")
    print("=" * 70)

    for indice, coluna in enumerate(df.columns, start=1):

        print(
            f"{indice}. {coluna} "
            f"({df[coluna].dtype})"
        )

    while True:

        escolha = input(
            "\nDigite o número da coluna que será a variável alvo: "
        ).strip()

        try:

            numero = int(escolha)

            if 1 <= numero <= len(df.columns):

                target = df.columns[numero - 1]

                print(
                    f"\nVariável alvo selecionada: {target}"
                )

                return target

            print("Número inválido.")

        except ValueError:

            print("Digite um número inteiro.")


# ============================================================
# VALIDAÇÃO DE VALORES AUSENTES
# ============================================================

def validar_valores_ausentes(df, nome_dataset):
    """Interrompe a execução caso existam valores ausentes."""

    ausentes = df.isna().sum()

    ausentes = ausentes[
        ausentes > 0
    ]

    if ausentes.empty:
        return

    print("\n" + "=" * 70)
    print(
        f"VALORES AUSENTES - {nome_dataset}"
    )
    print("=" * 70)

    for coluna, quantidade in ausentes.items():

        percentual = (
            quantidade / len(df)
        ) * 100

        print(
            f"- {coluna}: "
            f"{quantidade} "
            f"({percentual:.2f}%)"
        )

    raise ValueError(
        f"\nO dataset {nome_dataset} possui valores ausentes. "
        "Realize o tratamento dos valores ausentes antes "
        "da classificação."
    )


# ============================================================
# VALIDAÇÃO DOS PREDITORES
# ============================================================

def validar_preditores_numericos(
    df,
    target,
    nome_dataset
):
    """Verifica se todos os preditores são numéricos."""

    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != target
    ]

    if not colunas_preditoras:

        raise ValueError(
            "O dataset não possui nenhuma variável preditora."
        )

    nao_numericas = []

    for coluna in colunas_preditoras:

        if not pd.api.types.is_numeric_dtype(
            df[coluna]
        ):
            nao_numericas.append(coluna)

    if nao_numericas:

        print("\n" + "=" * 70)
        print(
            f"PREDITORES INCOMPATÍVEIS - {nome_dataset}"
        )
        print("=" * 70)

        for coluna in nao_numericas:

            print(
                f"- {coluna}: "
                f"{df[coluna].dtype}"
            )

        raise ValueError(
            "\nTodos os preditores precisam ser numéricos. "
            "O programa não realiza codificação automática "
            "das variáveis preditoras."
        )

    return colunas_preditoras


# ============================================================
# PREPARAÇÃO DO TARGET
# ============================================================

def preparar_target_treino(df, target):
    """
    Prepara o target do dataset de treinamento.

    Target numérico permanece numérico.
    Target categórico recebe códigos numéricos.
    """

    serie = df[target]

    if pd.api.types.is_numeric_dtype(serie):

        y = serie.copy()

        mapeamento = None

    else:

        categorias = sorted(
            serie.astype(str).unique()
        )

        mapeamento = {
            categoria: indice
            for indice, categoria
            in enumerate(categorias)
        }

        print("\n" + "=" * 70)
        print("MAPEAMENTO DA VARIÁVEL ALVO")
        print("=" * 70)

        for categoria, codigo in mapeamento.items():

            print(
                f"{categoria} -> {codigo}"
            )

        y = (
            serie
            .astype(str)
            .map(mapeamento)
        )

    if y.nunique() < 2:

        raise ValueError(
            "A variável alvo precisa possuir pelo menos "
            "duas classes diferentes."
        )

    return y, mapeamento


def preparar_target_teste(
    df,
    target,
    mapeamento
):
    """
    Prepara o target do dataset de teste.

    Quando o target de treino foi categórico,
    utiliza exatamente o mesmo mapeamento.
    """

    serie = df[target]

    if mapeamento is None:

        if not pd.api.types.is_numeric_dtype(serie):

            raise ValueError(
                "O target do teste não é numérico, "
                "mas o target do treino é numérico."
            )

        y = serie.copy()

        valores_treino = set(
            mapeamento.keys()
        ) if mapeamento else None

    else:

        categorias_teste = set(
            serie.astype(str).unique()
        )

        categorias_treino = set(
            mapeamento.keys()
        )

        desconhecidas = (
            categorias_teste
            - categorias_treino
        )

        if desconhecidas:

            raise ValueError(
                "O dataset de teste possui categorias "
                "de target que não existem no dataset "
                "de treinamento:\n"
                + "\n".join(
                    f"- {categoria}"
                    for categoria in sorted(desconhecidas)
                )
            )

        y = (
            serie
            .astype(str)
            .map(mapeamento)
        )

    return y


# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

def mostrar_configuracao(parametros):
    """Exibe a configuração atual."""

    print(
        f"criterion       = '{parametros['criterion']}'"
    )

    print(
        f"splitter        = '{parametros['splitter']}'"
    )

    print(
        f"max_depth       = {parametros['max_depth']}"
    )

    print(
        f"min_samples_split = "
        f"{parametros['min_samples_split']}"
    )

    print(
        f"min_samples_leaf  = "
        f"{parametros['min_samples_leaf']}"
    )

    print(
        f"max_features    = "
        f"{parametros['max_features']}"
    )

    print(
        f"max_leaf_nodes  = "
        f"{parametros['max_leaf_nodes']}"
    )

    print(
        f"min_impurity_decrease = "
        f"{parametros['min_impurity_decrease']}"
    )

    print(
        f"class_weight    = "
        f"{parametros['class_weight']}"
    )

    print(
        f"ccp_alpha       = "
        f"{parametros['ccp_alpha']}"
    )

    print(
        f"random_state    = "
        f"{RANDOM_STATE}"
    )


def configurar_modelo():
    """Configura os parâmetros da Decision Tree."""

    parametros = {

        "criterion": "gini",

        "splitter": "best",

        "max_depth": None,

        "min_samples_split": 2,

        "min_samples_leaf": 1,

        "max_features": None,

        "max_leaf_nodes": None,

        "min_impurity_decrease": 0.0,

        "class_weight": None,

        "ccp_alpha": 0.0
    }

    print("\n" + "=" * 70)
    print("CONFIGURAÇÃO DO MODELO")
    print("=" * 70)

    print("\nModelo: DecisionTreeClassifier")

    print("\nParâmetros padrão:")

    mostrar_configuracao(parametros)

    print("\n" + "-" * 70)
    print("Deseja utilizar os parâmetros padrão?")
    print("1. Sim")
    print("2. Não")

    while True:

        escolha = input(
            "\nEscolha: "
        ).strip()

        if escolha in ["1", "2"]:
            break

        print("Opção inválida.")

    if escolha == "1":
        return parametros

    while True:

        print("\n" + "=" * 70)
        print("ALTERAÇÃO DE PARÂMETROS")
        print("=" * 70)

        print("1. criterion")
        print("2. splitter")
        print("3. max_depth")
        print("4. min_samples_split")
        print("5. min_samples_leaf")
        print("6. max_features")
        print("7. max_leaf_nodes")
        print("8. min_impurity_decrease")
        print("9. class_weight")
        print("10. ccp_alpha")
        print("0. Finalizar")

        opcao = input(
            "\nEscolha: "
        ).strip()

        # ----------------------------------------------------
        # CRITERION
        # ----------------------------------------------------

        if opcao == "1":

            print("\ncriterion atual:")
            print(
                parametros["criterion"]
            )

            print("\n1. gini")
            print("2. entropy")
            print("3. log_loss")

            escolha = input(
                "Escolha: "
            ).strip()

            opcoes = {
                "1": "gini",
                "2": "entropy",
                "3": "log_loss"
            }

            if escolha in opcoes:

                parametros["criterion"] = (
                    opcoes[escolha]
                )

            else:
                print("Opção inválida.")

        # ----------------------------------------------------
        # SPLITTER
        # ----------------------------------------------------

        elif opcao == "2":

            print("\nsplitter atual:")
            print(
                parametros["splitter"]
            )

            print("\n1. best")
            print("2. random")

            escolha = input(
                "Escolha: "
            ).strip()

            if escolha == "1":

                parametros["splitter"] = "best"

            elif escolha == "2":

                parametros["splitter"] = "random"

            else:

                print("Opção inválida.")

        # ----------------------------------------------------
        # MAX DEPTH
        # ----------------------------------------------------

        elif opcao == "3":

            print("\nmax_depth atual:")
            print(
                parametros["max_depth"]
            )

            print("\n1. None")
            print("2. 3")
            print("3. 5")
            print("4. 10")
            print("5. Personalizado")

            escolha = input(
                "Escolha: "
            ).strip()

            if escolha == "1":

                parametros["max_depth"] = None

            elif escolha == "2":

                parametros["max_depth"] = 3

            elif escolha == "3":

                parametros["max_depth"] = 5

            elif escolha == "4":

                parametros["max_depth"] = 10

            elif escolha == "5":

                try:

                    valor = int(
                        input(
                            "Digite o valor: "
                        )
                    )

                    if valor > 0:

                        parametros["max_depth"] = valor

                    else:

                        print(
                            "O valor precisa ser maior que zero."
                        )

                except ValueError:

                    print(
                        "Digite um número inteiro."
                    )

            else:

                print("Opção inválida.")

        # ----------------------------------------------------
        # MIN SAMPLES SPLIT
        # ----------------------------------------------------

        elif opcao == "4":

            print(
                "\nmin_samples_split atual:",
                parametros["min_samples_split"]
            )

            print(
                "Valores comuns: 2, 5, 10"
            )

            try:

                valor = int(
                    input(
                        "Digite o novo valor: "
                    )
                )

                if valor >= 2:

                    parametros[
                        "min_samples_split"
                    ] = valor

                else:

                    print(
                        "O valor precisa ser pelo menos 2."
                    )

            except ValueError:

                print(
                    "Digite um número inteiro."
                )

        # ----------------------------------------------------
        # MIN SAMPLES LEAF
        # ----------------------------------------------------

        elif opcao == "5":

            print(
                "\nmin_samples_leaf atual:",
                parametros["min_samples_leaf"]
            )

            print(
                "Valores comuns: 1, 2, 5, 10"
            )

            try:

                valor = int(
                    input(
                        "Digite o novo valor: "
                    )
                )

                if valor >= 1:

                    parametros[
                        "min_samples_leaf"
                    ] = valor

                else:

                    print(
                        "O valor precisa ser pelo menos 1."
                    )

            except ValueError:

                print(
                    "Digite um número inteiro."
                )

        # ----------------------------------------------------
        # MAX FEATURES
        # ----------------------------------------------------

        elif opcao == "6":

            print(
                "\nmax_features atual:",
                parametros["max_features"]
            )

            print("\n1. None")
            print("2. sqrt")
            print("3. log2")
            print("4. Inteiro personalizado")

            escolha = input(
                "Escolha: "
            ).strip()

            if escolha == "1":

                parametros["max_features"] = None

            elif escolha == "2":

                parametros["max_features"] = "sqrt"

            elif escolha == "3":

                parametros["max_features"] = "log2"

            elif escolha == "4":

                try:

                    valor = int(
                        input(
                            "Digite a quantidade de features: "
                        )
                    )

                    if valor >= 1:

                        parametros[
                            "max_features"
                        ] = valor

                    else:

                        print(
                            "O valor precisa ser pelo menos 1."
                        )

                except ValueError:

                    print(
                        "Digite um número inteiro."
                    )

            else:

                print("Opção inválida.")

        # ----------------------------------------------------
        # MAX LEAF NODES
        # ----------------------------------------------------

        elif opcao == "7":

            print(
                "\nmax_leaf_nodes atual:",
                parametros["max_leaf_nodes"]
            )

            print("\n1. None")
            print("2. Personalizado")

            escolha = input(
                "Escolha: "
            ).strip()

            if escolha == "1":

                parametros["max_leaf_nodes"] = None

            elif escolha == "2":

                try:

                    valor = int(
                        input(
                            "Digite o valor: "
                        )
                    )

                    if valor >= 1:

                        parametros[
                            "max_leaf_nodes"
                        ] = valor

                    else:

                        print(
                            "O valor precisa ser pelo menos 1."
                        )

                except ValueError:

                    print(
                        "Digite um número inteiro."
                    )

            else:

                print("Opção inválida.")

        # ----------------------------------------------------
        # MIN IMPURITY DECREASE
        # ----------------------------------------------------

        elif opcao == "8":

            print(
                "\nmin_impurity_decrease atual:",
                parametros[
                    "min_impurity_decrease"
                ]
            )

            try:

                valor = float(
                    input(
                        "Digite o novo valor: "
                    )
                )

                if valor >= 0:

                    parametros[
                        "min_impurity_decrease"
                    ] = valor

                else:

                    print(
                        "O valor não pode ser negativo."
                    )

            except ValueError:

                print(
                    "Digite um número válido."
                )

        # ----------------------------------------------------
        # CLASS WEIGHT
        # ----------------------------------------------------

        elif opcao == "9":

            print(
                "\nclass_weight atual:",
                parametros["class_weight"]
            )

            print("\n1. None")
            print("2. balanced")

            escolha = input(
                "Escolha: "
            ).strip()

            if escolha == "1":

                parametros["class_weight"] = None

            elif escolha == "2":

                parametros["class_weight"] = "balanced"

            else:

                print("Opção inválida.")

        # ----------------------------------------------------
        # CCP ALPHA
        # ----------------------------------------------------

        elif opcao == "10":

            print(
                "\nccp_alpha atual:",
                parametros["ccp_alpha"]
            )

            try:

                valor = float(
                    input(
                        "Digite o novo valor: "
                    )
                )

                if valor >= 0:

                    parametros["ccp_alpha"] = valor

                else:

                    print(
                        "O valor não pode ser negativo."
                    )

            except ValueError:

                print(
                    "Digite um número válido."
                )

        # ----------------------------------------------------
        # FINALIZAR
        # ----------------------------------------------------

        elif opcao == "0":

            break

        else:

            print("Opção inválida.")

    print("\n" + "=" * 70)
    print("CONFIGURAÇÃO FINAL DO MODELO")
    print("=" * 70)

    print("\nModelo: DecisionTreeClassifier")

    mostrar_configuracao(parametros)

    return parametros


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def criar_modelo(parametros):

    return DecisionTreeClassifier(
        criterion=parametros["criterion"],
        splitter=parametros["splitter"],
        max_depth=parametros["max_depth"],
        min_samples_split=parametros["min_samples_split"],
        min_samples_leaf=parametros["min_samples_leaf"],
        max_features=parametros["max_features"],
        max_leaf_nodes=parametros["max_leaf_nodes"],
        min_impurity_decrease=parametros[
            "min_impurity_decrease"
        ],
        class_weight=parametros["class_weight"],
        ccp_alpha=parametros["ccp_alpha"],
        random_state=RANDOM_STATE
    )


# ============================================================
# CÁLCULO DAS MÉTRICAS
# ============================================================

def calcular_metricas(y_real, y_pred):

    acuracia = accuracy_score(
        y_real,
        y_pred
    )

    precisao = precision_score(
        y_real,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_real,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_real,
        y_pred,
        average="weighted",
        zero_division=0
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

    return {
        "accuracy": acuracia,
        "precision": precisao,
        "recall": recall,
        "f1": f1
    }, matriz, relatorio


# ============================================================
# EXECUÇÃO DO AUTO TESTE
# ============================================================

def executar_auto_teste(
    df,
    target,
    parametros
):

    validar_valores_ausentes(
        df,
        "DATASET"
    )

    colunas_preditoras = (
        validar_preditores_numericos(
            df,
            target,
            "DATASET"
        )
    )

    y, mapeamento = preparar_target_treino(
        df,
        target
    )

    X = df[
        colunas_preditoras
    ].copy()

    modelo = criar_modelo(
        parametros
    )

    modelo.fit(
        X,
        y
    )

    predicoes = modelo.predict(
        X
    )

    metricas, matriz, relatorio = (
        calcular_metricas(
            y,
            predicoes
        )
    )

    return {
        "metricas": metricas,
        "matriz": matriz,
        "relatorio": relatorio,
        "treino": len(X),
        "teste": len(X),
        "mapeamento": mapeamento
    }


# ============================================================
# EXECUÇÃO DO HOLD OUT
# ============================================================

def executar_hold_out(
    df,
    target,
    parametros
):

    print("\n" + "=" * 70)
    print("HOLD OUT")
    print("=" * 70)

    print(
        "\nOs datasets de treino e teste já estão separados?"
    )

    print("1. Sim")
    print("2. Não")

    while True:

        escolha = input(
            "\nEscolha: "
        ).strip()

        if escolha in ["1", "2"]:
            break

        print("Opção inválida.")

    # --------------------------------------------------------
    # DATASET ÚNICO
    # --------------------------------------------------------

    if escolha == "2":

        validar_valores_ausentes(
            df,
            "DATASET"
        )

        colunas_preditoras = (
            validar_preditores_numericos(
                df,
                target,
                "DATASET"
            )
        )

        y, mapeamento = preparar_target_treino(
            df,
            target
        )

        X = df[
            colunas_preditoras
        ].copy()

        print(
            "\nDigite a porcentagem dos dados "
            "que será utilizada para treinamento."
        )

        while True:

            try:

                percentual = float(
                    input(
                        "Porcentagem de treino: "
                    )
                )

                if 1 <= percentual <= 99:
                    break

                print(
                    "Digite um valor entre 1 e 99."
                )

            except ValueError:

                print(
                    "Digite um número válido."
                )

        X_treino, X_teste, y_treino, y_teste = (
            train_test_split(
                X,
                y,
                train_size=percentual / 100,
                random_state=RANDOM_STATE,
                stratify=y
            )
        )

        modelo = criar_modelo(
            parametros
        )

        modelo.fit(
            X_treino,
            y_treino
        )

        predicoes = modelo.predict(
            X_teste
        )

        metricas, matriz, relatorio = (
            calcular_metricas(
                y_teste,
                predicoes
            )
        )

        return {
            "metricas": metricas,
            "matriz": matriz,
            "relatorio": relatorio,
            "treino": len(X_treino),
            "teste": len(X_teste),
            "mapeamento": mapeamento
        }

    # --------------------------------------------------------
    # DOIS DATASETS
    # --------------------------------------------------------

    caminho_treino = selecionar_arquivo(
        "SELECIONE O DATASET DE TREINO"
    )

    if caminho_treino is None:

        raise ValueError(
            "Dataset de treino não selecionado."
        )

    caminho_teste = selecionar_arquivo(
        "SELECIONE O DATASET DE TESTE"
    )

    if caminho_teste is None:

        raise ValueError(
            "Dataset de teste não selecionado."
        )

    df_treino = ler_arquivo(
        caminho_treino
    )

    df_teste = ler_arquivo(
        caminho_teste
    )

    if target not in df_treino.columns:

        raise ValueError(
            f"A variável alvo '{target}' "
            "não existe no dataset de treino."
        )

    if target not in df_teste.columns:

        raise ValueError(
            f"A variável alvo '{target}' "
            "não existe no dataset de teste."
        )

    validar_valores_ausentes(
        df_treino,
        "DATASET DE TREINO"
    )

    validar_valores_ausentes(
        df_teste,
        "DATASET DE TESTE"
    )

    colunas_treino = (
        validar_preditores_numericos(
            df_treino,
            target,
            "DATASET DE TREINO"
        )
    )

    colunas_teste = (
        validar_preditores_numericos(
            df_teste,
            target,
            "DATASET DE TESTE"
        )
    )

    if set(colunas_treino) != set(colunas_teste):

        faltantes = [
            coluna
            for coluna in colunas_treino
            if coluna not in colunas_teste
        ]

        extras = [
            coluna
            for coluna in colunas_teste
            if coluna not in colunas_treino
        ]

        mensagem = (
            "Os preditores dos datasets de treino "
            "e teste precisam ser os mesmos."
        )

        if faltantes:

            mensagem += (
                "\n\nPresentes no treino e ausentes no teste:"
            )

            mensagem += "".join(
                f"\n- {coluna}"
                for coluna in faltantes
            )

        if extras:

            mensagem += (
                "\n\nPresentes no teste e ausentes no treino:"
            )

            mensagem += "".join(
                f"\n- {coluna}"
                for coluna in extras
            )

        raise ValueError(
            mensagem
        )

    y_treino, mapeamento = (
        preparar_target_treino(
            df_treino,
            target
        )
    )

    y_teste = preparar_target_teste(
        df_teste,
        target,
        mapeamento
    )

    X_treino = df_treino[
        colunas_treino
    ].copy()

    # Mantém exatamente a ordem das colunas do treino.
    X_teste = df_teste[
        colunas_treino
    ].copy()

    modelo = criar_modelo(
        parametros
    )

    modelo.fit(
        X_treino,
        y_treino
    )

    predicoes = modelo.predict(
        X_teste
    )

    metricas, matriz, relatorio = (
        calcular_metricas(
            y_teste,
            predicoes
        )
    )

    return {
        "metricas": metricas,
        "matriz": matriz,
        "relatorio": relatorio,
        "treino": len(X_treino),
        "teste": len(X_teste),
        "mapeamento": mapeamento
    }


# ============================================================
# EXECUÇÃO DA CROSS-VALIDATION
# ============================================================

def executar_cross_validation(
    df,
    target,
    parametros
):

    validar_valores_ausentes(
        df,
        "DATASET"
    )

    colunas_preditoras = (
        validar_preditores_numericos(
            df,
            target,
            "DATASET"
        )
    )

    y, mapeamento = preparar_target_treino(
        df,
        target
    )

    X = df[
        colunas_preditoras
    ].copy()

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)

    while True:

        try:

            folds = int(
                input(
                    "\nDigite o número de folds: "
                )
            )

            if folds >= 2:
                break

            print(
                "O número de folds precisa ser pelo menos 2."
            )

        except ValueError:

            print(
                "Digite um número inteiro."
            )

    menor_classe = y.value_counts().min()

    if folds > menor_classe:

        raise ValueError(
            f"O número de folds ({folds}) não pode ser "
            f"maior que a quantidade de amostras da "
            f"classe menos representada ({menor_classe})."
        )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados = []

    print("\n" + "=" * 70)
    print(
        f"RESULTADOS DOS {folds} FOLDS"
    )
    print("=" * 70)

    for numero_fold, (
        indices_treino,
        indices_teste
    ) in enumerate(
        cv.split(X, y),
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

        predicoes = modelo.predict(
            X_teste
        )

        metricas, matriz, relatorio = (
            calcular_metricas(
                y_teste,
                predicoes
            )
        )

        resultados.append(
            metricas
        )

        print(
            f"\n{'-' * 16} FOLD {numero_fold} "
            f"{'-' * 16}"
        )

        print(
            f"Acurácia:           "
            f"{metricas['accuracy']:.4f}"
        )

        print(
            f"Precisão ponderada:  "
            f"{metricas['precision']:.4f}"
        )

        print(
            f"Recall ponderado:    "
            f"{metricas['recall']:.4f}"
        )

        print(
            f"F1 ponderado:        "
            f"{metricas['f1']:.4f}"
        )

        print(
            "\nMatriz de confusão:"
        )

        print(matriz)

        print(
            "\nClassification report:"
        )

        print(relatorio)

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    df_resultados = pd.DataFrame(
        resultados
    )

    medias = df_resultados.mean()

    desvios = df_resultados.std(
        ddof=1
    )

    print("\n" + "=" * 70)
    print(
        "RESULTADO FINAL DA CROSS-VALIDATION"
    )
    print("=" * 70)

    print(
        f"Acurácia:           "
        f"{medias['accuracy']:.4f} "
        f"+/- {desvios['accuracy']:.4f}"
    )

    print(
        f"Precisão ponderada:  "
        f"{medias['precision']:.4f} "
        f"+/- {desvios['precision']:.4f}"
    )

    print(
        f"Recall ponderado:    "
        f"{medias['recall']:.4f} "
        f"+/- {desvios['recall']:.4f}"
    )

    print(
        f"F1 ponderado:        "
        f"{medias['f1']:.4f} "
        f"+/- {desvios['f1']:.4f}"
    )

    return {
        "folds": folds,
        "resultados": resultados,
        "medias": medias,
        "desvios": desvios,
        "mapeamento": mapeamento
    }


# ============================================================
# SALVAMENTO DO RELATÓRIO
# ============================================================

def salvar_relatorio(
    caminho_dataset,
    target,
    metodo,
    parametros,
    resultado
):
    """Salva o relatório no padrão dos demais modelos."""

    nome_base = Path(
        caminho_dataset
    ).stem

    caminho_saida = (
        PASTA_SAIDA
        / f"017_Decision_Tree_{nome_base}.txt"
    )

    linhas = []

    # ========================================================
    # CABEÇALHO
    # ========================================================

    linhas.append(
        "=" * 70
    )

    linhas.append(
        "ÁRVORE DE DECISÃO - CLASSIFICAÇÃO"
    )

    linhas.append(
        "=" * 70
    )

    linhas.append(
        f"Dataset: {caminho_dataset}"
    )

    linhas.append(
        f"Método: {metodo}"
    )

    linhas.append(
        f"Variável alvo: {target}"
    )

    linhas.append("")

    # ========================================================
    # CONFIGURAÇÃO
    # ========================================================

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
        "Modelo: DecisionTreeClassifier"
    )

    linhas.append(
        f"criterion           = "
        f"'{parametros['criterion']}'"
    )

    linhas.append(
        f"splitter            = "
        f"'{parametros['splitter']}'"
    )

    linhas.append(
        f"max_depth           = "
        f"{parametros['max_depth']}"
    )

    linhas.append(
        f"min_samples_split   = "
        f"{parametros['min_samples_split']}"
    )

    linhas.append(
        f"min_samples_leaf    = "
        f"{parametros['min_samples_leaf']}"
    )

    linhas.append(
        f"max_features        = "
        f"{parametros['max_features']}"
    )

    linhas.append(
        f"max_leaf_nodes      = "
        f"{parametros['max_leaf_nodes']}"
    )

    linhas.append(
        f"min_impurity_decrease = "
        f"{parametros['min_impurity_decrease']}"
    )

    linhas.append(
        f"class_weight        = "
        f"{parametros['class_weight']}"
    )

    linhas.append(
        f"ccp_alpha           = "
        f"{parametros['ccp_alpha']}"
    )

    linhas.append(
        f"random_state        = "
        f"{RANDOM_STATE}"
    )

    # ========================================================
    # OBSERVAÇÕES
    # ========================================================

    linhas.append("")

    linhas.append(
        "Observação:"
    )

    linhas.append(
        "A Árvore de Decisão realiza divisões recursivas "
        "dos dados para criar nós e folhas de decisão."
    )

    linhas.append(
        "max_depth controla a profundidade máxima da árvore."
    )

    linhas.append(
        "min_samples_split define o número mínimo de amostras "
        "necessário para dividir um nó."
    )

    linhas.append(
        "min_samples_leaf define o número mínimo de amostras "
        "permitido em uma folha."
    )

    linhas.append(
        "ccp_alpha controla a poda por complexidade de custo."
    )

    # ========================================================
    # CROSS-VALIDATION
    # ========================================================

    if metodo == "Cross-validation":

        folds = resultado["folds"]

        resultados = resultado["resultados"]

        medias = resultado["medias"]

        desvios = resultado["desvios"]

        linhas.append("")

        linhas.append(
            "=" * 70
        )

        linhas.append(
            f"RESULTADOS DOS {folds} FOLDS"
        )

        linhas.append(
            "=" * 70
        )

        for numero_fold, metricas in enumerate(
            resultados,
            start=1
        ):

            # ------------------------------------------------
            # Cabeçalho do Fold
            # ------------------------------------------------

            linhas.append("")

            linhas.append(
                f"{'-' * 16} FOLD {numero_fold} "
                f"{'-' * 16}"
            )

            linhas.append(
                f"Acurácia:           "
                f"{metricas['accuracy']:.4f}"
            )

            linhas.append(
                f"Precisão ponderada:  "
                f"{metricas['precision']:.4f}"
            )

            linhas.append(
                f"Recall ponderado:    "
                f"{metricas['recall']:.4f}"
            )

            linhas.append(
                f"F1 ponderado:        "
                f"{metricas['f1']:.4f}"
            )

            # ------------------------------------------------
            # IMPORTANTE:
            # A matriz e o classification report precisam ser
            # armazenados por fold.
            # ------------------------------------------------

        # Para manter matriz e classification report por fold,
        # eles são recalculados utilizando os mesmos folds.

        validar_valores_ausentes(
            pd.DataFrame(
                resultado.get(
                    "_dataset_original",
                    []
                )
            ),
            "DATASET"
        ) if False else None

        # As informações detalhadas dos folds são adicionadas
        # pelo resultado salvo em resultado["detalhes_folds"].

        linhas = adicionar_detalhes_cross_validation(
            linhas,
            resultado
        )

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

        linhas.append(
            f"Acurácia:           "
            f"{medias['accuracy']:.4f} "
            f"+/- {desvios['accuracy']:.4f}"
        )

        linhas.append(
            f"Precisão ponderada:  "
            f"{medias['precision']:.4f} "
            f"+/- {desvios['precision']:.4f}"
        )

        linhas.append(
            f"Recall ponderado:    "
            f"{medias['recall']:.4f} "
            f"+/- {desvios['recall']:.4f}"
        )

        linhas.append(
            f"F1 ponderado:        "
            f"{medias['f1']:.4f} "
            f"+/- {desvios['f1']:.4f}"
        )

    # ========================================================
    # AUTO TESTE / HOLD OUT
    # ========================================================

    else:

        metricas = resultado["metricas"]

        matriz = resultado["matriz"]

        relatorio = resultado["relatorio"]

        linhas.append("")

        linhas.append(
            "=" * 70
        )

        linhas.append(
            "RESULTADOS"
        )

        linhas.append(
            "=" * 70
        )

        linhas.append(
            f"Acurácia:           "
            f"{metricas['accuracy']:.4f}"
        )

        linhas.append(
            f"Precisão ponderada:  "
            f"{metricas['precision']:.4f}"
        )

        linhas.append(
            f"Recall ponderado:    "
            f"{metricas['recall']:.4f}"
        )

        linhas.append(
            f"F1 ponderado:        "
            f"{metricas['f1']:.4f}"
        )

        linhas.append("")

        linhas.append(
            "Matriz de confusão:"
        )

        linhas.append(
            str(matriz)
        )

        linhas.append("")

        linhas.append(
            "Classification report:"
        )

        linhas.append(
            relatorio
        )

        linhas.append("")

        linhas.append(
            f"Amostras de treino: "
            f"{resultado['treino']}"
        )

        linhas.append(
            f"Amostras de teste: "
            f"{resultado['teste']}"
        )

    # ========================================================
    # FINAL
    # ========================================================

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

    caminho_saida.write_text(
        "\n".join(linhas),
        encoding="utf-8"
    )

    print(
        f"\nRelatório salvo em:\n{caminho_saida}"
    )

    return caminho_saida


# ============================================================
# DETALHES DA CROSS-VALIDATION
# ============================================================

def adicionar_detalhes_cross_validation(
    linhas,
    resultado
):
    """
    Adiciona matriz de confusão e classification report
    de cada fold ao relatório.
    """

    detalhes = resultado.get(
        "detalhes_folds",
        []
    )

    # Remove os blocos anteriores de folds simples,
    # substituindo-os pelo formato completo.

    inicio = None

    for indice, linha in enumerate(linhas):

        if (
            linha.startswith(
                "---------------- FOLD"
            )
            or linha.startswith(
                "---------------- FOLD"
            )
        ):

            inicio = indice
            break

    if inicio is not None:

        linhas = linhas[:inicio]

    for detalhe in detalhes:

        numero_fold = detalhe[
            "fold"
        ]

        metricas = detalhe[
            "metricas"
        ]

        matriz = detalhe[
            "matriz"
        ]

        relatorio = detalhe[
            "relatorio"
        ]

        linhas.append("")

        linhas.append(
            f"{'-' * 16} FOLD {numero_fold} "
            f"{'-' * 16}"
        )

        linhas.append(
            f"Acurácia:           "
            f"{metricas['accuracy']:.4f}"
        )

        linhas.append(
            f"Precisão ponderada:  "
            f"{metricas['precision']:.4f}"
        )

        linhas.append(
            f"Recall ponderado:    "
            f"{metricas['recall']:.4f}"
        )

        linhas.append(
            f"F1 ponderado:        "
            f"{metricas['f1']:.4f}"
        )

        linhas.append("")

        linhas.append(
            "Matriz de confusão:"
        )

        linhas.append(
            str(matriz)
        )

        linhas.append("")

        linhas.append(
            "Classification report:"
        )

        linhas.append(
            relatorio
        )

    return linhas


# ============================================================
# CROSS-VALIDATION COM DETALHES
# ============================================================

def executar_cross_validation_com_detalhes(
    df,
    target,
    parametros
):

    validar_valores_ausentes(
        df,
        "DATASET"
    )

    colunas_preditoras = (
        validar_preditores_numericos(
            df,
            target,
            "DATASET"
        )
    )

    y, mapeamento = preparar_target_treino(
        df,
        target
    )

    X = df[
        colunas_preditoras
    ].copy()

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)

    while True:

        try:

            folds = int(
                input(
                    "\nDigite o número de folds: "
                )
            )

            if folds >= 2:
                break

            print(
                "O número de folds precisa ser pelo menos 2."
            )

        except ValueError:

            print(
                "Digite um número inteiro."
            )

    menor_classe = y.value_counts().min()

    if folds > menor_classe:

        raise ValueError(
            f"O número de folds ({folds}) não pode ser "
            f"maior que a quantidade de amostras da "
            f"classe menos representada ({menor_classe})."
        )

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultados = []

    detalhes_folds = []

    print("\n" + "=" * 70)
    print(
        f"RESULTADOS DOS {folds} FOLDS"
    )
    print("=" * 70)

    for numero_fold, (
        indices_treino,
        indices_teste
    ) in enumerate(
        cv.split(X, y),
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

        predicoes = modelo.predict(
            X_teste
        )

        metricas, matriz, relatorio = (
            calcular_metricas(
                y_teste,
                predicoes
            )
        )

        resultados.append(
            metricas
        )

        detalhes_folds.append(
            {
                "fold": numero_fold,
                "metricas": metricas,
                "matriz": matriz,
                "relatorio": relatorio
            }
        )

        print(
            f"\n{'-' * 16} FOLD {numero_fold} "
            f"{'-' * 16}"
        )

        print(
            f"Acurácia:           "
            f"{metricas['accuracy']:.4f}"
        )

        print(
            f"Precisão ponderada:  "
            f"{metricas['precision']:.4f}"
        )

        print(
            f"Recall ponderado:    "
            f"{metricas['recall']:.4f}"
        )

        print(
            f"F1 ponderado:        "
            f"{metricas['f1']:.4f}"
        )

        print(
            "\nMatriz de confusão:"
        )

        print(matriz)

        print(
            "\nClassification report:"
        )

        print(relatorio)

    df_resultados = pd.DataFrame(
        resultados
    )

    medias = df_resultados.mean()

    desvios = df_resultados.std(
        ddof=1
    )

    print("\n" + "=" * 70)
    print(
        "RESULTADO FINAL DA CROSS-VALIDATION"
    )
    print("=" * 70)

    print(
        f"Acurácia:           "
        f"{medias['accuracy']:.4f} "
        f"+/- {desvios['accuracy']:.4f}"
    )

    print(
        f"Precisão ponderada:  "
        f"{medias['precision']:.4f} "
        f"+/- {desvios['precision']:.4f}"
    )

    print(
        f"Recall ponderado:    "
        f"{medias['recall']:.4f} "
        f"+/- {desvios['recall']:.4f}"
    )

    print(
        f"F1 ponderado:        "
        f"{medias['f1']:.4f} "
        f"+/- {desvios['f1']:.4f}"
    )

    return {
        "folds": folds,
        "resultados": resultados,
        "medias": medias,
        "desvios": desvios,
        "detalhes_folds": detalhes_folds,
        "mapeamento": mapeamento
    }


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("ÁRVORE DE DECISÃO - CLASSIFICAÇÃO")
    print("=" * 70)

    # ========================================================
    # MÉTODO
    # ========================================================

    print("\nMétodo de avaliação:")

    print("1. Auto Teste")
    print("2. Hold Out")
    print("3. Cross-validation")

    while True:

        escolha = input(
            "\nEscolha: "
        ).strip()

        if escolha in ["1", "2", "3"]:
            break

        print("Opção inválida.")

    metodos = {
        "1": "Auto Teste",
        "2": "Hold Out",
        "3": "Cross-validation"
    }

    metodo = metodos[
        escolha
    ]

    # ========================================================
    # DATASET
    # ========================================================

    caminho = selecionar_arquivo(
        "SELECIONE O DATASET"
    )

    if caminho is None:
        return

    try:

        df = ler_arquivo(
            caminho
        )

    except Exception as erro:

        print(
            f"\nErro ao carregar o arquivo:\n{erro}"
        )

        return

    print(
        f"\nDimensões: "
        f"{df.shape[0]} linhas x "
        f"{df.shape[1]} colunas"
    )

    # ========================================================
    # TARGET
    # ========================================================

    target = selecionar_target(
        df
    )

    # ========================================================
    # CONFIGURAÇÃO
    # ========================================================

    parametros = configurar_modelo()

    # ========================================================
    # EXECUÇÃO
    # ========================================================

    try:

        if metodo == "Auto Teste":

            resultado = executar_auto_teste(
                df,
                target,
                parametros
            )

        elif metodo == "Hold Out":

            resultado = executar_hold_out(
                df,
                target,
                parametros
            )

        else:

            resultado = (
                executar_cross_validation_com_detalhes(
                    df,
                    target,
                    parametros
                )
            )

    except Exception as erro:

        print("\n" + "=" * 70)
        print("ERRO DURANTE A CLASSIFICAÇÃO")
        print("=" * 70)

        print(
            f"\n{erro}"
        )

        return

    # ========================================================
    # RELATÓRIO
    # ========================================================

    try:

        salvar_relatorio(
            caminho.name,
            target,
            metodo,
            parametros,
            resultado
        )

    except Exception as erro:

        print(
            "\nErro ao salvar o relatório:"
            f"\n{erro}"
        )

    print("\n" + "=" * 70)
    print("PROCESSO CONCLUÍDO")
    print("=" * 70)


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()