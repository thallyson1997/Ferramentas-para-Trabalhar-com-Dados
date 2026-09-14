from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# CONFIGURAÇÃO DAS PASTAS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"

PASTA_SAIDA.mkdir(exist_ok=True)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def listar_arquivos():
    extensoes = {
        ".csv",
        ".xlsx",
        ".xls",
        ".json",
        ".parquet"
    }

    return sorted(
        [
            arquivo
            for arquivo in PASTA_ENTRADA.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in extensoes
        ]
    )


def ler_arquivo(caminho):
    extensao = caminho.suffix.lower()

    if extensao == ".csv":
        return pd.read_csv(caminho)

    elif extensao in {".xlsx", ".xls"}:
        return pd.read_excel(caminho)

    elif extensao == ".json":
        return pd.read_json(caminho)

    elif extensao == ".parquet":
        return pd.read_parquet(caminho)

    else:
        raise ValueError(
            "Formato de arquivo não suportado."
        )


def nome_seguro(texto):
    caracteres_invalidos = '<>:"/\\|?*'

    for caractere in caracteres_invalidos:
        texto = texto.replace(
            caractere,
            "_"
        )

    return texto


def resposta_sim_nao(pergunta):
    while True:

        resposta = input(
            pergunta
        ).strip().lower()

        if resposta in {"s", "sim"}:
            return True

        elif resposta in {
            "n",
            "nao",
            "não"
        }:
            return False

        print(
            "Digite S para sim ou N para não."
        )


def selecionar_coluna(colunas, mensagem):
    while True:

        try:

            escolha = int(
                input(mensagem)
            )

            if 1 <= escolha <= len(colunas):

                return colunas[
                    escolha - 1
                ]

            print(
                "Escolha uma opção válida."
            )

        except ValueError:

            print(
                "Digite apenas o número."
            )


def colunas_numericas(df):
    """
    Retorna somente colunas numéricas
    que não sejam booleanas.
    """

    return [
        coluna
        for coluna in df.columns
        if (
            pd.api.types.is_numeric_dtype(
                df[coluna]
            )
            and not pd.api.types.is_bool_dtype(
                df[coluna]
            )
        )
    ]


def coluna_pode_ser_categorica(serie):
    """
    Verifica se uma coluna pode ser utilizada
    como variável categórica.

    Aceita:
    - texto;
    - category;
    - boolean;
    - numérica com até 10 valores únicos.
    """

    quantidade_unicos = serie.nunique(
        dropna=True
    )

    eh_textual = (
        pd.api.types.is_object_dtype(serie)
        or pd.api.types.is_string_dtype(serie)
        or isinstance(
            serie.dtype,
            pd.CategoricalDtype
        )
    )

    eh_booleana = (
        pd.api.types.is_bool_dtype(serie)
    )

    eh_numerica_discreta = (
        pd.api.types.is_numeric_dtype(serie)
        and not pd.api.types.is_bool_dtype(serie)
        and quantidade_unicos <= 10
    )

    return (
        eh_textual
        or eh_booleana
        or eh_numerica_discreta
    )


def colunas_categoricas(df):
    return [
        coluna
        for coluna in df.columns
        if coluna_pode_ser_categorica(
            df[coluna]
        )
    ]


def identificar_colunas_datas(df):
    """
    Identifica colunas que já são datetime ou que
    aparentam conter datas armazenadas como texto.

    Colunas numéricas não são consideradas como datas
    automaticamente.
    """

    candidatas = []

    for coluna in df.columns:

        serie = df[coluna]

        # ----------------------------------------------------
        # Já é datetime
        # ----------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(
            serie
        ):

            candidatas.append(
                coluna
            )

            continue

        # ----------------------------------------------------
        # Não tenta interpretar números como datas
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            serie
        ):

            continue

        # ----------------------------------------------------
        # Tenta converter texto para datetime
        # ----------------------------------------------------

        serie_texto = serie.dropna()

        if serie_texto.empty:
            continue

        try:

            convertida = pd.to_datetime(
                serie_texto,
                errors="coerce",
                format="mixed"
            )

            proporcao_valida = (
                convertida.notna().mean()
            )

            # Considera candidata quando pelo menos
            # 80% dos valores conseguem ser interpretados
            # como datas.
            if proporcao_valida >= 0.80:

                candidatas.append(
                    coluna
                )

        except Exception:

            continue

    return candidatas


def converter_coluna_data(df, coluna):
    """
    Converte a coluna selecionada para datetime.
    """

    try:

        df[coluna] = pd.to_datetime(
            df[coluna],
            errors="coerce",
            format="mixed"
        )

    except Exception:

        df[coluna] = pd.to_datetime(
            df[coluna],
            errors="coerce"
        )

    return df


def selecionar_intervalo_datas(serie):
    """
    Permite selecionar um intervalo de datas.

    Exemplos:
    01/01/2025:31/12/2025
    :31/12/2025
    01/01/2025:
    :
    """

    serie_valida = serie.dropna()

    data_minima = serie_valida.min()
    data_maxima = serie_valida.max()

    print(
        f"\nData mínima: "
        f"{data_minima.strftime('%d/%m/%Y %H:%M:%S')}"
    )

    print(
        f"Data máxima: "
        f"{data_maxima.strftime('%d/%m/%Y %H:%M:%S')}"
    )

    print(
        "\nDigite o intervalo no formato:"
    )

    print(
        "menor_data:maior_data"
    )

    print(
        "\nExemplos:"
    )

    print(
        "  01/01/2025:31/12/2025"
    )

    print(
        "  :31/12/2025"
    )

    print(
        "  01/01/2025:"
    )

    print(
        "  :"
    )

    while True:

        intervalo = input(
            "\nIntervalo: "
        ).strip()

        partes = intervalo.split(":")

        if len(partes) != 2:

            print(
                "Formato inválido."
            )

            continue

        limite_inferior, limite_superior = partes

        try:

            if limite_inferior.strip() == "":
                limite_inferior = data_minima
            else:
                limite_inferior = pd.to_datetime(
                    limite_inferior.strip(),
                    dayfirst=True
                )

            if limite_superior.strip() == "":
                limite_superior = data_maxima
            else:
                limite_superior = pd.to_datetime(
                    limite_superior.strip(),
                    dayfirst=True
                )

        except Exception:

            print(
                "Não foi possível interpretar "
                "uma das datas."
            )

            print(
                "Utilize o formato DD/MM/AAAA."
            )

            continue

        # ----------------------------------------------------
        # Se o usuário informou somente uma data, o limite
        # superior/inferior pode receber 00:00:00.
        # Ajuste para abranger o dia inteiro quando for
        # explicitamente uma data sem horário.
        # ----------------------------------------------------

        if (
            limite_superior.hour == 0
            and limite_superior.minute == 0
            and limite_superior.second == 0
        ):

            texto_original = (
                partes[1].strip()
            )

            if texto_original:

                limite_superior = (
                    limite_superior
                    + pd.Timedelta(days=1)
                    - pd.Timedelta(seconds=1)
                )

        if limite_inferior > limite_superior:

            print(
                "A data inicial não pode ser "
                "posterior à data final."
            )

            continue

        if limite_inferior < data_minima:

            print(
                "A data inicial não pode ser "
                "anterior à data mínima."
            )

            continue

        if limite_superior > data_maxima:

            print(
                "A data final não pode ser "
                "posterior à data máxima."
            )

            continue

        return (
            limite_inferior,
            limite_superior
        )


def aplicar_granularidade(serie, granularidade):
    """
    Converte as datas para a granularidade escolhida
    antes da agregação.

    Isso permite agrupar, por exemplo:
    - vários horários em um mesmo dia;
    - vários dias em um mesmo mês;
    - vários dias em uma mesma semana.
    """

    if granularidade == "Dia":

        return serie.dt.floor(
            "D"
        )

    elif granularidade == "Semana":

        return (
            serie
            .dt.to_period("W")
            .dt.start_time
        )

    elif granularidade == "Mês":

        return (
            serie
            .dt.to_period("M")
            .dt.start_time
        )

    elif granularidade == "Trimestre":

        return (
            serie
            .dt.to_period("Q")
            .dt.start_time
        )

    elif granularidade == "Ano":

        return (
            serie
            .dt.to_period("Y")
            .dt.start_time
        )

    else:

        return serie


# ============================================================
# SELEÇÃO DO ARQUIVO
# ============================================================

arquivos = listar_arquivos()

if not arquivos:

    print(
        "Nenhum arquivo compatível foi encontrado "
        "na pasta ENTRADA."
    )

    raise SystemExit


print(
    "\n" + "=" * 60
)

print(
    "GRÁFICO TEMPORAL"
)

print(
    "=" * 60
)

print(
    "\nArquivos disponíveis:"
)

for i, arquivo in enumerate(
    arquivos,
    start=1
):

    print(
        f"{i} - {arquivo.name}"
    )


while True:

    try:

        escolha = int(
            input(
                "\nEscolha o arquivo: "
            )
        )

        if 1 <= escolha <= len(arquivos):

            caminho_arquivo = arquivos[
                escolha - 1
            ]

            break

        print(
            "Escolha uma opção válida."
        )

    except ValueError:

        print(
            "Digite apenas o número."
        )


# ============================================================
# LEITURA DO DATASET
# ============================================================

try:

    df = ler_arquivo(
        caminho_arquivo
    )

except Exception as erro:

    print(
        f"\nErro ao ler o arquivo: {erro}"
    )

    raise SystemExit


print(
    f"\nArquivo selecionado: "
    f"{caminho_arquivo.name}"
)

print(
    f"Linhas: {len(df)}"
)

print(
    f"Colunas: {len(df.columns)}"
)


# ============================================================
# IDENTIFICAÇÃO DAS DATAS
# ============================================================

colunas_datas = identificar_colunas_datas(
    df
)


if not colunas_datas:

    print(
        "\n" + "=" * 60
    )

    print(
        "NENHUMA COLUNA DE DATA ENCONTRADA"
    )

    print(
        "=" * 60
    )

    print(
        "\nO dataset não possui uma coluna "
        "identificada como data."
    )

    print(
        "O programa será encerrado."
    )

    raise SystemExit


# ============================================================
# SELEÇÃO DA DATA
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "VARIÁVEL TEMPORAL"
)

print(
    "-" * 60
)

print(
    "\nColunas identificadas como datas:"
)

for i, coluna in enumerate(
    colunas_datas,
    start=1
):

    print(
        f"{i} - {coluna}"
    )


coluna_data = selecionar_coluna(
    colunas_datas,
    "\nEscolha a coluna de datas: "
)


# ============================================================
# CONVERSÃO DA DATA
# ============================================================

df = converter_coluna_data(
    df,
    coluna_data
)


quantidade_datas_invalidas = (
    df[coluna_data].isna().sum()
)


if quantidade_datas_invalidas > 0:

    print(
        f"\nAviso: "
        f"{quantidade_datas_invalidas} "
        f"valor(es) não puderam ser interpretados "
        f"como data e serão ignorados."
    )


if df[coluna_data].notna().sum() == 0:

    print(
        "\nNenhuma data válida foi encontrada."
    )

    raise SystemExit


# ============================================================
# TIPO DA SEGUNDA VARIÁVEL
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "TIPO DA SEGUNDA VARIÁVEL"
)

print(
    "-" * 60
)

print(
    "\n1 - Numérica"
)

print(
    "2 - Categórica"
)


while True:

    try:

        escolha_tipo = int(
            input(
                "\nEscolha o tipo da variável: "
            )
        )

        if escolha_tipo in {1, 2}:
            break

        print(
            "Escolha 1 ou 2."
        )

    except ValueError:

        print(
            "Digite apenas o número."
        )


# ============================================================
# VARIÁVEL NUMÉRICA
# ============================================================

coluna_numerica = None
metodo_agregacao = None


if escolha_tipo == 1:

    numericas = colunas_numericas(
        df
    )

    numericas = [
        coluna
        for coluna in numericas
        if coluna != coluna_data
    ]

    if not numericas:

        print(
            "\nNenhuma coluna numérica foi encontrada."
        )

        raise SystemExit


    print(
        "\n" + "-" * 60
    )

    print(
        "VARIÁVEL NUMÉRICA"
    )

    print(
        "-" * 60
    )

    print(
        "\nColunas numéricas disponíveis:"
    )

    for i, coluna in enumerate(
        numericas,
        start=1
    ):

        print(
            f"{i} - {coluna}"
        )


    coluna_numerica = selecionar_coluna(
        numericas,
        "\nEscolha a coluna numérica: "
    )


    # ========================================================
    # MÉTODO DE AGREGAÇÃO
    # ========================================================

    print(
        "\n" + "-" * 60
    )

    print(
        "AGREGAÇÃO DOS VALORES"
    )

    print(
        "-" * 60
    )

    print(
        "\nQuando houver mais de uma observação "
        "na mesma data:"
    )

    print(
        "\n1 - Soma"
    )

    print(
        "2 - Média"
    )

    print(
        "3 - Mediana"
    )


    while True:

        try:

            escolha_agregacao = int(
                input(
                    "\nEscolha a forma de agregação: "
                )
            )

            if escolha_agregacao == 1:

                metodo_agregacao = "sum"
                nome_agregacao = "Soma"
                break

            elif escolha_agregacao == 2:

                metodo_agregacao = "mean"
                nome_agregacao = "Média"
                break

            elif escolha_agregacao == 3:

                metodo_agregacao = "median"
                nome_agregacao = "Mediana"
                break

            print(
                "Escolha 1, 2 ou 3."
            )

        except ValueError:

            print(
                "Digite apenas o número."
            )


# ============================================================
# VARIÁVEL CATEGÓRICA
# ============================================================

coluna_categorica = None


if escolha_tipo == 2:

    categoricas = colunas_categoricas(
        df
    )

    categoricas = [
        coluna
        for coluna in categoricas
        if coluna != coluna_data
    ]

    if not categoricas:

        print(
            "\nNenhuma coluna adequada "
            "para análise categórica foi encontrada."
        )

        raise SystemExit


    print(
        "\n" + "-" * 60
    )

    print(
        "VARIÁVEL CATEGÓRICA"
    )

    print(
        "-" * 60
    )

    print(
        "\nColunas disponíveis:"
    )

    for i, coluna in enumerate(
        categoricas,
        start=1
    ):

        quantidade = df[
            coluna
        ].nunique(
            dropna=True
        )

        print(
            f"{i} - {coluna} "
            f"({quantidade} valores únicos)"
        )


    coluna_categorica = selecionar_coluna(
        categoricas,
        "\nEscolha a coluna categórica: "
    )


# ============================================================
# HUE
# ============================================================

coluna_hue = None


# Hue só existe no modo numérico.
if escolha_tipo == 1:

    usar_hue = resposta_sim_nao(
        "\nDeseja separar os dados por uma "
        "variável categórica (hue)? [S/N]: "
    )


    if usar_hue:

        hues = colunas_categoricas(
            df
        )

        hues = [
            coluna
            for coluna in hues
            if coluna not in {
                coluna_data,
                coluna_numerica
            }
        ]

        if not hues:

            print(
                "\nNenhuma coluna adequada "
                "para hue foi encontrada."
            )

        else:

            print(
                "\nColunas disponíveis para hue:"
            )

            for i, coluna in enumerate(
                hues,
                start=1
            ):

                quantidade = df[
                    coluna
                ].nunique(
                    dropna=True
                )

                print(
                    f"{i} - {coluna} "
                    f"({quantidade} valores únicos)"
                )


            coluna_hue = selecionar_coluna(
                hues,
                "\nEscolha a coluna para hue: "
            )


else:

    print(
        "\nNo modo categórico, não é utilizado hue."
    )

    print(
        "Cada categoria será representada "
        "por uma linha temporal."
    )


# ============================================================
# GRANULARIDADE TEMPORAL
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "GRANULARIDADE TEMPORAL"
)

print(
    "-" * 60
)

print(
    "\nComo deseja distribuir os dados no tempo?"
)

print(
    "\n1 - Data/Hora original"
)

print(
    "2 - Dia"
)

print(
    "3 - Semana"
)

print(
    "4 - Mês"
)

print(
    "5 - Trimestre"
)

print(
    "6 - Ano"
)


while True:

    try:

        escolha_granularidade = int(
            input(
                "\nEscolha a granularidade: "
            )
        )

        granularidades = {
            1: "Original",
            2: "Dia",
            3: "Semana",
            4: "Mês",
            5: "Trimestre",
            6: "Ano"
        }

        if escolha_granularidade in granularidades:

            granularidade = granularidades[
                escolha_granularidade
            ]

            break

        print(
            "Escolha uma opção entre 1 e 6."
        )

    except ValueError:

        print(
            "Digite apenas o número."
        )


# ============================================================
# INTERVALO DE DATAS
# ============================================================

usar_intervalo = resposta_sim_nao(
    "\nDeseja trabalhar com um intervalo "
    "específico de datas? [S/N]: "
)


limite_data_inferior = None
limite_data_superior = None


if usar_intervalo:

    (
        limite_data_inferior,
        limite_data_superior
    ) = selecionar_intervalo_datas(
        df[coluna_data]
    )


# ============================================================
# MARCADORES
# ============================================================

mostrar_marcadores = resposta_sim_nao(
    "\nDeseja mostrar marcadores nos pontos "
    "da linha temporal? [S/N]: "
)


# ============================================================
# TRANSPARÊNCIA
# ============================================================

alterar_transparencia = resposta_sim_nao(
    "\nDeseja alterar a transparência? [S/N]: "
)


if alterar_transparencia:

    while True:

        try:

            alpha = float(
                input(
                    "Digite a transparência (0 a 1): "
                )
            )

            if 0 < alpha <= 1:
                break

            print(
                "Digite um valor maior que 0 "
                "e menor ou igual a 1."
            )

        except ValueError:

            print(
                "Digite um número válido."
            )

else:

    alpha = 0.8


# ============================================================
# ESPESSURA DA LINHA
# ============================================================

while True:

    try:

        espessura_linha = float(
            input(
                "\nDigite a espessura da linha "
                "(padrão: 2): "
            )
        )

        if espessura_linha > 0:
            break

        print(
            "A espessura deve ser maior que zero."
        )

    except ValueError:

        print(
            "Digite um número válido."
        )


# ============================================================
# GRADE
# ============================================================

mostrar_grade = resposta_sim_nao(
    "\nDeseja mostrar a grade do gráfico? [S/N]: "
)


# ============================================================
# TÍTULO
# ============================================================

personalizar_titulo = resposta_sim_nao(
    "\nDeseja personalizar o título? [S/N]: "
)


if personalizar_titulo:

    while True:

        titulo = input(
            "Digite o título do gráfico: "
        ).strip()

        if titulo:
            break

        print(
            "O título não pode ficar vazio."
        )

else:

    if escolha_tipo == 1:

        titulo = (
            f"{coluna_numerica} ao longo do tempo"
        )

    else:

        titulo = (
            f"Distribuição temporal de "
            f"{coluna_categorica}"
        )


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

if escolha_tipo == 1:

    colunas_analise = [
        coluna_data,
        coluna_numerica
    ]

    if coluna_hue:

        colunas_analise.append(
            coluna_hue
        )

else:

    colunas_analise = [
        coluna_data,
        coluna_categorica
    ]


dados = df[
    colunas_analise
].dropna(
    subset=[
        coluna_data
    ]
)


if dados.empty:

    print(
        "\nNão existem datas válidas "
        "para gerar o gráfico."
    )

    raise SystemExit


linhas_antes_filtro = len(
    dados
)


# ============================================================
# APLICAÇÃO DO INTERVALO
# ============================================================

if usar_intervalo:

    dados = dados[
        (dados[coluna_data] >= limite_data_inferior)
        &
        (dados[coluna_data] <= limite_data_superior)
    ]


if dados.empty:

    print(
        "\nNão existem dados dentro "
        "do intervalo selecionado."
    )

    raise SystemExit


# ============================================================
# APLICAÇÃO DA GRANULARIDADE
# ============================================================

dados = dados.copy()


dados["_DATA_ANALISE_"] = (
    aplicar_granularidade(
        dados[coluna_data],
        granularidade
    )
)


# ============================================================
# AGREGAÇÃO — MODO NUMÉRICO
# ============================================================

if escolha_tipo == 1:

    colunas_groupby = [
        "_DATA_ANALISE_"
    ]

    if coluna_hue:

        colunas_groupby.append(
            coluna_hue
        )


    dados_agregados = (
        dados
        .groupby(
            colunas_groupby,
            dropna=False
        )[coluna_numerica]
        .agg(
            metodo_agregacao
        )
        .reset_index()
    )


# ============================================================
# AGREGAÇÃO — MODO CATEGÓRICO
# ============================================================

else:

    dados_agregados = (
        dados
        .groupby(
            [
                "_DATA_ANALISE_",
                coluna_categorica
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="Quantidade"
        )
    )


# ============================================================
# VERIFICAÇÃO DOS DADOS AGREGADOS
# ============================================================

if dados_agregados.empty:

    print(
        "\nNão existem dados suficientes "
        "após a agregação."
    )

    raise SystemExit


# ============================================================
# ORDENAÇÃO
# ============================================================

dados_agregados = dados_agregados.sort_values(
    "_DATA_ANALISE_"
)


# ============================================================
# GERAÇÃO DO GRÁFICO
# ============================================================

sns.set_theme(
    style="whitegrid"
)

plt.figure(
    figsize=(12, 6)
)


# ============================================================
# MODO NUMÉRICO
# ============================================================

if escolha_tipo == 1:

    coluna_y = coluna_numerica


    argumentos = {
        "data": dados_agregados,
        "x": "_DATA_ANALISE_",
        "y": coluna_y,
        "linewidth": espessura_linha,
        "alpha": alpha,
        "marker": "o" if mostrar_marcadores else None,
        "errorbar": None
    }


    if coluna_hue:

        argumentos["hue"] = coluna_hue


    sns.lineplot(
        **argumentos
    )


# ============================================================
# MODO CATEGÓRICO
# ============================================================

else:

    coluna_y = "Quantidade"


    sns.lineplot(
        data=dados_agregados,
        x="_DATA_ANALISE_",
        y=coluna_y,
        hue=coluna_categorica,
        linewidth=espessura_linha,
        alpha=alpha,
        marker="o" if mostrar_marcadores else None,
        errorbar=None
    )


# ============================================================
# CONFIGURAÇÃO DO EIXO TEMPORAL
# ============================================================

ax = plt.gca()


if granularidade == "Original":

    # Mostra data e horário quando disponíveis.
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d/%m/%Y"
        )
    )

elif granularidade == "Dia":

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d/%m/%Y"
        )
    )

elif granularidade == "Semana":

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d/%m/%Y"
        )
    )

elif granularidade == "Mês":

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%m/%Y"
        )
    )

elif granularidade == "Trimestre":

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%m/%Y"
        )
    )

elif granularidade == "Ano":

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y"
        )
    )


plt.xticks(
    rotation=45
)


# ============================================================
# GRADE
# ============================================================

ax.grid(
    mostrar_grade
)


# ============================================================
# TÍTULOS DOS EIXOS
# ============================================================

plt.title(
    titulo
)

plt.xlabel(
    coluna_data
)

if escolha_tipo == 1:

    plt.ylabel(
        f"{coluna_numerica} "
        f"({nome_agregacao})"
    )

else:

    plt.ylabel(
        "Quantidade de ocorrências"
    )


# ============================================================
# AJUSTES FINAIS
# ============================================================

plt.tight_layout()


# ============================================================
# NOME DO ARQUIVO
# ============================================================

nome_saida = (
    f"009_Temporal_"
    f"{nome_seguro(coluna_data)}"
)


if escolha_tipo == 1:

    nome_saida += (
        f"_"
        f"{nome_seguro(coluna_numerica)}"
        f"_"
        f"{nome_seguro(nome_agregacao)}"
    )

    if coluna_hue:

        nome_saida += (
            f"_por_"
            f"{nome_seguro(coluna_hue)}"
        )

else:

    nome_saida += (
        f"_"
        f"{nome_seguro(coluna_categorica)}"
        f"_Contagem"
    )


nome_saida += (
    f"_{nome_seguro(granularidade)}"
)


if usar_intervalo:

    nome_saida += (
        f"_"
        f"{nome_seguro(limite_data_inferior.strftime('%Y-%m-%d'))}"
        f"_"
        f"{nome_seguro(limite_data_superior.strftime('%Y-%m-%d'))}"
    )


nome_saida += ".png"


# ============================================================
# SALVAMENTO
# ============================================================

caminho_saida = (
    PASTA_SAIDA / nome_saida
)


plt.savefig(
    caminho_saida,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# FINALIZAÇÃO
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "ANÁLISE CONCLUÍDA"
)

print(
    "=" * 60
)

print(
    f"\nVariável temporal: "
    f"{coluna_data}"
)

if escolha_tipo == 1:

    print(
        f"Tipo da segunda variável: Numérica"
    )

    print(
        f"Variável numérica: "
        f"{coluna_numerica}"
    )

    print(
        f"Agregação: "
        f"{nome_agregacao}"
    )

    if coluna_hue:

        print(
            f"Hue: "
            f"{coluna_hue}"
        )

else:

    print(
        f"Tipo da segunda variável: Categórica"
    )

    print(
        f"Variável categórica: "
        f"{coluna_categorica}"
    )

    print(
        "Método: Contagem de ocorrências"
    )

    print(
        "Hue: Não utilizado"
    )


print(
    f"Granularidade temporal: "
    f"{granularidade}"
)

print(
    f"\nLinhas antes do filtro temporal: "
    f"{linhas_antes_filtro}"
)

print(
    f"Linhas utilizadas: "
    f"{len(dados)}"
)

print(
    f"Linhas após agregação: "
    f"{len(dados_agregados)}"
)


if usar_intervalo:

    print(
        f"\nIntervalo temporal: "
        f"{limite_data_inferior.strftime('%d/%m/%Y %H:%M:%S')}"
        f" até "
        f"{limite_data_superior.strftime('%d/%m/%Y %H:%M:%S')}"
    )


print(
    "\nGráfico salvo em:"
)

print(
    caminho_saida
)