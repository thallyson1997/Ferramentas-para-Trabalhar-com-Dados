from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def colunas_numericas(df):
    """
    Retorna apenas colunas numéricas que não sejam booleanas.
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

    São aceitas:
    - colunas textuais;
    - colunas categóricas;
    - booleanas;
    - colunas numéricas com até 10 valores únicos.
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


def selecionar_intervalo(serie, nome_coluna):
    """
    Permite selecionar um intervalo numérico.

    Exemplos:
    2:100
    :100
    100:
    :
    """

    serie_valida = serie.dropna()

    menor_valor = serie_valida.min()
    maior_valor = serie_valida.max()

    print(
        f"\nMenor valor de {nome_coluna}: "
        f"{menor_valor}"
    )

    print(
        f"Maior valor de {nome_coluna}: "
        f"{maior_valor}"
    )

    print(
        "\nDigite o intervalo no formato menor:maior."
    )

    print(
        "Deixe um dos lados vazio para usar "
        "o limite original."
    )

    print(
        "Exemplos:"
    )

    print(
        "  2:100"
    )

    print(
        "  :100"
    )

    print(
        "  100:"
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
                "Formato inválido. "
                "Use menor:maior."
            )

            continue

        limite_inferior, limite_superior = partes

        try:

            if limite_inferior.strip() == "":
                limite_inferior = menor_valor
            else:
                limite_inferior = float(
                    limite_inferior
                )

            if limite_superior.strip() == "":
                limite_superior = maior_valor
            else:
                limite_superior = float(
                    limite_superior
                )

        except ValueError:

            print(
                "Digite apenas valores numéricos."
            )

            continue

        if limite_inferior > limite_superior:

            print(
                "O limite inferior não pode "
                "ser maior que o superior."
            )

            continue

        if limite_inferior < menor_valor:

            print(
                f"O limite inferior não pode "
                f"ser menor que {menor_valor}."
            )

            continue

        if limite_superior > maior_valor:

            print(
                f"O limite superior não pode "
                f"ser maior que {maior_valor}."
            )

            continue

        return (
            limite_inferior,
            limite_superior
        )


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
    "BOXPLOT"
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
# COLUNA NUMÉRICA
# ============================================================

numericas = colunas_numericas(
    df
)

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
    "\nEscolha a variável numérica: "
)


# ============================================================
# HUE
# ============================================================

coluna_hue = None

usar_hue = resposta_sim_nao(
    "\nDeseja diferenciar grupos por uma "
    "variável categórica (hue)? [S/N]: "
)


if usar_hue:

    hues = colunas_categoricas(
        df
    )

    hues = [
        coluna
        for coluna in hues
        if coluna != coluna_numerica
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


# ============================================================
# ORIENTAÇÃO
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "ORIENTAÇÃO"
)

print(
    "-" * 60
)

print(
    "\n1 - Vertical"
)

print(
    "2 - Horizontal"
)


while True:

    try:

        escolha_orientacao = int(
            input(
                "\nEscolha a orientação: "
            )
        )

        if escolha_orientacao == 1:

            orientacao = "v"
            break

        elif escolha_orientacao == 2:

            orientacao = "h"
            break

        print(
            "Escolha 1 ou 2."
        )

    except ValueError:

        print(
            "Digite apenas o número."
        )


# ============================================================
# INTERVALO
# ============================================================

usar_intervalo = resposta_sim_nao(
    "\nDeseja trabalhar com um intervalo "
    "específico para a variável numérica? [S/N]: "
)


limite_inferior = None
limite_superior = None


if usar_intervalo:

    (
        limite_inferior,
        limite_superior
    ) = selecionar_intervalo(
        df[coluna_numerica],
        coluna_numerica
    )


# ============================================================
# OUTLIERS
# ============================================================

mostrar_outliers = resposta_sim_nao(
    "\nDeseja mostrar os valores considerados "
    "outliers no boxplot? [S/N]: "
)


# ============================================================
# MÉDIA
# ============================================================

mostrar_media = resposta_sim_nao(
    "\nDeseja mostrar a média no boxplot? [S/N]: "
)


# ============================================================
# PONTOS INDIVIDUAIS
# ============================================================

mostrar_pontos = resposta_sim_nao(
    "\nDeseja mostrar os pontos individuais "
    "sobre o boxplot? [S/N]: "
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

    alpha = 0.7


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

    titulo = (
        f"Distribuição de {coluna_numerica}"
    )


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

colunas_analise = [
    coluna_numerica
]


if coluna_hue:

    colunas_analise.append(
        coluna_hue
    )


dados = df[
    colunas_analise
].dropna()


if dados.empty:

    print(
        "\nNão existem dados suficientes "
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
        (dados[coluna_numerica] >= limite_inferior)
        &
        (dados[coluna_numerica] <= limite_superior)
    ]


if dados.empty:

    print(
        "\nNão existem dados dentro "
        "do intervalo selecionado."
    )

    raise SystemExit


# ============================================================
# CONFIGURAÇÃO DO BOXPLOT
# ============================================================

sns.set_theme(
    style="whitegrid"
)

plt.figure(
    figsize=(10, 6)
)


# ============================================================
# BOXPLOT
# ============================================================

argumentos_boxplot = {
    "data": dados,
    "y": coluna_numerica
    if orientacao == "v"
    else None,
    "x": coluna_numerica
    if orientacao == "h"
    else None,
    "hue": coluna_hue,
    "showfliers": mostrar_outliers,
    "showmeans": mostrar_media,
    "meanline": False,
    "orient": orientacao,
    "linewidth": 1.5
}


sns.boxplot(
    **argumentos_boxplot
)


# ============================================================
# PONTOS INDIVIDUAIS
# ============================================================

if mostrar_pontos:

    if coluna_hue:

        sns.stripplot(
            data=dados,
            x=(
                coluna_numerica
                if orientacao == "h"
                else None
            ),
            y=(
                coluna_numerica
                if orientacao == "v"
                else None
            ),
            hue=coluna_hue,
            dodge=True,
            alpha=alpha,
            jitter=True,
            legend=False
        )

    else:

        sns.stripplot(
            data=dados,
            x=(
                coluna_numerica
                if orientacao == "h"
                else None
            ),
            y=(
                coluna_numerica
                if orientacao == "v"
                else None
            ),
            color=None,
            alpha=alpha,
            jitter=True
        )


# ============================================================
# LIMITES DOS EIXOS
# ============================================================

if usar_intervalo:

    if orientacao == "v":

        plt.ylim(
            limite_inferior,
            limite_superior
        )

    else:

        plt.xlim(
            limite_inferior,
            limite_superior
        )


# ============================================================
# GRADE
# ============================================================

ax = plt.gca()

ax.grid(
    mostrar_grade
)


# ============================================================
# CONFIGURAÇÃO FINAL
# ============================================================

plt.title(
    titulo
)


if orientacao == "v":

    plt.xlabel(
        coluna_hue
        if coluna_hue
        else ""
    )

    plt.ylabel(
        coluna_numerica
    )

else:

    plt.xlabel(
        coluna_numerica
    )

    plt.ylabel(
        coluna_hue
        if coluna_hue
        else ""
    )


plt.tight_layout()


# ============================================================
# NOME DO ARQUIVO
# ============================================================

nome_saida = (
    f"008_Caixa_"
    f"{nome_seguro(coluna_numerica)}"
)


if coluna_hue:

    nome_saida += (
        f"_por_"
        f"{nome_seguro(coluna_hue)}"
    )


if usar_intervalo:

    nome_saida += (
        f"_"
        f"{nome_seguro(str(limite_inferior))}"
        f"_"
        f"{nome_seguro(str(limite_superior))}"
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
    f"\nVariável numérica: "
    f"{coluna_numerica}"
)

if coluna_hue:

    print(
        f"Hue: {coluna_hue}"
    )

print(
    f"Orientação: "
    f"{'Vertical' if orientacao == 'v' else 'Horizontal'}"
)

print(
    f"Linhas após remoção de valores ausentes: "
    f"{linhas_antes_filtro}"
)

print(
    f"Linhas utilizadas no gráfico: "
    f"{len(dados)}"
)

print(
    f"Mostrar outliers: "
    f"{'Sim' if mostrar_outliers else 'Não'}"
)

print(
    f"Mostrar média: "
    f"{'Sim' if mostrar_media else 'Não'}"
)

print(
    f"Mostrar pontos individuais: "
    f"{'Sim' if mostrar_pontos else 'Não'}"
)

if usar_intervalo:

    print(
        f"\nIntervalo: "
        f"{limite_inferior} até "
        f"{limite_superior}"
    )

print(
    "\nGráfico salvo em:"
)

print(
    caminho_saida
)