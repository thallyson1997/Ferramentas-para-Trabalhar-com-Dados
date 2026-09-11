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
    Verifica se uma coluna pode ser utilizada como
    variável categórica.

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
    "SCATTERPLOT"
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
# COLUNAS NUMÉRICAS
# ============================================================

numericas = colunas_numericas(
    df
)

if len(numericas) < 2:

    print(
        "\nÉ necessário possuir pelo menos "
        "duas colunas numéricas."
    )

    raise SystemExit


# ============================================================
# SELEÇÃO DE X
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "VARIÁVEL X"
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


coluna_x = selecionar_coluna(
    numericas,
    "\nEscolha a coluna para X: "
)


# ============================================================
# SELEÇÃO DE Y
# ============================================================

print(
    "\n" + "-" * 60
)

print(
    "VARIÁVEL Y"
)

print(
    "-" * 60
)


numericas_y = [
    coluna
    for coluna in numericas
    if coluna != coluna_x
]


print(
    "\nColunas numéricas disponíveis:"
)

for i, coluna in enumerate(
    numericas_y,
    start=1
):

    print(
        f"{i} - {coluna}"
    )


coluna_y = selecionar_coluna(
    numericas_y,
    "\nEscolha a coluna para Y: "
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
        if coluna not in {
            coluna_x,
            coluna_y
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


# ============================================================
# TAMANHO DOS PONTOS
# ============================================================

coluna_size = None

usar_size = resposta_sim_nao(
    "\nDeseja utilizar uma terceira variável numérica "
    "para controlar o tamanho dos pontos? [S/N]: "
)


if usar_size:

    sizes = [
        coluna
        for coluna in numericas
        if coluna not in {
            coluna_x,
            coluna_y
        }
    ]

    if not sizes:

        print(
            "\nNenhuma outra coluna numérica "
            "está disponível para size."
        )

    else:

        print(
            "\nColunas disponíveis para size:"
        )

        for i, coluna in enumerate(
            sizes,
            start=1
        ):

            print(
                f"{i} - {coluna}"
            )

        coluna_size = selecionar_coluna(
            sizes,
            "\nEscolha a coluna para size: "
        )


# ============================================================
# STYLE
# ============================================================

coluna_style = None

usar_style = resposta_sim_nao(
    "\nDeseja diferenciar grupos pelo formato "
    "dos pontos (style)? [S/N]: "
)


if usar_style:

    styles = colunas_categoricas(
        df
    )

    styles = [
        coluna
        for coluna in styles
        if coluna not in {
            coluna_x,
            coluna_y,
            coluna_hue
        }
    ]

    if not styles:

        print(
            "\nNenhuma coluna adequada "
            "para style foi encontrada."
        )

    else:

        print(
            "\nColunas disponíveis para style:"
        )

        for i, coluna in enumerate(
            styles,
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

        coluna_style = selecionar_coluna(
            styles,
            "\nEscolha a coluna para style: "
        )


# ============================================================
# INTERVALO DE X
# ============================================================

usar_intervalo_x = resposta_sim_nao(
    "\nDeseja trabalhar com um intervalo "
    "específico para X? [S/N]: "
)


limite_x_inferior = None
limite_x_superior = None


if usar_intervalo_x:

    (
        limite_x_inferior,
        limite_x_superior
    ) = selecionar_intervalo(
        df[coluna_x],
        coluna_x
    )


# ============================================================
# INTERVALO DE Y
# ============================================================

usar_intervalo_y = resposta_sim_nao(
    "\nDeseja trabalhar com um intervalo "
    "específico para Y? [S/N]: "
)


limite_y_inferior = None
limite_y_superior = None


if usar_intervalo_y:

    (
        limite_y_inferior,
        limite_y_superior
    ) = selecionar_intervalo(
        df[coluna_y],
        coluna_y
    )


# ============================================================
# CURVA DE REGRESSÃO
# ============================================================

usar_regressao = resposta_sim_nao(
    "\nDeseja adicionar uma linha de regressão? [S/N]: "
)


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

colunas_analise = [
    coluna_x,
    coluna_y
]


if coluna_hue:

    colunas_analise.append(
        coluna_hue
    )


if coluna_size:

    colunas_analise.append(
        coluna_size
    )


if coluna_style:

    colunas_analise.append(
        coluna_style
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


# ============================================================
# APLICAÇÃO DOS INTERVALOS
# ============================================================

linhas_antes_filtro = len(
    dados
)


if usar_intervalo_x:

    dados = dados[
        (dados[coluna_x] >= limite_x_inferior)
        &
        (dados[coluna_x] <= limite_x_superior)
    ]


if usar_intervalo_y:

    dados = dados[
        (dados[coluna_y] >= limite_y_inferior)
        &
        (dados[coluna_y] <= limite_y_superior)
    ]


if dados.empty:

    print(
        "\nNão existem dados dentro "
        "dos intervalos selecionados."
    )

    raise SystemExit


# ============================================================
# TAMANHO BASE DOS PONTOS
# ============================================================

while True:

    try:

        tamanho_pontos = float(
            input(
                "\nDigite o tamanho dos pontos "
                "(padrão: 60): "
            )
        )

        if tamanho_pontos > 0:
            break

        print(
            "O tamanho deve ser maior que zero."
        )

    except ValueError:

        print(
            "Digite um número válido."
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
        f"{coluna_y} por {coluna_x}"
    )


# ============================================================
# GERAÇÃO DO GRÁFICO
# ============================================================

sns.set_theme(
    style="whitegrid"
)

plt.figure(
    figsize=(10, 6)
)


# ------------------------------------------------------------
# CONFIGURAÇÃO DOS ARGUMENTOS
# ------------------------------------------------------------

argumentos = {
    "data": dados,
    "x": coluna_x,
    "y": coluna_y,
    "alpha": alpha,
    "s": tamanho_pontos
}


if coluna_hue:

    argumentos["hue"] = coluna_hue


if coluna_size:

    argumentos["size"] = coluna_size


if coluna_style:

    argumentos["style"] = coluna_style


# ------------------------------------------------------------
# SCATTERPLOT
# ------------------------------------------------------------

ax = sns.scatterplot(
    **argumentos
)


# ============================================================
# REGRESSÃO
# ============================================================

if usar_regressao:

    if coluna_hue or coluna_size or coluna_style:

        print(
            "\nAviso: a linha de regressão será "
            "calculada considerando apenas X e Y, "
            "independentemente dos grupos adicionais."
        )

    sns.regplot(
        data=dados,
        x=coluna_x,
        y=coluna_y,
        scatter=False,
        ax=ax
    )


# ============================================================
# LIMITES DOS EIXOS
# ============================================================

if usar_intervalo_x:

    ax.set_xlim(
        limite_x_inferior,
        limite_x_superior
    )


if usar_intervalo_y:

    ax.set_ylim(
        limite_y_inferior,
        limite_y_superior
    )


# ============================================================
# GRADE
# ============================================================

ax.grid(
    mostrar_grade
)


# ============================================================
# CONFIGURAÇÃO FINAL
# ============================================================

plt.title(
    titulo
)

plt.xlabel(
    coluna_x
)

plt.ylabel(
    coluna_y
)

plt.tight_layout()


# ============================================================
# NOME DO ARQUIVO
# ============================================================

nome_saida = (
    f"007_Pontos_"
    f"{nome_seguro(coluna_x)}"
    f"_por_"
    f"{nome_seguro(coluna_y)}"
)


if coluna_hue:

    nome_saida += (
        f"_por_"
        f"{nome_seguro(coluna_hue)}"
    )


if coluna_size:

    nome_saida += (
        f"_size_"
        f"{nome_seguro(coluna_size)}"
    )


if coluna_style:

    nome_saida += (
        f"_style_"
        f"{nome_seguro(coluna_style)}"
    )


if usar_intervalo_x:

    nome_saida += (
        f"_X_"
        f"{nome_seguro(str(limite_x_inferior))}"
        f"_"
        f"{nome_seguro(str(limite_x_superior))}"
    )


if usar_intervalo_y:

    nome_saida += (
        f"_Y_"
        f"{nome_seguro(str(limite_y_inferior))}"
        f"_"
        f"{nome_seguro(str(limite_y_superior))}"
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
    f"\nVariável X: {coluna_x}"
)

print(
    f"Variável Y: {coluna_y}"
)

if coluna_hue:

    print(
        f"Hue: {coluna_hue}"
    )

if coluna_size:

    print(
        f"Size: {coluna_size}"
    )

if coluna_style:

    print(
        f"Style: {coluna_style}"
    )

print(
    f"\nLinhas após remoção de valores ausentes: "
    f"{linhas_antes_filtro}"
)

print(
    f"Linhas utilizadas no gráfico: "
    f"{len(dados)}"
)

if usar_intervalo_x:

    print(
        f"\nIntervalo X: "
        f"{limite_x_inferior} até "
        f"{limite_x_superior}"
    )

if usar_intervalo_y:

    print(
        f"Intervalo Y: "
        f"{limite_y_inferior} até "
        f"{limite_y_superior}"
    )

print(
    "\nGráfico salvo em:"
)

print(
    caminho_saida
)