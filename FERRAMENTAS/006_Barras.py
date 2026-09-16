from utilidades import ler_arquivo
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
    extensoes = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

    return sorted(
        [
            arquivo
            for arquivo in PASTA_ENTRADA.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in extensoes
        ]
    )

def nome_seguro(texto):
    caracteres_invalidos = '<>:"/\\|?*'

    for caractere in caracteres_invalidos:
        texto = texto.replace(caractere, "_")

    return texto


def resposta_sim_nao(pergunta):
    while True:

        resposta = input(pergunta).strip().lower()

        if resposta in {"s", "sim"}:
            return True

        elif resposta in {"n", "nao", "não"}:
            return False

        print("Digite S para sim ou N para não.")


def coluna_pode_ser_categorica(serie):
    """
    Verifica se uma coluna pode ser utilizada como variável
    categórica.

    São aceitas:
    - colunas textuais;
    - colunas categóricas;
    - colunas booleanas;
    - colunas numéricas com até 10 valores únicos.
    """

    quantidade_unicos = serie.nunique(dropna=True)

    eh_textual = (
        pd.api.types.is_object_dtype(serie)
        or pd.api.types.is_string_dtype(serie)
        or isinstance(serie.dtype, pd.CategoricalDtype)
    )

    eh_booleana = pd.api.types.is_bool_dtype(serie)

    eh_numerica_com_poucos_valores = (
        pd.api.types.is_numeric_dtype(serie)
        and not pd.api.types.is_bool_dtype(serie)
        and quantidade_unicos <= 10
    )

    return (
        eh_textual
        or eh_booleana
        or eh_numerica_com_poucos_valores
    )


def colunas_categoricas(df):
    """
    Retorna as colunas que podem ser utilizadas como
    variável categórica.
    """

    return [
        coluna
        for coluna in df.columns
        if coluna_pode_ser_categorica(df[coluna])
    ]


def colunas_hue(df, coluna_principal):
    """
    Retorna colunas que podem ser utilizadas como hue.

    A coluna principal é excluída.
    """

    candidatas = []

    for coluna in df.columns:

        if coluna == coluna_principal:
            continue

        if coluna_pode_ser_categorica(df[coluna]):
            candidatas.append(coluna)

    return candidatas


def selecionar_coluna(colunas, mensagem):
    """
    Exibe uma lista numerada de colunas e permite que o
    usuário selecione uma delas.
    """

    while True:

        try:

            escolha = int(
                input(mensagem)
            )

            if 1 <= escolha <= len(colunas):
                return colunas[escolha - 1]

            print("Escolha uma opção válida.")

        except ValueError:

            print("Digite apenas o número.")


def selecionar_intervalo_categorias():
    """
    Permite definir uma quantidade máxima de categorias
    para exibir no gráfico.
    """

    while True:

        try:

            quantidade = int(
                input(
                    "\nDigite a quantidade máxima de categorias "
                    "que deseja exibir: "
                )
            )

            if quantidade > 0:
                return quantidade

            print(
                "A quantidade deve ser maior que zero."
            )

        except ValueError:

            print(
                "Digite um número inteiro válido."
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


print("\n" + "=" * 60)
print("GRÁFICO DE BARRAS")
print("=" * 60)

print("\nArquivos disponíveis:")

for i, arquivo in enumerate(arquivos, start=1):

    print(
        f"{i} - {arquivo.name}"
    )


caminho_arquivo = None

while True:

    try:

        escolha = int(
            input("\nEscolha o arquivo: ")
        )

        if 1 <= escolha <= len(arquivos):

            caminho_arquivo = arquivos[escolha - 1]

            break

        print("Escolha uma opção válida.")

    except ValueError:

        print("Digite apenas o número.")


# ============================================================
# LEITURA DO DATASET
# ============================================================

try:

    df = ler_arquivo(caminho_arquivo)

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
# SELEÇÃO DA VARIÁVEL CATEGÓRICA
# ============================================================

categoricas = colunas_categoricas(df)

if not categoricas:

    print(
        "\nNenhuma coluna categórica adequada foi encontrada."
    )

    print(
        "Para colunas numéricas, são aceitas apenas "
        "colunas com até 10 valores únicos."
    )

    raise SystemExit


print("\n" + "-" * 60)
print("COLUNA CATEGÓRICA")
print("-" * 60)

print(
    "\nColunas disponíveis:"
)

for i, coluna in enumerate(categoricas, start=1):

    quantidade = df[coluna].nunique(
        dropna=True
    )

    tipo = df[coluna].dtype

    print(
        f"{i} - {coluna} "
        f"({tipo}, {quantidade} valores únicos)"
    )


coluna_principal = selecionar_coluna(
    categoricas,
    "\nEscolha a coluna categórica: "
)


# ============================================================
# HUE
# ============================================================

coluna_hue = None

usar_hue = resposta_sim_nao(
    "\nDeseja adicionar uma segunda variável para "
    "diferenciar os grupos (hue)? [S/N]: "
)


if usar_hue:

    hues = colunas_hue(
        df,
        coluna_principal
    )

    if not hues:

        print(
            "\nNenhuma coluna adequada para hue "
            "foi encontrada."
        )

    else:

        print(
            "\nColunas disponíveis para hue:"
        )

        for i, coluna in enumerate(hues, start=1):

            quantidade = df[coluna].nunique(
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
# PREPARAÇÃO DOS DADOS
# ============================================================

colunas_analise = [
    coluna_principal
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


# ============================================================
# QUANTIDADE DE CATEGORIAS
# ============================================================

quantidade_categorias = dados[
    coluna_principal
].nunique()


print("\n" + "-" * 60)
print("CATEGORIAS")
print("-" * 60)

print(
    f"\nQuantidade de categorias encontradas: "
    f"{quantidade_categorias}"
)


# ============================================================
# ORDENAÇÃO
# ============================================================

print("\nComo deseja ordenar as categorias?")
print("1 - Ordem original")
print("2 - Menor para maior quantidade")
print("3 - Maior para menor quantidade")


while True:

    try:

        escolha_ordenacao = int(
            input("\nEscolha uma opção: ")
        )

        if escolha_ordenacao in {1, 2, 3}:
            break

        print(
            "Escolha uma opção entre 1 e 3."
        )

    except ValueError:

        print(
            "Digite apenas o número."
        )


# ============================================================
# LIMITE DE CATEGORIAS
# ============================================================

limitar_categorias = resposta_sim_nao(
    "\nDeseja limitar a quantidade de categorias "
    "exibidas? [S/N]: "
)


limite_categorias = None

if limitar_categorias:

    limite_categorias = selecionar_intervalo_categorias()


# ============================================================
# PREPARAÇÃO DAS CATEGORIAS
# ============================================================

contagem_principal = dados[
    coluna_principal
].value_counts()


if escolha_ordenacao == 1:

    categorias_ordenadas = list(
        dados[coluna_principal].drop_duplicates()
    )

elif escolha_ordenacao == 2:

    categorias_ordenadas = list(
        contagem_principal.sort_values().index
    )

else:

    categorias_ordenadas = list(
        contagem_principal.sort_values(
            ascending=False
        ).index
    )


# ============================================================
# APLICAÇÃO DO LIMITE DE CATEGORIAS
# ============================================================

if limite_categorias is not None:

    categorias_ordenadas = (
        categorias_ordenadas[
            :limite_categorias
        ]
    )

    dados = dados[
        dados[coluna_principal].isin(
            categorias_ordenadas
        )
    ]


if dados.empty:

    print(
        "\nNão existem dados suficientes "
        "após a aplicação do limite."
    )

    raise SystemExit


# ============================================================
# FORMA DE APRESENTAÇÃO DO HUE
# ============================================================

multiple = "layer"


if coluna_hue:

    print(
        "\nComo deseja apresentar os grupos?"
    )

    print(
        "1 - Sobrepostos"
    )

    print(
        "2 - Lado a lado"
    )

    print(
        "3 - Empilhados"
    )

    while True:

        try:

            escolha_multiple = int(
                input("\nEscolha uma opção: ")
            )

            if escolha_multiple == 1:

                multiple = "layer"

                break

            elif escolha_multiple == 2:

                multiple = "dodge"

                break

            elif escolha_multiple == 3:

                multiple = "stack"

                break

            print(
                "Escolha uma opção entre 1 e 3."
            )

        except ValueError:

            print(
                "Digite apenas o número."
            )


# ============================================================
# ORIENTAÇÃO
# ============================================================

print("\nComo deseja apresentar as barras?")
print("1 - Vertical")
print("2 - Horizontal")


while True:

    try:

        escolha_orientacao = int(
            input("\nEscolha uma opção: ")
        )

        if escolha_orientacao in {1, 2}:
            break

        print(
            "Escolha uma opção entre 1 e 2."
        )

    except ValueError:

        print(
            "Digite apenas o número."
        )


orientacao_horizontal = (
    escolha_orientacao == 2
)


# ============================================================
# VALORES SOBRE AS BARRAS
# ============================================================

mostrar_valores = resposta_sim_nao(
    "\nDeseja mostrar os valores sobre as barras? [S/N]: "
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

    if coluna_hue:

        titulo = (
            f"Contagem de {coluna_principal} "
            f"por {coluna_hue}"
        )

    else:

        titulo = (
            f"Contagem de {coluna_principal}"
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


if coluna_hue:

    ax = sns.countplot(
        data=dados,
        x=coluna_principal,
        hue=coluna_hue,
        order=categorias_ordenadas,
        hue_order=sorted(
            dados[coluna_hue].unique(),
            key=str
        ),
        alpha=alpha
    )

else:

    ax = sns.countplot(
        data=dados,
        x=coluna_principal,
        order=categorias_ordenadas,
        alpha=alpha
    )


# ============================================================
# ORIENTAÇÃO HORIZONTAL
# ============================================================

if orientacao_horizontal:

    # Remove o gráfico atual
    plt.close()

    plt.figure(
        figsize=(10, 6)
    )

    if coluna_hue:

        ax = sns.countplot(
            data=dados,
            y=coluna_principal,
            hue=coluna_hue,
            order=categorias_ordenadas,
            hue_order=sorted(
                dados[coluna_hue].unique(),
                key=str
            ),
            alpha=alpha
        )

    else:

        ax = sns.countplot(
            data=dados,
            y=coluna_principal,
            order=categorias_ordenadas,
            alpha=alpha
        )


# ============================================================
# VALORES SOBRE AS BARRAS
# ============================================================

if mostrar_valores:

    if orientacao_horizontal:

        for container in ax.containers:

            try:

                ax.bar_label(
                    container,
                    fmt="%d",
                    padding=3
                )

            except (AttributeError, TypeError):

                pass

    else:

        for container in ax.containers:

            try:

                ax.bar_label(
                    container,
                    fmt="%d",
                    padding=3
                )

            except (AttributeError, TypeError):

                pass


# ============================================================
# CONFIGURAÇÃO FINAL
# ============================================================

plt.title(
    titulo
)

if orientacao_horizontal:

    plt.xlabel(
        "Contagem"
    )

    plt.ylabel(
        coluna_principal
    )

else:

    plt.xlabel(
        coluna_principal
    )

    plt.ylabel(
        "Contagem"
    )


plt.tight_layout()


# ============================================================
# NOME DO ARQUIVO
# ============================================================

nome_saida = (
    f"006_Barras_"
    f"{nome_seguro(coluna_principal)}"
)


if coluna_hue:

    nome_saida += (
        f"_por_"
        f"{nome_seguro(coluna_hue)}"
    )


if limite_categorias is not None:

    nome_saida += (
        f"_top_{limite_categorias}"
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

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA")
print("=" * 60)

print(
    f"\nCategoria analisada: "
    f"{coluna_principal}"
)

if coluna_hue:

    print(
        f"Hue utilizado: "
        f"{coluna_hue}"
    )

print(
    f"Categorias exibidas: "
    f"{len(categorias_ordenadas)}"
)

print(
    f"Linhas utilizadas: "
    f"{len(dados)}"
)

print(
    "\nGráfico salvo em:"
)

print(
    caminho_saida
)