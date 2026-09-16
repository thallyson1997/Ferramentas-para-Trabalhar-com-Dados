from utilidades import ler_arquivo
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAÇÃO DOS CAMINHOS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_PROJETO / "ENTRADA"
PASTA_SAIDA = PASTA_PROJETO / "SAIDA"


# ============================================================
# FUNÇÃO PARA EXIBIR E ARMAZENAR A SAÍDA
# ============================================================

relatorio = []


def exibir(texto=""):
    """
    Exibe a informação no terminal e também
    armazena a informação para o arquivo TXT.
    """

    print(texto)
    relatorio.append(texto)


# ============================================================
# LOCALIZAR DOCUMENTOS
# ============================================================

documentos = [
    arquivo
    for arquivo in PASTA_ENTRADA.iterdir()
    if arquivo.is_file()
]


if not documentos:

    exibir("Nenhum documento foi encontrado na pasta ENTRADA.")
    exit()


# ============================================================
# LISTAR DOCUMENTOS
# ============================================================

exibir()
exibir("=" * 70)
exibir("DOCUMENTOS DISPONÍVEIS")
exibir("=" * 70)

for numero, documento in enumerate(documentos, start=1):
    exibir(f"{numero} - {documento.name}")


# ============================================================
# SELECIONAR DOCUMENTO
# ============================================================

while True:

    try:

        escolha = int(
            input(
                "\nDigite o número do documento que deseja analisar: "
            )
        )

        if 1 <= escolha <= len(documentos):
            break

        print(
            "Número inválido. Escolha um dos documentos listados."
        )

    except ValueError:

        print(
            "Digite apenas o número correspondente ao documento."
        )


arquivo_selecionado = documentos[escolha - 1]


# ============================================================
# LEITURA DO DATASET
# ============================================================

exibir()
exibir("=" * 70)
exibir("CARREGANDO DATASET")
exibir("=" * 70)

exibir(
    f"Arquivo selecionado: {arquivo_selecionado.name}"
)


try:

    df = ler_arquivo(arquivo_selecionado)

except Exception as erro:

    exibir()
    exibir("Não foi possível ler o arquivo.")
    exibir(f"Erro: {erro}")

    exit()


# ============================================================
# ESTRUTURA GERAL
# ============================================================

exibir()
exibir("=" * 70)
exibir("ESTRUTURA GERAL DO DATASET")
exibir("=" * 70)

exibir(
    f"Arquivo: {arquivo_selecionado.name}"
)

exibir(
    f"Linhas: {df.shape[0]:,}"
)

exibir(
    f"Colunas: {df.shape[1]:,}"
)


# ============================================================
# INFORMAÇÕES DAS COLUNAS
# ============================================================

exibir()
exibir("=" * 70)
exibir("INFORMAÇÕES DAS COLUNAS")
exibir("=" * 70)


informacoes = pd.DataFrame({

    "Coluna": df.columns,

    "Tipo": df.dtypes.astype(str).values,

    "Não nulos": df.notna().sum().values,

    "Nulos": df.isna().sum().values,

    "% Nulos": (
        df.isna().mean().values * 100
    ).round(2),

    "Valores únicos": (
        df.nunique(dropna=True).values
    )

})


# Converter a tabela para texto
# para que ela seja exibida e salva exatamente igual.

tabela_colunas = informacoes.to_string(
    index=False
)

exibir(tabela_colunas)


# ============================================================
# POSSÍVEIS IDENTIFICADORES
# ============================================================

exibir()
exibir("=" * 70)
exibir("POSSÍVEIS IDENTIFICADORES")
exibir("=" * 70)


possiveis_ids = []


for coluna in df.columns:

    quantidade_unicos = df[coluna].nunique(
        dropna=True
    )

    if (
        quantidade_unicos == len(df)
        and len(df) > 0
    ):

        possiveis_ids.append(coluna)


if possiveis_ids:

    exibir()
    exibir(
        "Colunas onde todos os valores são únicos:"
    )

    for coluna in possiveis_ids:
        exibir(f"- {coluna}")

    exibir()
    exibir(
        "Essas colunas podem representar identificadores."
    )

else:

    exibir()
    exibir(
        "Nenhuma coluna possui valores totalmente únicos."
    )


# ============================================================
# TIPOS DE VARIÁVEIS
# ============================================================

exibir()
exibir("=" * 70)
exibir("RESUMO DOS TIPOS DE VARIÁVEIS")
exibir("=" * 70)


numericas = df.select_dtypes(
    include="number"
).columns


categoricas = df.select_dtypes(
    include=["str", "object", "category"]
).columns


datas = df.select_dtypes(
    include=["datetime"]
).columns


booleanas = df.select_dtypes(
    include="bool"
).columns


exibir(
    f"\nVariáveis numéricas: {len(numericas)}"
)

exibir(
    f"Variáveis categóricas/textuais: {len(categoricas)}"
)

exibir(
    f"Variáveis de data/hora: {len(datas)}"
)

exibir(
    f"Variáveis booleanas: {len(booleanas)}"
)


# ============================================================
# LISTAGEM DOS TIPOS
# ============================================================

if len(numericas) > 0:

    exibir()
    exibir("Numéricas:")

    for coluna in numericas:
        exibir(f"  - {coluna}")


if len(categoricas) > 0:

    exibir()
    exibir("Categóricas/Textuais:")

    for coluna in categoricas:
        exibir(f"  - {coluna}")


if len(datas) > 0:

    exibir()
    exibir("Data/Hora:")

    for coluna in datas:
        exibir(f"  - {coluna}")


if len(booleanas) > 0:

    exibir()
    exibir("Booleanas:")

    for coluna in booleanas:
        exibir(f"  - {coluna}")


# ============================================================
# FINALIZAÇÃO
# ============================================================

exibir()
exibir("=" * 70)
exibir("ANÁLISE ESTRUTURAL CONCLUÍDA")
exibir("=" * 70)


# ============================================================
# SALVAR RELATÓRIO
# ============================================================

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


nome_saida = (
    f"001_Estrutura_Geral_"
    f"{arquivo_selecionado.stem}.txt"
)


arquivo_saida = PASTA_SAIDA / nome_saida


with open(
    arquivo_saida,
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write(
        "\n".join(relatorio)
    )


print()
print("=" * 70)
print("RELATÓRIO SALVO")
print("=" * 70)
print(f"Arquivo: {arquivo_saida.name}")
print(f"Local: {arquivo_saida}")