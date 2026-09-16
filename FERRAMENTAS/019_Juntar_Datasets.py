from pathlib import Path
import pandas as pd

from utilidades import ler_arquivo


# ============================================================
# CONFIGURAÇÃO DOS CAMINHOS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_PROJETO / "ENTRADA"
PASTA_SAIDA = PASTA_PROJETO / "SAIDA"


PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOCALIZAR DOCUMENTOS
# ============================================================

documentos = [
    arquivo
    for arquivo in PASTA_ENTRADA.iterdir()
    if (
        arquivo.is_file()
        and arquivo.suffix.lower() in [
            ".csv",
            ".xlsx",
            ".xls",
            ".json",
            ".parquet"
        ]
    )
]


if len(documentos) < 2:

    print()
    print("=" * 70)
    print("JUNTAR DATASETS")
    print("=" * 70)

    print()
    print(
        "É necessário ter pelo menos 2 arquivos "
        "compatíveis na pasta ENTRADA."
    )

    exit()


# ============================================================
# LISTAR DOCUMENTOS
# ============================================================

print()
print("=" * 70)
print("DOCUMENTOS DISPONÍVEIS")
print("=" * 70)

for numero, documento in enumerate(
    documentos,
    start=1
):

    print(
        f"{numero} - {documento.name}"
    )


# ============================================================
# SELEÇÃO DOS DATASETS
# ============================================================

documentos_selecionados = []


print()
print("=" * 70)
print("SELEÇÃO DOS DATASETS")
print("=" * 70)

print(
    "\nDigite o número dos arquivos que deseja juntar."
)

print(
    "Digite 0 quando terminar a seleção."
)


while True:

    try:

        escolha = int(
            input(
                "\nNúmero do arquivo: "
            )
        )

    except ValueError:

        print(
            "Digite apenas um número."
        )

        continue


    # --------------------------------------------------------
    # FINALIZAR SELEÇÃO
    # --------------------------------------------------------

    if escolha == 0:

        if len(documentos_selecionados) < 2:

            print(
                "Selecione pelo menos 2 arquivos."
            )

            continue

        break


    # --------------------------------------------------------
    # VALIDAR NÚMERO
    # --------------------------------------------------------

    if not 1 <= escolha <= len(documentos):

        print(
            "Número inválido. "
            "Escolha um dos arquivos listados."
        )

        continue


    documento = documentos[escolha - 1]


    # --------------------------------------------------------
    # VERIFICAR DUPLICIDADE
    # --------------------------------------------------------

    if documento in documentos_selecionados:

        print(
            "Esse arquivo já foi selecionado."
        )

        continue


    documentos_selecionados.append(
        documento
    )

    print(
        f"Arquivo selecionado: {documento.name}"
    )


# ============================================================
# MOSTRAR DATASETS SELECIONADOS
# ============================================================

print()
print("=" * 70)
print("DATASETS SELECIONADOS")
print("=" * 70)

for numero, documento in enumerate(
    documentos_selecionados,
    start=1
):

    print(
        f"{numero} - {documento.name}"
    )


# ============================================================
# ESCOLHER TIPO DE JUNÇÃO
# ============================================================

print()
print("=" * 70)
print("TIPO DE JUNÇÃO")
print("=" * 70)

print(
    "\n1 - Horizontal"
)

print(
    "    Um dataset ao lado do outro."
)

print(
    "\n2 - Vertical"
)

print(
    "    Um dataset abaixo do outro."
)


while True:

    tipo_juncao = input(
        "\nDigite o número do tipo de junção: "
    ).strip()


    if tipo_juncao in ["1", "2"]:
        break


    print(
        "Opção inválida. "
        "Escolha 1 ou 2."
    )


# ============================================================
# LEITURA DOS DATASETS
# ============================================================

print()
print("=" * 70)
print("CARREGANDO DATASETS")
print("=" * 70)


datasets = []


for documento in documentos_selecionados:

    print(
        f"\nCarregando: {documento.name}"
    )

    try:

        df = ler_arquivo(documento)

        datasets.append(df)

        print(
            f"Linhas: {df.shape[0]:,}"
        )

        print(
            f"Colunas: {df.shape[1]:,}"
        )

    except Exception as erro:

        print()
        print(
            f"Erro ao carregar: {documento.name}"
        )

        print(
            f"Detalhes: {erro}"
        )

        exit()


# ============================================================
# JUNÇÃO HORIZONTAL
# ============================================================

if tipo_juncao == "1":

    print()
    print("=" * 70)
    print("VALIDAÇÃO DA JUNÇÃO HORIZONTAL")
    print("=" * 70)


    quantidade_linhas = [
        len(df)
        for df in datasets
    ]


    print()

    for numero, df in enumerate(
        datasets,
        start=1
    ):

        print(
            f"Dataset {numero}: "
            f"{len(df):,} linhas"
        )


    # --------------------------------------------------------
    # VERIFICAR QUANTIDADE DE LINHAS
    # --------------------------------------------------------

    if len(set(quantidade_linhas)) != 1:

        print()
        print(
            "Não é possível realizar a junção horizontal."
        )

        print(
            "Todos os datasets precisam possuir "
            "o mesmo número de linhas."
        )

        exit()


    # --------------------------------------------------------
    # REALIZAR JUNÇÃO
    # --------------------------------------------------------

    print()
    print(
        "Todos os datasets possuem "
        "o mesmo número de linhas."
    )

    print(
        "Realizando junção horizontal..."
    )


    try:

        dataset_final = pd.concat(
            datasets,
            axis=1
        )

    except Exception as erro:

        print()
        print(
            "Não foi possível realizar a junção."
        )

        print(
            f"Erro: {erro}"
        )

        exit()


# ============================================================
# JUNÇÃO VERTICAL
# ============================================================

else:

    print()
    print("=" * 70)
    print("VALIDAÇÃO DA JUNÇÃO VERTICAL")
    print("=" * 70)


    quantidade_colunas = [
        len(df.columns)
        for df in datasets
    ]


    print()

    for numero, df in enumerate(
        datasets,
        start=1
    ):

        print(
            f"Dataset {numero}: "
            f"{len(df.columns):,} colunas"
        )


    # --------------------------------------------------------
    # VERIFICAR QUANTIDADE DE COLUNAS
    # --------------------------------------------------------

    if len(set(quantidade_colunas)) != 1:

        print()
        print(
            "Não é possível realizar a junção vertical."
        )

        print(
            "Todos os datasets precisam possuir "
            "o mesmo número de colunas."
        )

        exit()


    # --------------------------------------------------------
    # VERIFICAR NOMES DAS COLUNAS
    # --------------------------------------------------------

    colunas_referencia = list(
        datasets[0].columns
    )


    nomes_compativeis = True


    for numero, df in enumerate(
        datasets[1:],
        start=2
    ):

        colunas_atual = list(
            df.columns
        )


        if colunas_atual != colunas_referencia:

            nomes_compativeis = False

            print()
            print(
                f"As colunas do Dataset {numero} "
                "não são compatíveis."
            )

            print()
            print(
                "Colunas esperadas:"
            )

            for coluna in colunas_referencia:

                print(
                    f"  - {coluna}"
                )


            print()
            print(
                f"Colunas encontradas no Dataset {numero}:"
            )

            for coluna in colunas_atual:

                print(
                    f"  - {coluna}"
                )

            break


    if not nomes_compativeis:

        print()
        print(
            "A junção vertical foi cancelada."
        )

        print(
            "Os datasets precisam possuir "
            "as mesmas colunas, na mesma ordem."
        )

        exit()


    # --------------------------------------------------------
    # REALIZAR JUNÇÃO
    # --------------------------------------------------------

    print()
    print(
        "Todos os datasets possuem "
        "as mesmas colunas."
    )

    print(
        "Realizando junção vertical..."
    )


    try:

        dataset_final = pd.concat(
            datasets,
            axis=0,
            ignore_index=True
        )

    except Exception as erro:

        print()
        print(
            "Não foi possível realizar a junção."
        )

        print(
            f"Erro: {erro}"
        )

        exit()


# ============================================================
# INFORMAÇÕES DO DATASET FINAL
# ============================================================

print()
print("=" * 70)
print("JUNÇÃO CONCLUÍDA")
print("=" * 70)

print()
print(
    f"Linhas finais: "
    f"{dataset_final.shape[0]:,}"
)

print(
    f"Colunas finais: "
    f"{dataset_final.shape[1]:,}"
)


# ============================================================
# NOME DO ARQUIVO DE SAÍDA
# ============================================================

nomes = [
    documento.stem
    for documento in documentos_selecionados
]


nome_saida = (
    "Juntado_"
    + "_".join(nomes)
    + ".csv"
)


arquivo_saida = (
    PASTA_SAIDA / nome_saida
)


# ============================================================
# SALVAR DATASET
# ============================================================

try:

    dataset_final.to_csv(
        arquivo_saida,
        index=False,
        encoding="utf-8-sig"
    )

except Exception as erro:

    print()
    print(
        "Não foi possível salvar o arquivo."
    )

    print(
        f"Erro: {erro}"
    )

    exit()


# ============================================================
# FINALIZAÇÃO
# ============================================================

print()
print("=" * 70)
print("ARQUIVO SALVO")
print("=" * 70)

print(
    f"Arquivo: {arquivo_saida.name}"
)

print(
    f"Local: {arquivo_saida}"
)