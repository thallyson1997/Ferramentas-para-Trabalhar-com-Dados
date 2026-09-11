from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURAÇÃO DOS CAMINHOS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"


# ============================================================
# LEITURA DOS ARQUIVOS
# ============================================================

def ler_arquivo(caminho):

    extensao = caminho.suffix.lower()

    if extensao == ".csv":
        return pd.read_csv(caminho)

    elif extensao in [".xlsx", ".xls"]:
        return pd.read_excel(caminho)

    elif extensao == ".json":
        return pd.read_json(caminho)

    elif extensao == ".parquet":
        return pd.read_parquet(caminho)

    else:
        raise ValueError(
            f"Formato não suportado: {extensao}"
        )


# ============================================================
# MOSTRAR COLUNAS
# ============================================================

def mostrar_colunas(df):

    print("\n" + "=" * 70)
    print("COLUNAS DO DATASET")
    print("=" * 70)

    for numero, coluna in enumerate(df.columns, start=1):

        print(
            f"{numero}. {coluna}"
        )


# ============================================================
# LER LISTA DE NÚMEROS
# ============================================================

def ler_lista_colunas():

    while True:

        entrada = input(
            "\nDigite os números das colunas "
            "separados por vírgula: "
        ).strip()

        try:

            numeros = [
                int(numero.strip())
                for numero in entrada.split(",")
                if numero.strip()
            ]

            if not numeros:
                raise ValueError

            if len(numeros) != len(set(numeros)):

                print(
                    "\nVocê informou alguma coluna mais de uma vez."
                )

                continue

            return numeros

        except ValueError:

            print(
                "\nEntrada inválida."
            )

            print(
                "Exemplo: 1, 3, 5, 8"
            )


# ============================================================
# VALIDAR NÚMEROS DAS COLUNAS
# ============================================================

def validar_colunas(numeros, total_colunas):

    invalidas = [
        numero
        for numero in numeros
        if numero < 1 or numero > total_colunas
    ]

    if invalidas:

        print(
            "\nOs seguintes números são inválidos:"
        )

        print(
            ", ".join(map(str, invalidas))
        )

        print(
            f"Escolha números entre 1 e {total_colunas}."
        )

        return False

    return True


# ============================================================
# CONVERTER NÚMEROS EM NOMES DE COLUNAS
# ============================================================

def numeros_para_colunas(df, numeros):

    return [
        df.columns[numero - 1]
        for numero in numeros
    ]


# ============================================================
# ESCOLHA DAS COLUNAS
# ============================================================

def escolher_colunas(df):

    total_colunas = len(df.columns)

    while True:

        print("\n" + "=" * 70)
        print("COMO DESEJA VERIFICAR AS DUPLICATAS?")
        print("=" * 70)

        print(
            "\n1. Usar todas as colunas"
        )

        print(
            "2. Usar todas as colunas, "
            "exceto algumas"
        )

        print(
            "3. Usar apenas algumas colunas"
        )

        opcao = input(
            "\nDigite a opção desejada: "
        ).strip()


        # ----------------------------------------------------
        # TODAS AS COLUNAS
        # ----------------------------------------------------

        if opcao == "1":

            colunas = list(df.columns)

            print(
                "\nTodas as colunas serão utilizadas "
                "para verificar duplicatas."
            )

            return colunas


        # ----------------------------------------------------
        # TODAS, EXCETO ALGUMAS
        # ----------------------------------------------------

        elif opcao == "2":

            mostrar_colunas(df)

            print(
                "\nInforme as colunas que NÃO serão "
                "utilizadas na verificação."
            )

            numeros_excluir = ler_lista_colunas()

            if not validar_colunas(
                numeros_excluir,
                total_colunas
            ):
                continue

            numeros_utilizar = [
                numero
                for numero in range(
                    1,
                    total_colunas + 1
                )
                if numero not in numeros_excluir
            ]

            if not numeros_utilizar:

                print(
                    "\nVocê não pode excluir todas "
                    "as colunas."
                )

                continue

            colunas = numeros_para_colunas(
                df,
                numeros_utilizar
            )

            print(
                "\nColunas que serão ignoradas:"
            )

            for numero in numeros_excluir:

                print(
                    f"- {df.columns[numero - 1]}"
                )

            return colunas


        # ----------------------------------------------------
        # APENAS ALGUMAS
        # ----------------------------------------------------

        elif opcao == "3":

            mostrar_colunas(df)

            print(
                "\nInforme as colunas que serão "
                "utilizadas na verificação."
            )

            numeros_utilizar = ler_lista_colunas()

            if not validar_colunas(
                numeros_utilizar,
                total_colunas
            ):
                continue

            colunas = numeros_para_colunas(
                df,
                numeros_utilizar
            )

            return colunas


        else:

            print(
                "\nOpção inválida."
            )


# ============================================================
# MOSTRAR COLUNAS SELECIONADAS
# ============================================================

def mostrar_colunas_selecionadas(colunas):

    print("\n" + "=" * 70)
    print("COLUNAS UTILIZADAS NA VERIFICAÇÃO")
    print("=" * 70)

    for numero, coluna in enumerate(
        colunas,
        start=1
    ):

        print(
            f"{numero}. {coluna}"
        )


# ============================================================
# ANALISAR DUPLICATAS
# ============================================================

def analisar_duplicatas(df, colunas):

    mascara_duplicatas = df.duplicated(
        subset=colunas,
        keep=False
    )

    linhas_duplicadas = df[
        mascara_duplicatas
    ]

    quantidade_linhas_duplicadas = len(
        linhas_duplicadas
    )

    quantidade_linhas_removidas = (
        df.duplicated(
            subset=colunas,
            keep="first"
        ).sum()
    )

    grupos_duplicados = (
        df.loc[
            mascara_duplicatas,
            colunas
        ]
        .value_counts()
    )

    quantidade_grupos = len(
        grupos_duplicados
    )

    return (
        mascara_duplicatas,
        quantidade_linhas_duplicadas,
        quantidade_linhas_removidas,
        quantidade_grupos,
        grupos_duplicados
    )


# ============================================================
# MOSTRAR RESULTADO DA ANÁLISE
# ============================================================

def mostrar_analise(
    df,
    colunas,
    mascara_duplicatas,
    quantidade_linhas_duplicadas,
    quantidade_linhas_removidas,
    quantidade_grupos,
    grupos_duplicados
):

    print("\n" + "=" * 70)
    print("ANÁLISE DE DUPLICATAS")
    print("=" * 70)

    print(
        f"\nTotal de linhas do dataset: "
        f"{len(df):,}"
    )

    print(
        f"Linhas envolvidas em duplicações: "
        f"{quantidade_linhas_duplicadas:,}"
    )

    print(
        f"Grupos de valores duplicados: "
        f"{quantidade_grupos:,}"
    )

    print(
        f"Linhas que serão removidas: "
        f"{quantidade_linhas_removidas:,}"
    )

    linhas_restantes = (
        len(df) -
        quantidade_linhas_removidas
    )

    print(
        f"Linhas após a remoção: "
        f"{linhas_restantes:,}"
    )


    # --------------------------------------------------------
    # NÃO HÁ DUPLICATAS
    # --------------------------------------------------------

    if quantidade_linhas_duplicadas == 0:

        print(
            "\nNenhuma duplicata foi encontrada "
            "com base nas colunas selecionadas."
        )

        return


    # --------------------------------------------------------
    # MOSTRAR GRUPOS
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "GRUPOS DE DUPLICATAS ENCONTRADOS"
    )

    print(
        "-" * 70
    )

    for valores, quantidade in grupos_duplicados.items():

        if not isinstance(valores, tuple):
            valores = (valores,)

        descricao = " | ".join(
            [
                f"{coluna}={valor}"
                for coluna, valor in zip(
                    colunas,
                    valores
                )
            ]
        )

        print(
            f"\n{descricao}"
        )

        print(
            f"Quantidade de ocorrências: "
            f"{quantidade:,}"
        )


# ============================================================
# REMOVER DUPLICATAS
# ============================================================

def remover_duplicatas(df, colunas):

    quantidade_antes = len(df)

    df_tratado = df.drop_duplicates(
        subset=colunas,
        keep="first"
    ).copy()

    quantidade_depois = len(
        df_tratado
    )

    removidas = (
        quantidade_antes -
        quantidade_depois
    )

    return (
        df_tratado,
        removidas
    )


# ============================================================
# INÍCIO DO PROGRAMA
# ============================================================

print("=" * 70)
print("REMOÇÃO DE DUPLICATAS")
print("=" * 70)


# ============================================================
# ARQUIVOS DISPONÍVEIS
# ============================================================

arquivos = [
    arquivo
    for arquivo in PASTA_ENTRADA.iterdir()
    if arquivo.is_file()
]


if not arquivos:

    print(
        "\nNenhum arquivo encontrado "
        "na pasta ENTRADA."
    )

    raise SystemExit


print("\nARQUIVOS DISPONÍVEIS:")
print("-" * 70)

for numero, arquivo in enumerate(
    arquivos,
    start=1
):

    print(
        f"{numero}. {arquivo.name}"
    )


# ============================================================
# ESCOLHA DO ARQUIVO
# ============================================================

while True:

    try:

        escolha = int(
            input(
                "\nDigite o número do arquivo "
                "que deseja analisar: "
            )
        )

        if 1 <= escolha <= len(arquivos):
            break

        print(
            "Número inválido."
        )

    except ValueError:

        print(
            "Digite apenas o número "
            "correspondente ao arquivo."
        )


arquivo_selecionado = arquivos[
    escolha - 1
]


print(
    f"\nOk, vamos trabalhar com o documento: "
    f"{arquivo_selecionado.name}"
)


# ============================================================
# LEITURA DO DATASET
# ============================================================

try:

    df = ler_arquivo(
        arquivo_selecionado
    )

except Exception as erro:

    print(
        "\nERRO AO LER O ARQUIVO:"
    )

    print(
        erro
    )

    raise SystemExit


print(
    f"\nDataset carregado com "
    f"{len(df):,} linhas e "
    f"{len(df.columns):,} colunas."
)


# ============================================================
# ESCOLHA DAS COLUNAS
# ============================================================

colunas_selecionadas = escolher_colunas(
    df
)


# ============================================================
# MOSTRAR SELEÇÃO
# ============================================================

mostrar_colunas_selecionadas(
    colunas_selecionadas
)


# ============================================================
# ANALISAR DUPLICATAS
# ============================================================

(
    mascara_duplicatas,
    quantidade_linhas_duplicadas,
    quantidade_linhas_removidas,
    quantidade_grupos,
    grupos_duplicados
) = analisar_duplicatas(
    df,
    colunas_selecionadas
)


# ============================================================
# MOSTRAR ANÁLISE
# ============================================================

mostrar_analise(
    df,
    colunas_selecionadas,
    mascara_duplicatas,
    quantidade_linhas_duplicadas,
    quantidade_linhas_removidas,
    quantidade_grupos,
    grupos_duplicados
)


# ============================================================
# SE NÃO HOUVER DUPLICATAS
# ============================================================

if quantidade_linhas_duplicadas == 0:

    print(
        "\nNenhum tratamento necessário."
    )

    raise SystemExit


# ============================================================
# PERGUNTAR SE DESEJA REMOVER
# ============================================================

while True:

    resposta = input(
        "\nDeseja remover as duplicatas? (s/n): "
    ).strip().lower()

    if resposta in ["s", "n"]:
        break

    print(
        "Digite 's' para sim ou 'n' para não."
    )


# ============================================================
# CANCELAMENTO
# ============================================================

if resposta == "n":

    print(
        "\nAs duplicatas não foram removidas."
    )

    print(
        "O dataset original permanece inalterado."
    )

    raise SystemExit


# ============================================================
# REMOÇÃO
# ============================================================

df_tratado, removidas = remover_duplicatas(
    df,
    colunas_selecionadas
)


# ============================================================
# SALVAR CSV
# ============================================================

nome_saida = (
    f"004_Sem_Duplicatas_"
    f"{arquivo_selecionado.stem}.csv"
)

caminho_saida = (
    PASTA_SAIDA /
    nome_saida
)


df_tratado.to_csv(
    caminho_saida,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DUPLICATAS REMOVIDAS"
)

print(
    "=" * 70
)

print(
    f"Linhas antes: "
    f"{len(df):,}"
)

print(
    f"Linhas removidas: "
    f"{removidas:,}"
)

print(
    f"Linhas depois: "
    f"{len(df_tratado):,}"
)

print(
    f"\nArquivo tratado salvo em:"
)

print(
    caminho_saida
)