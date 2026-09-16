from utilidades import ler_arquivo
from pathlib import Path
import pandas as pd
import time


# ============================================================
# CONFIGURAÇÃO DOS CAMINHOS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"

# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def obter_moda(serie):
    valores = serie.dropna()

    if valores.empty:
        return None

    moda = valores.mode()

    if moda.empty:
        return None

    return moda.iloc[0]


def formatar_valor(valor):
    if valor is None:
        return "-"

    if pd.isna(valor):
        return "-"

    if isinstance(valor, float):
        return f"{valor:.4f}"

    return str(valor)


def obter_tipo(serie):

    if pd.api.types.is_bool_dtype(serie):
        return "Booleano"

    elif pd.api.types.is_datetime64_any_dtype(serie):
        return "Data/Hora"

    elif pd.api.types.is_numeric_dtype(serie):
        return "Numérico"

    else:
        return "Categórico/Textual"


def obter_colunas_com_nulos(df):
    return [
        coluna
        for coluna in df.columns
        if df[coluna].isna().sum() > 0
    ]


# ============================================================
# MOSTRAR COLUNAS COM VALORES AUSENTES
# ============================================================

def mostrar_colunas_com_nulos(df):

    colunas_nulas = obter_colunas_com_nulos(df)

    print("\n" + "=" * 70)
    print("COLUNAS COM VALORES AUSENTES")
    print("=" * 70)

    if not colunas_nulas:
        print("\nNenhuma coluna possui valores ausentes.")
        return

    total_linhas = len(df)

    for coluna in colunas_nulas:

        quantidade = df[coluna].isna().sum()
        percentual = (quantidade / total_linhas) * 100

        print(
            f"\n{coluna}"
        )

        print(
            f"  Valores ausentes: {quantidade:,}"
        )

        print(
            f"  Percentual: {percentual:.2f}%"
        )

    print("\n" + "-" * 70)
    print(
        f"Total de colunas com valores ausentes: "
        f"{len(colunas_nulas)}"
    )


# ============================================================
# MOSTRAR INFORMAÇÕES DA COLUNA
# ============================================================

def mostrar_informacoes_coluna(df, coluna):

    serie = df[coluna]

    tipo = obter_tipo(serie)

    total = len(serie)
    nulos = serie.isna().sum()

    if total > 0:
        percentual_nulos = (nulos / total) * 100
    else:
        percentual_nulos = 0

    valores_validos = serie.dropna()

    print("\n" + "=" * 70)
    print(f"COLUNA: {coluna}")
    print("=" * 70)

    print(f"Tipo de dado: {tipo}")

    print(f"Total de registros: {total:,}")

    print(
        f"Valores ausentes: "
        f"{nulos:,}"
    )

    print(
        f"Percentual de valores ausentes: "
        f"{percentual_nulos:.2f}%"
    )

    print(
        f"Valores não nulos: "
        f"{len(valores_validos):,}"
    )

    # --------------------------------------------------------
    # MÉDIA
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):

        if not valores_validos.empty:
            media = valores_validos.mean()
            print(f"Média: {formatar_valor(media)}")
        else:
            print("Média: -")

    else:
        print("Média: não aplicável")


    # --------------------------------------------------------
    # MEDIANA
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):

        if not valores_validos.empty:
            mediana = valores_validos.median()
            print(f"Mediana: {formatar_valor(mediana)}")
        else:
            print("Mediana: -")

    else:
        print("Mediana: não aplicável")


    # --------------------------------------------------------
    # MODA
    # --------------------------------------------------------

    moda = obter_moda(serie)

    if moda is not None:
        print(
            f"Moda: "
            f"{formatar_valor(moda)}"
        )
    else:
        print("Moda: -")


    # --------------------------------------------------------
    # MÍNIMO E MÁXIMO
    # --------------------------------------------------------

    if not valores_validos.empty:

        try:
            minimo = valores_validos.min()
            maximo = valores_validos.max()

            print(
                f"Mínimo: "
                f"{formatar_valor(minimo)}"
            )

            print(
                f"Máximo: "
                f"{formatar_valor(maximo)}"
            )

        except TypeError:

            print("Mínimo: não aplicável")
            print("Máximo: não aplicável")

    else:

        print("Mínimo: -")
        print("Máximo: -")


# ============================================================
# TRATAMENTO POR REMOÇÃO
# ============================================================

def remover_nulos(df, coluna):

    quantidade_antes = len(df)

    df = df.dropna(subset=[coluna])

    quantidade_depois = len(df)

    removidas = quantidade_antes - quantidade_depois

    print(
        f"\n{removidas:,} linhas foram removidas."
    )

    return df


# ============================================================
# TRATAMENTO POR SUBSTITUIÇÃO
# ============================================================

def substituir_nulos(df, coluna, metodo):

    serie = df[coluna]

    if metodo == "media":

        if not (
            pd.api.types.is_numeric_dtype(serie)
            and not pd.api.types.is_bool_dtype(serie)
        ):
            print(
                "\nA média não pode ser utilizada "
                "nesta coluna."
            )
            return df, False

        valor = serie.mean()

        if pd.isna(valor):
            print(
                "\nNão foi possível calcular a média."
            )
            return df, False

        df[coluna] = serie.fillna(valor)

        print(
            f"\nValores ausentes substituídos pela média: "
            f"{valor:.4f}"
        )

        return df, True


    elif metodo == "mediana":

        if not (
            pd.api.types.is_numeric_dtype(serie)
            and not pd.api.types.is_bool_dtype(serie)
        ):
            print(
                "\nA mediana não pode ser utilizada "
                "nesta coluna."
            )
            return df, False

        valor = serie.median()

        if pd.isna(valor):
            print(
                "\nNão foi possível calcular a mediana."
            )
            return df, False

        df[coluna] = serie.fillna(valor)

        print(
            f"\nValores ausentes substituídos pela mediana: "
            f"{valor:.4f}"
        )

        return df, True


    elif metodo == "moda":

        valor = obter_moda(serie)

        if valor is None:

            print(
                "\nNão foi possível calcular a moda."
            )

            return df, False

        df[coluna] = serie.fillna(valor)

        print(
            f"\nValores ausentes substituídos pela moda: "
            f"{formatar_valor(valor)}"
        )

        return df, True


    elif metodo == "novo":

        while True:

            novo_valor = input(
                "\nDigite o novo valor para substituir "
                "os valores ausentes: "
            )

            if novo_valor.strip() == "":
                print(
                    "O valor não pode ser vazio."
                )
                continue

            break


        # ----------------------------------------------------
        # Conversão de acordo com o tipo da coluna
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie):

            try:

                if pd.api.types.is_integer_dtype(serie):

                    novo_valor = int(novo_valor)

                else:

                    novo_valor = float(novo_valor)

            except ValueError:

                print(
                    "\nValor inválido para uma coluna numérica."
                )

                return df, False


        elif pd.api.types.is_bool_dtype(serie):

            valor_lower = novo_valor.lower()

            if valor_lower in ["true", "1", "sim", "s"]:

                novo_valor = True

            elif valor_lower in ["false", "0", "nao", "não", "n"]:

                novo_valor = False

            else:

                print(
                    "\nPara uma coluna booleana, "
                    "digite True/False ou Sim/Não."
                )

                return df, False


        elif pd.api.types.is_datetime64_any_dtype(serie):

            try:

                novo_valor = pd.to_datetime(
                    novo_valor
                )

            except Exception:

                print(
                    "\nData inválida."
                )

                return df, False


        df[coluna] = serie.fillna(novo_valor)

        print(
            f"\nValores ausentes substituídos por: "
            f"{formatar_valor(novo_valor)}"
        )

        return df, True


# ============================================================
# MENU DE TRATAMENTO
# ============================================================

def tratar_coluna(df, coluna):

    while True:

        print("\n" + "-" * 70)
        print("ESCOLHA O TRATAMENTO")
        print("-" * 70)

        print("\n1. Remover linhas com valor ausente")
        print("2. Substituir pela média")
        print("3. Substituir pela mediana")
        print("4. Substituir pela moda")
        print("5. Substituir por um novo valor")
        print("6. Não tratar esta coluna agora")

        opcao = input(
            "\nDigite a opção desejada: "
        ).strip()


        # ----------------------------------------------------
        # REMOÇÃO
        # ----------------------------------------------------

        if opcao == "1":

            confirmacao = input(
                "\nATENÇÃO: todas as linhas que possuem "
                "valor ausente nesta coluna serão removidas.\n"
                "Deseja realmente continuar? (s/n): "
            ).strip().lower()

            if confirmacao == "s":

                df = remover_nulos(
                    df,
                    coluna
                )

                return df, True

            else:

                print(
                    "\nOperação cancelada."
                )


        # ----------------------------------------------------
        # MÉDIA
        # ----------------------------------------------------

        elif opcao == "2":

            df, sucesso = substituir_nulos(
                df,
                coluna,
                "media"
            )

            if sucesso:
                return df, True


        # ----------------------------------------------------
        # MEDIANA
        # ----------------------------------------------------

        elif opcao == "3":

            df, sucesso = substituir_nulos(
                df,
                coluna,
                "mediana"
            )

            if sucesso:
                return df, True


        # ----------------------------------------------------
        # MODA
        # ----------------------------------------------------

        elif opcao == "4":

            df, sucesso = substituir_nulos(
                df,
                coluna,
                "moda"
            )

            if sucesso:
                return df, True


        # ----------------------------------------------------
        # NOVO VALOR
        # ----------------------------------------------------

        elif opcao == "5":

            df, sucesso = substituir_nulos(
                df,
                coluna,
                "novo"
            )

            if sucesso:
                return df, True


        # ----------------------------------------------------
        # NÃO TRATAR
        # ----------------------------------------------------

        elif opcao == "6":

            print(
                "\nColuna não tratada."
            )

            return df, False


        else:

            print(
                "\nOpção inválida."
            )


# ============================================================
# INÍCIO DO PROGRAMA
# ============================================================

print("=" * 70)
print("TRATAMENTO DE VALORES AUSENTES")
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
        "\nNenhum arquivo encontrado na pasta ENTRADA."
    )

    raise SystemExit


print("\nARQUIVOS DISPONÍVEIS:")
print("-" * 70)

for i, arquivo in enumerate(arquivos, start=1):

    print(
        f"{i}. {arquivo.name}"
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
            "Digite apenas o número do arquivo."
        )


arquivo_selecionado = arquivos[
    escolha - 1
]


print(
    f"\nOk, vamos trabalhar com o documento: "
    f"{arquivo_selecionado.name}"
)

time.sleep(2)


# ============================================================
# LEITURA
# ============================================================

try:

    df = ler_arquivo(
        arquivo_selecionado
    )

except Exception as erro:

    print(
        "\nERRO AO LER O ARQUIVO:"
    )

    print(erro)

    raise SystemExit


print(
    f"\nDataset carregado com "
    f"{len(df):,} linhas e "
    f"{len(df.columns):,} colunas."
)


# ============================================================
# DIAGNÓSTICO INICIAL
# ============================================================

mostrar_colunas_com_nulos(df)


colunas_nulas = obter_colunas_com_nulos(df)


if not colunas_nulas:

    print(
        "\nNão há valores ausentes para tratar."
    )

    raise SystemExit


# ============================================================
# PERGUNTA SE DESEJA COMEÇAR
# ============================================================

while True:

    iniciar = input(
        "\nDeseja começar o tratamento "
        "dos valores ausentes? (s/n): "
    ).strip().lower()

    if iniciar in ["s", "n"]:
        break

    print(
        "Digite 's' para sim ou 'n' para não."
    )


if iniciar == "n":

    print(
        "\nTratamento cancelado pelo usuário."
    )

    raise SystemExit


# ============================================================
# TRATAMENTO DAS COLUNAS
# ============================================================

while True:

    # --------------------------------------------------------
    # Atualiza a lista depois de cada tratamento
    # --------------------------------------------------------

    colunas_nulas = obter_colunas_com_nulos(df)


    if not colunas_nulas:

        print(
            "\n" + "=" * 70
        )

        print(
            "NÃO HÁ MAIS VALORES AUSENTES"
        )

        print(
            "=" * 70
        )

        break


    # --------------------------------------------------------
    # Mostra situação atual
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SITUAÇÃO ATUAL"
    )

    print(
        "=" * 70
    )

    print(
        f"Linhas atuais: {len(df):,}"
    )

    print(
        f"Colunas com valores ausentes: "
        f"{len(colunas_nulas)}"
    )

    print(
        "\nColunas ainda pendentes:"
    )

    for i, coluna in enumerate(
        colunas_nulas,
        start=1
    ):

        quantidade = df[coluna].isna().sum()

        percentual = (
            quantidade / len(df)
        ) * 100

        print(
            f"{i}. {coluna} "
            f"({quantidade:,} nulos - "
            f"{percentual:.2f}%)"
        )


    # --------------------------------------------------------
    # Sempre pega a primeira coluna pendente
    # --------------------------------------------------------

    coluna_atual = colunas_nulas[0]


    print(
        "\n" + "=" * 70
    )

    print(
        f"PRÓXIMA COLUNA: {coluna_atual}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Informações da coluna
    # --------------------------------------------------------

    mostrar_informacoes_coluna(
        df,
        coluna_atual
    )


    # --------------------------------------------------------
    # Tratamento
    # --------------------------------------------------------

    df, tratada = tratar_coluna(
        df,
        coluna_atual
    )


    # --------------------------------------------------------
    # Atualização
    # --------------------------------------------------------

    if tratada:

        print(
            "\nDataset atualizado."
        )

        print(
            f"Linhas atuais: {len(df):,}"
        )

    else:

        print(
            "\nA coluna continua com "
            "valores ausentes."
        )

        # Evita loop infinito caso o usuário
        # escolha não tratar a coluna.
        break


# ============================================================
# DIAGNÓSTICO FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DIAGNÓSTICO FINAL"
)

print(
    "=" * 70
)

colunas_restantes = obter_colunas_com_nulos(df)


if colunas_restantes:

    print(
        "\nAinda existem valores ausentes "
        "nas seguintes colunas:"
    )

    for coluna in colunas_restantes:

        quantidade = df[coluna].isna().sum()

        percentual = (
            quantidade / len(df)
        ) * 100

        print(
            f"- {coluna}: "
            f"{quantidade:,} "
            f"({percentual:.2f}%)"
        )

else:

    print(
        "\nNenhum valor ausente restante."
    )


# ============================================================
# SALVAR CSV FINAL
# ============================================================

nome_saida = (
    f"003_Tratado_Valores_Ausentes_"
    f"{arquivo_selecionado.stem}.csv"
)

caminho_saida = PASTA_SAIDA / nome_saida


df.to_csv(
    caminho_saida,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# FINALIZAÇÃO
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TRATAMENTO CONCLUÍDO"
)

print(
    "=" * 70
)

print(
    f"Arquivo final salvo em:\n"
    f"{caminho_saida}"
)

print(
    f"\nLinhas finais: {len(df):,}"
)

print(
    f"Colunas finais: {len(df.columns):,}"
)