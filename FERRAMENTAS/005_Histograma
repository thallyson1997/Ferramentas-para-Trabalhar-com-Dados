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
        raise ValueError("Formato de arquivo não suportado.")


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


def colunas_numericas(df):
    return [
        coluna
        for coluna in df.columns
        if (
            pd.api.types.is_numeric_dtype(df[coluna])
            and not pd.api.types.is_bool_dtype(df[coluna])
        )
    ]


def colunas_hue(df):
    """
    Retorna colunas que podem ser utilizadas como hue.

    São aceitas:
    - colunas categóricas/textuais;
    - booleanas;
    - colunas numéricas com até 10 valores únicos.
    """

    candidatas = []

    for coluna in df.columns:

        serie = df[coluna]

        quantidade_unicos = serie.nunique(dropna=True)

        eh_categorica = (
            pd.api.types.is_object_dtype(serie)
            or pd.api.types.is_string_dtype(serie)
            or isinstance(serie.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(serie)
        )

        eh_numerica_com_poucos_valores = (
            pd.api.types.is_numeric_dtype(serie)
            and not pd.api.types.is_bool_dtype(serie)
            and quantidade_unicos <= 10
        )

        if eh_categorica or eh_numerica_com_poucos_valores:
            candidatas.append(coluna)

    return candidatas


def selecionar_intervalo(serie):
    """
    Permite selecionar um intervalo de valores para a visualização.

    Exemplos:
    2:100  -> de 2 até 100
    :100   -> do menor valor até 100
    100:   -> de 100 até o maior valor
    :      -> todos os valores
    """

    serie_valida = serie.dropna()

    menor_valor = serie_valida.min()
    maior_valor = serie_valida.max()

    print(f"\nMenor valor disponível: {menor_valor}")
    print(f"Maior valor disponível: {maior_valor}")

    print("\nDigite o intervalo no formato menor:maior.")
    print("Deixe um dos lados vazio para usar o limite original.")
    print("Exemplos:")
    print("  2:100")
    print("  :100")
    print("  100:")
    print("  :")

    while True:

        intervalo = input("\nIntervalo: ").strip()

        partes = intervalo.split(":")

        if len(partes) != 2:
            print("Formato inválido. Use o formato menor:maior.")
            continue

        limite_inferior, limite_superior = partes

        try:

            # ------------------------------------------------
            # LIMITE INFERIOR
            # ------------------------------------------------

            if limite_inferior.strip() == "":
                limite_inferior = menor_valor
            else:
                limite_inferior = float(limite_inferior)

            # ------------------------------------------------
            # LIMITE SUPERIOR
            # ------------------------------------------------

            if limite_superior.strip() == "":
                limite_superior = maior_valor
            else:
                limite_superior = float(limite_superior)

        except ValueError:

            print(
                "Intervalo inválido. "
                "Digite apenas valores numéricos."
            )

            continue

        # ----------------------------------------------------
        # VALIDAÇÃO DA ORDEM
        # ----------------------------------------------------

        if limite_inferior > limite_superior:

            print(
                "O limite inferior não pode ser maior "
                "que o limite superior."
            )

            continue

        # ----------------------------------------------------
        # VALIDAÇÃO DO LIMITE INFERIOR
        # ----------------------------------------------------

        if limite_inferior < menor_valor:

            print(
                f"O limite inferior não pode ser menor "
                f"que {menor_valor}."
            )

            continue

        # ----------------------------------------------------
        # VALIDAÇÃO DO LIMITE SUPERIOR
        # ----------------------------------------------------

        if limite_superior > maior_valor:

            print(
                f"O limite superior não pode ser maior "
                f"que {maior_valor}."
            )

            continue

        return limite_inferior, limite_superior


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
print("HISTOGRAMA E DISTRIBUIÇÃO")
print("=" * 60)

print("\nArquivos disponíveis:")

for i, arquivo in enumerate(arquivos, start=1):
    print(f"{i} - {arquivo.name}")


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

    print(f"\nErro ao ler o arquivo: {erro}")

    raise SystemExit


print(
    f"\nArquivo selecionado: "
    f"{caminho_arquivo.name}"
)

print(f"Linhas: {len(df)}")
print(f"Colunas: {len(df.columns)}")


# ============================================================
# SELEÇÃO DA VARIÁVEL NUMÉRICA
# ============================================================

numericas = colunas_numericas(df)

if not numericas:

    print(
        "\nNenhuma coluna numérica foi encontrada."
    )

    raise SystemExit


print("\n" + "-" * 60)
print("COLUNA NUMÉRICA")
print("-" * 60)

print("\nColunas numéricas disponíveis:")

for i, coluna in enumerate(numericas, start=1):

    print(
        f"{i} - {coluna}"
    )


while True:

    try:

        escolha = int(
            input("\nEscolha a coluna numérica: ")
        )

        if 1 <= escolha <= len(numericas):

            coluna_numerica = numericas[escolha - 1]

            break

        print("Escolha uma opção válida.")

    except ValueError:

        print("Digite apenas o número.")


# ============================================================
# HUE
# ============================================================

coluna_hue = None

usar_hue = resposta_sim_nao(
    "\nDeseja adicionar uma informação para diferenciar "
    "a distribuição por grupos (hue)? [S/N]: "
)


if usar_hue:

    hues = colunas_hue(df)

    # A própria coluna numérica não pode ser utilizada como hue
    hues = [
        coluna
        for coluna in hues
        if coluna != coluna_numerica
    ]

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

        while True:

            try:

                escolha = int(
                    input("\nEscolha a coluna para hue: ")
                )

                if 1 <= escolha <= len(hues):

                    coluna_hue = hues[escolha - 1]

                    break

                print("Escolha uma opção válida.")

            except ValueError:

                print("Digite apenas o número.")


# ============================================================
# INTERVALO DOS DADOS
# ============================================================

trabalhar_intervalo = resposta_sim_nao(
    "\nDeseja trabalhar com um intervalo específico? [S/N]: "
)

limite_inferior = None
limite_superior = None


if trabalhar_intervalo:

    limite_inferior, limite_superior = selecionar_intervalo(
        df[coluna_numerica]
    )

    print("\nIntervalo selecionado:")
    print(f"Limite inferior: {limite_inferior}")
    print(f"Limite superior: {limite_superior}")


# ============================================================
# CONFIGURAÇÕES DO HISTOGRAMA
# ============================================================

print("\n" + "=" * 60)
print("CONFIGURAÇÕES DO HISTOGRAMA")
print("=" * 60)


# ------------------------------------------------------------
# KDE
# ------------------------------------------------------------

usar_kde = resposta_sim_nao(
    "\nDeseja adicionar curva de densidade (KDE)? [S/N]: "
)


# ------------------------------------------------------------
# BINS
# ------------------------------------------------------------

definir_bins = resposta_sim_nao(
    "\nDeseja definir o número de bins manualmente? [S/N]: "
)


if definir_bins:

    while True:

        try:

            bins = int(
                input("Digite o número de bins: ")
            )

            if bins > 0:
                break

            print(
                "O número de bins deve ser maior que zero."
            )

        except ValueError:

            print(
                "Digite um número inteiro válido."
            )

else:

    bins = "auto"


# ------------------------------------------------------------
# FORMA DE APRESENTAÇÃO DOS GRUPOS
# ------------------------------------------------------------

multiple = "layer"


if coluna_hue:

    alterar_apresentacao = resposta_sim_nao(
        "\nDeseja alterar a forma de apresentação dos grupos? [S/N]: "
    )

    if alterar_apresentacao:

        print("\nComo deseja apresentar?")
        print("1 - Sobrepostos")
        print("2 - Empilhados")
        print("3 - Lado a lado")
        print("4 - Proporção")

        while True:

            try:

                escolha = int(
                    input("\nEscolha uma opção: ")
                )

                if escolha == 1:

                    multiple = "layer"

                    break

                elif escolha == 2:

                    multiple = "stack"

                    break

                elif escolha == 3:

                    multiple = "dodge"

                    break

                elif escolha == 4:

                    multiple = "fill"

                    break

                print(
                    "Escolha uma opção entre 1 e 4."
                )

            except ValueError:

                print(
                    "Digite apenas o número."
                )

else:

    print(
        "\nA forma de apresentação dos grupos não será "
        "configurada porque nenhum hue foi selecionado."
    )


# ------------------------------------------------------------
# TRANSPARÊNCIA
# ------------------------------------------------------------

alterar_transparencia = resposta_sim_nao(
    "\nDeseja alterar a transparência? [S/N]: "
)


if alterar_transparencia:

    while True:

        try:

            alpha = float(
                input("Digite a transparência (0 a 1): ")
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

    # Transparência padrão.
    #
    # Com hue, utiliza uma transparência maior
    # para facilitar a visualização das distribuições
    # sobrepostas.

    alpha = 0.5 if coluna_hue else 0.8


# ------------------------------------------------------------
# TÍTULO
# ------------------------------------------------------------

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
            f"Distribuição de "
            f"{coluna_numerica} "
            f"por "
            f"{coluna_hue}"
        )

    else:

        titulo = (
            f"Distribuição de "
            f"{coluna_numerica}"
        )


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

if coluna_hue:

    dados = df[
        [
            coluna_numerica,
            coluna_hue
        ]
    ].dropna()

else:

    dados = df[
        [
            coluna_numerica
        ]
    ].dropna()


# ============================================================
# APLICAÇÃO DO INTERVALO
# ============================================================

linhas_antes_intervalo = len(dados)


if trabalhar_intervalo:

    dados = dados[
        (dados[coluna_numerica] >= limite_inferior)
        &
        (dados[coluna_numerica] <= limite_superior)
    ]


if dados.empty:

    print(
        "\nNão existem dados suficientes "
        "para gerar o gráfico."
    )

    raise SystemExit


# ============================================================
# INFORMAÇÕES SOBRE OS DADOS UTILIZADOS
# ============================================================

print("\n" + "-" * 60)
print("DADOS UTILIZADOS")
print("-" * 60)

print(
    f"\nLinhas disponíveis após remoção de valores ausentes: "
    f"{linhas_antes_intervalo}"
)

print(
    f"Linhas utilizadas no gráfico: "
    f"{len(dados)}"
)


if trabalhar_intervalo:

    linhas_removidas = (
        linhas_antes_intervalo
        - len(dados)
    )

    print(
        f"Linhas fora do intervalo: "
        f"{linhas_removidas}"
    )

    print(
        f"Intervalo utilizado: "
        f"{limite_inferior} até {limite_superior}"
    )


# ============================================================
# GERAÇÃO DO HISTOGRAMA
# ============================================================

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))


if coluna_hue:

    sns.histplot(
        data=dados,
        x=coluna_numerica,
        hue=coluna_hue,
        bins=bins,
        kde=usar_kde,
        multiple=multiple,
        alpha=alpha,
        edgecolor="black"
    )

    nome_saida = (
        f"005_Histograma_"
        f"{nome_seguro(coluna_numerica)}"
        f"_por_"
        f"{nome_seguro(coluna_hue)}"
    )

else:

    sns.histplot(
        data=dados,
        x=coluna_numerica,
        bins=bins,
        kde=usar_kde,
        alpha=alpha,
        edgecolor="black"
    )

    nome_saida = (
        f"005_Histograma_"
        f"{nome_seguro(coluna_numerica)}"
    )


# ============================================================
# INFORMAÇÃO DO INTERVALO NO NOME DO ARQUIVO
# ============================================================

if trabalhar_intervalo:

    inferior_nome = nome_seguro(
        str(limite_inferior)
    )

    superior_nome = nome_seguro(
        str(limite_superior)
    )

    nome_saida += (
        f"_intervalo_"
        f"{inferior_nome}_"
        f"{superior_nome}"
    )


nome_saida += ".png"


# ============================================================
# CONFIGURAÇÃO FINAL DO GRÁFICO
# ============================================================

plt.title(titulo)

plt.xlabel(
    coluna_numerica
)

plt.ylabel(
    "Frequência"
)

plt.tight_layout()


# ============================================================
# SALVAMENTO
# ============================================================

caminho_saida = PASTA_SAIDA / nome_saida

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

print("\nGráfico salvo em:")
print(caminho_saida)