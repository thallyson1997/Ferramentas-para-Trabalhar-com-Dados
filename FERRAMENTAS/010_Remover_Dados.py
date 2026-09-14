# ============================================================
# REMOVER.py
# Remove linhas de um dataset com base em valores de uma coluna
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAÇÃO DAS PASTAS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def listar_arquivos():
    """Lista os arquivos compatíveis encontrados na pasta ENTRADA."""

    extensoes_suportadas = {
        ".csv",
        ".xlsx",
        ".xls",
        ".json",
        ".parquet"
    }

    arquivos = [
        arquivo
        for arquivo in PASTA_ENTRADA.iterdir()
        if arquivo.is_file() and arquivo.suffix.lower() in extensoes_suportadas
    ]

    return sorted(arquivos)


def carregar_arquivo(caminho):
    """Carrega o arquivo utilizando o pandas."""

    extensao = caminho.suffix.lower()

    if extensao == ".csv":
        return pd.read_csv(caminho)

    elif extensao == ".xlsx":
        return pd.read_excel(caminho)

    elif extensao == ".xls":
        return pd.read_excel(caminho)

    elif extensao == ".json":
        return pd.read_json(caminho)

    elif extensao == ".parquet":
        return pd.read_parquet(caminho)

    else:
        raise ValueError("Formato de arquivo não suportado.")


def eh_coluna_numerica(serie):
    """
    Verifica se a coluna é numérica.

    Booleanos não são considerados numéricos neste programa.
    """

    if pd.api.types.is_bool_dtype(serie):
        return False

    return pd.api.types.is_numeric_dtype(serie)


def exibir_colunas(df):
    """Exibe as colunas disponíveis para seleção."""

    print("\n" + "=" * 60)
    print("COLUNAS DISPONÍVEIS")
    print("=" * 60)

    for i, coluna in enumerate(df.columns, start=1):
        tipo = str(df[coluna].dtype)
        valores_unicos = df[coluna].nunique(dropna=True)

        print(
            f"{i:>3} - {coluna} "
            f"(tipo: {tipo}, únicos: {valores_unicos})"
        )


def selecionar_coluna(df):
    """Solicita ao usuário a coluna que será utilizada."""

    while True:
        exibir_colunas(df)

        entrada = input(
            "\nDigite o número da coluna que deseja utilizar: "
        ).strip()

        try:
            numero = int(entrada)

            if 1 <= numero <= len(df.columns):
                return df.columns[numero - 1]

            print("\nErro: número de coluna inválido.")

        except ValueError:
            print("\nErro: digite apenas o número da coluna.")


def converter_numero(valor):
    """
    Converte uma entrada do usuário para número.

    Aceita ponto ou vírgula como separador decimal.
    """

    valor = valor.strip().replace(",", ".")

    try:
        return float(valor)

    except ValueError:
        raise ValueError(f"'{valor}' não é um número válido.")


def formatar_valor(valor):
    """Formata valores para exibição mais amigável."""

    if isinstance(valor, float):
        if valor.is_integer():
            return str(int(valor))

    return str(valor)


# ============================================================
# TRATAMENTO DE COLUNA NUMÉRICA
# ============================================================

def definir_filtro_numerico(serie):
    """
    Solicita ao usuário o valor ou intervalo que deverá ser removido.

    Exemplos:
        3          -> remove valores iguais a 3
        100:200    -> remove valores entre 100 e 200
        100:       -> remove valores >= 100
        :100       -> remove valores <= 100
        :          -> remove todos os valores numéricos
    """

    minimo = serie.min()
    maximo = serie.max()

    print("\n" + "=" * 60)
    print("COLUNA NUMÉRICA")
    print("=" * 60)

    print(f"Valor mínimo: {formatar_valor(minimo)}")
    print(f"Valor máximo: {formatar_valor(maximo)}")

    print("\nComo deseja definir o valor a remover?")
    print("  Valor único  →  3")
    print("  Intervalo    →  100:200")
    print("  A partir de  →  100:")
    print("  Até          →  :100")
    print("  Todos        →  :")

    while True:
        entrada = input("\nDigite o valor ou intervalo: ").strip()

        if not entrada:
            print("Erro: informe um valor ou intervalo.")
            continue

        # ----------------------------------------------------
        # VALOR ÚNICO
        # ----------------------------------------------------

        if ":" not in entrada:
            try:
                valor = converter_numero(entrada)

                mascara = serie == valor

                descricao = f"valores iguais a {formatar_valor(valor)}"

                return mascara, descricao

            except ValueError as erro:
                print(f"Erro: {erro}")

                continue

        # ----------------------------------------------------
        # INTERVALO
        # ----------------------------------------------------

        partes = entrada.split(":")

        if len(partes) != 2:
            print(
                "Erro: utilize o formato "
                "mínimo:máximo."
            )
            continue

        limite_inferior = partes[0].strip()
        limite_superior = partes[1].strip()

        try:
            # ------------------------------------------------
            # :
            # ------------------------------------------------

            if not limite_inferior and not limite_superior:

                mascara = serie.notna()

                descricao = "todos os valores numéricos"

                print(
                    "\nATENÇÃO: o intervalo ':' seleciona "
                    "todos os valores não vazios da coluna."
                )

                return mascara, descricao

            # ------------------------------------------------
            # :100
            # ------------------------------------------------

            if not limite_inferior:

                limite = converter_numero(limite_superior)

                mascara = serie <= limite

                descricao = (
                    f"valores menores ou iguais a "
                    f"{formatar_valor(limite)}"
                )

                return mascara, descricao

            # ------------------------------------------------
            # 100:
            # ------------------------------------------------

            if not limite_superior:

                limite = converter_numero(limite_inferior)

                mascara = serie >= limite

                descricao = (
                    f"valores maiores ou iguais a "
                    f"{formatar_valor(limite)}"
                )

                return mascara, descricao

            # ------------------------------------------------
            # 100:200
            # ------------------------------------------------

            limite_inferior_num = converter_numero(limite_inferior)
            limite_superior_num = converter_numero(limite_superior)

            if limite_inferior_num > limite_superior_num:
                print(
                    "Erro: o limite inferior não pode "
                    "ser maior que o limite superior."
                )
                continue

            mascara = (
                (serie >= limite_inferior_num)
                & (serie <= limite_superior_num)
            )

            descricao = (
                f"valores entre "
                f"{formatar_valor(limite_inferior_num)} e "
                f"{formatar_valor(limite_superior_num)}"
            )

            return mascara, descricao

        except ValueError as erro:
            print(f"Erro: {erro}")


# ============================================================
# TRATAMENTO DE COLUNA CATEGÓRICA / NÃO NUMÉRICA
# ============================================================

def definir_filtro_categorico(serie):
    """
    Solicita ao usuário o valor categórico que deverá ser removido.

    Se houver até 10 valores únicos, eles são exibidos.
    Se houver mais de 10, nenhum valor é listado.
    """

    valores_unicos = serie.dropna().unique()
    quantidade_unicos = len(valores_unicos)

    print("\n" + "=" * 60)
    print("COLUNA CATEGÓRICA")
    print("=" * 60)

    print(f"Quantidade de valores únicos: {quantidade_unicos}")

    # --------------------------------------------------------
    # ATÉ 10 VALORES ÚNICOS
    # --------------------------------------------------------

    if quantidade_unicos <= 10:

        print("\nValores únicos:")

        for i, valor in enumerate(valores_unicos, start=1):
            print(f"  {i:>2} - {formatar_valor(valor)}")

    # --------------------------------------------------------
    # MAIS DE 10 VALORES ÚNICOS
    # --------------------------------------------------------

    else:

        print(
            "\nA coluna possui mais de 10 valores únicos."
        )

        print(
            "Os valores não serão listados para evitar "
            "poluir a tela."
        )

    # --------------------------------------------------------
    # ENTRADA DO VALOR
    # --------------------------------------------------------

    while True:

        valor_excluir = input(
            "\nDigite exatamente o valor que deseja remover: "
        )

        if not valor_excluir:
            print("Erro: informe um valor.")
            continue

        # Remove espaços apenas das extremidades.
        valor_excluir = valor_excluir.strip()

        # ----------------------------------------------------
        # COMPARAÇÃO COMO TEXTO
        # ----------------------------------------------------

        mascara = (
            serie.notna()
            & serie.astype(str).str.strip().eq(valor_excluir)
        )

        descricao = f"valor '{valor_excluir}'"

        return mascara, descricao


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("REMOVER LINHAS POR VALOR")
    print("=" * 60)

    # ========================================================
    # SELEÇÃO DO ARQUIVO
    # ========================================================

    arquivos = listar_arquivos()

    if not arquivos:
        print(
            "\nNenhum arquivo compatível foi encontrado "
            "na pasta ENTRADA."
        )
        return

    print("\nArquivos disponíveis:")

    for i, arquivo in enumerate(arquivos, start=1):
        print(f"  {i:>2} - {arquivo.name}")

    while True:

        entrada = input(
            "\nDigite o número do arquivo que deseja utilizar: "
        ).strip()

        try:
            numero = int(entrada)

            if 1 <= numero <= len(arquivos):
                arquivo_selecionado = arquivos[numero - 1]
                break

            print("\nErro: número de arquivo inválido.")

        except ValueError:
            print("\nErro: digite apenas o número do arquivo.")

    # ========================================================
    # CARREGAMENTO
    # ========================================================

    try:

        df = carregar_arquivo(arquivo_selecionado)

    except Exception as erro:

        print(
            f"\nErro ao carregar o arquivo: {erro}"
        )
        return

    print("\nArquivo carregado com sucesso.")
    print(f"Arquivo: {arquivo_selecionado.name}")
    print(f"Linhas: {len(df)}")
    print(f"Colunas: {len(df.columns)}")

    # ========================================================
    # SELEÇÃO DA COLUNA
    # ========================================================

    coluna = selecionar_coluna(df)

    serie = df[coluna]

    print("\n" + "=" * 60)
    print("COLUNA SELECIONADA")
    print("=" * 60)

    print(f"Coluna: {coluna}")
    print(f"Tipo: {serie.dtype}")
    print(
        f"Valores únicos: "
        f"{serie.nunique(dropna=True)}"
    )

    # ========================================================
    # DEFINIÇÃO DO FILTRO
    # ========================================================

    if eh_coluna_numerica(serie):

        mascara_remover, descricao = definir_filtro_numerico(
            serie
        )

    else:

        mascara_remover, descricao = definir_filtro_categorico(
            serie
        )

    # ========================================================
    # ANÁLISE DO QUE SERÁ REMOVIDO
    # ========================================================

    quantidade_remover = int(mascara_remover.sum())
    quantidade_original = len(df)
    quantidade_restante = quantidade_original - quantidade_remover

    print("\n" + "=" * 60)
    print("RESUMO DA REMOÇÃO")
    print("=" * 60)

    print(f"Arquivo: {arquivo_selecionado.name}")
    print(f"Coluna: {coluna}")
    print(f"Critério: {descricao}")
    print()
    print(f"Linhas originais: {quantidade_original}")
    print(f"Linhas a remover: {quantidade_remover}")
    print(f"Linhas restantes: {quantidade_restante}")

    # ========================================================
    # CONFIRMAÇÃO
    # ========================================================

    if quantidade_remover == 0:

        print(
            "\nNenhuma linha corresponde ao critério informado."
        )

        salvar = input(
            "Deseja salvar o arquivo mesmo assim? (s/n): "
        ).strip().lower()

        if salvar != "s":
            print("\nOperação cancelada.")
            return

    else:

        print(
            "\nAs linhas correspondentes ao critério "
            "serão removidas."
        )

        confirmacao = input(
            "Deseja confirmar a remoção? (s/n): "
        ).strip().lower()

        if confirmacao != "s":

            print("\nOperação cancelada.")
            return

    # ========================================================
    # REMOÇÃO
    # ========================================================

    df_resultado = df.loc[~mascara_remover].copy()

    # ========================================================
    # SALVAMENTO
    # ========================================================

    nome_saida = (
        f"010_Removido_{arquivo_selecionado.stem}.csv"
    )

    caminho_saida = PASTA_SAIDA / nome_saida

    try:

        df_resultado.to_csv(
            caminho_saida,
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as erro:

        print(
            f"\nErro ao salvar o arquivo: {erro}"
        )
        return

    # ========================================================
    # RESULTADO FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("OPERAÇÃO CONCLUÍDA")
    print("=" * 60)

    print(f"Linhas removidas: {quantidade_remover}")
    print(f"Linhas restantes: {len(df_resultado)}")
    print(f"\nArquivo salvo em:")
    print(caminho_saida)

    print(
        "\nO arquivo original na pasta ENTRADA "
        "não foi alterado."
    )


# ============================================================
# INÍCIO DO PROGRAMA
# ============================================================

if __name__ == "__main__":
    main()