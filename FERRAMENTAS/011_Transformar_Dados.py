from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURAÇÃO DOS DIRETÓRIOS
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

    arquivos = sorted(
        [
            arquivo
            for arquivo in PASTA_ENTRADA.iterdir()
            if arquivo.is_file()
            and arquivo.suffix.lower() in extensoes
        ]
    )

    return arquivos


def selecionar_arquivo():
    arquivos = listar_arquivos()

    if not arquivos:
        print("\nNenhum arquivo compatível encontrado em ENTRADA.")
        return None

    print("\n" + "=" * 60)
    print("ARQUIVOS DISPONÍVEIS")
    print("=" * 60)

    for i, arquivo in enumerate(arquivos, start=1):
        print(f"{i} - {arquivo.name}")

    while True:
        try:
            opcao = int(
                input("\nDigite o número do arquivo: ").strip()
            )

            if 1 <= opcao <= len(arquivos):
                return arquivos[opcao - 1]

            print("Erro: escolha um número da lista.")

        except ValueError:
            print("Erro: digite apenas um número.")


def carregar_arquivo(caminho):
    extensao = caminho.suffix.lower()

    try:

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
            print("\nFormato não suportado.")
            return None

    except Exception as erro:
        print(
            f"\nErro ao carregar o arquivo:\n{erro}"
        )
        return None


def selecionar_coluna(df):
    colunas = list(df.columns)

    print("\n" + "=" * 60)
    print("COLUNAS DISPONÍVEIS")
    print("=" * 60)

    for i, coluna in enumerate(colunas, start=1):

        dtype = str(df[coluna].dtype)

        print(
            f"{i} - {coluna} "
            f"[{dtype}]"
        )

    while True:

        try:
            opcao = int(
                input("\nDigite o número da coluna: ").strip()
            )

            if 1 <= opcao <= len(colunas):
                return colunas[opcao - 1]

            print("Erro: escolha um número da lista.")

        except ValueError:
            print("Erro: digite apenas um número.")


def eh_numerica(serie):
    """
    Verifica se a série é numérica.

    Booleanos são tratados separadamente porque
    o pandas pode considerar bool como tipo numérico.
    """

    if pd.api.types.is_bool_dtype(serie):
        return False

    return pd.api.types.is_numeric_dtype(serie)


def eh_categorica(serie):
    """
    Considera como categórica qualquer série que
    não seja numérica.
    """

    return not eh_numerica(serie)


def formatar_valor(valor):

    if pd.isna(valor):
        return "NaN"

    if isinstance(valor, (float, np.floating)):

        if valor.is_integer():
            return str(int(valor))

        return str(valor)

    return str(valor)


def converter_numero(texto):

    texto = texto.strip()

    if not texto:
        raise ValueError("Valor vazio.")

    texto = texto.replace(",", ".")

    numero = float(texto)

    if numero.is_integer():
        return int(numero)

    return numero


# ============================================================
# LEITURA DE INTERVALO NUMÉRICO
# ============================================================

def ler_intervalo_numerico():

    while True:

        entrada = input(
            "\nDigite um valor ou intervalo "
            "(ex.: 3, 100:200, 100:, :200): "
        ).strip()

        try:

            if ":" not in entrada:

                valor = converter_numero(entrada)

                return valor, valor

            partes = entrada.split(":")

            if len(partes) != 2:
                raise ValueError

            inicio_texto = partes[0].strip()
            fim_texto = partes[1].strip()

            inicio = (
                converter_numero(inicio_texto)
                if inicio_texto
                else None
            )

            fim = (
                converter_numero(fim_texto)
                if fim_texto
                else None
            )

            if inicio is None and fim is None:
                return None, None

            if (
                inicio is not None
                and fim is not None
                and inicio > fim
            ):
                print(
                    "Erro: o início não pode ser maior "
                    "que o fim."
                )
                continue

            return inicio, fim

        except ValueError:
            print(
                "Erro: informe um valor ou intervalo válido."
            )


# ============================================================
# LEITURA DE INTERVALO PARA NUMÉRICO → CATEGÓRICO
# ============================================================

def ler_intervalo_diretamente(valor_minimo):

    while True:

        entrada = input(
            f"\nDigite um valor ou intervalo começando "
            f"em {formatar_valor(valor_minimo)} "
            "(ex.: 18, 18:30, 18:): "
        ).strip()

        try:

            if ":" not in entrada:

                valor = converter_numero(entrada)

                if valor != valor_minimo:

                    print(
                        f"Erro: o valor precisa ser "
                        f"{formatar_valor(valor_minimo)}."
                    )
                    continue

                return valor, valor

            partes = entrada.split(":")

            if len(partes) != 2:
                raise ValueError

            inicio_texto = partes[0].strip()
            fim_texto = partes[1].strip()

            if not inicio_texto:

                print(
                    "Erro: o início do intervalo é obrigatório."
                )
                continue

            inicio = converter_numero(inicio_texto)

            if inicio != valor_minimo:

                print(
                    f"Erro: o intervalo precisa começar "
                    f"em {formatar_valor(valor_minimo)}."
                )
                continue

            fim = (
                converter_numero(fim_texto)
                if fim_texto
                else None
            )

            if fim is not None and inicio > fim:

                print(
                    "Erro: o início não pode ser maior "
                    "que o fim."
                )
                continue

            return inicio, fim

        except ValueError:
            print(
                "Erro: informe um valor ou intervalo válido."
            )


# ============================================================
# 1 - NUMÉRICO → NUMÉRICO
# ============================================================

def numerico_para_numerico(df, coluna):

    serie_original = df[coluna]

    regras = []

    print("\n" + "=" * 60)
    print("NUMÉRICO → NUMÉRICO")
    print("=" * 60)

    print(f"Coluna: {coluna}")

    while True:

        inicio, fim = ler_intervalo_numerico()

        print(
            f"\nIntervalo selecionado: "
            f"{formatar_valor(inicio)}"
            f" até "
            f"{formatar_valor(fim)}"
        )

        while True:

            novo_valor_texto = input(
                "Digite o novo valor: "
            ).strip()

            try:
                novo_valor = converter_numero(
                    novo_valor_texto
                )
                break

            except ValueError:
                print(
                    "Erro: informe um número válido."
                )

        regras.append(
            (inicio, fim, novo_valor)
        )

        print(
            f"\nRegra adicionada: "
            f"{formatar_valor(inicio)}"
            f" até "
            f"{formatar_valor(fim)}"
            f" → "
            f"{formatar_valor(novo_valor)}"
        )

        while True:

            continuar = input(
                "\nDeseja adicionar outra regra? "
                "(s/n): "
            ).strip().lower()

            if continuar in {"s", "n"}:
                break

            print("Erro: responda com s ou n.")

        if continuar == "n":
            break

    print("\n" + "-" * 60)
    print("REGRAS DEFINIDAS")
    print("-" * 60)

    for inicio, fim, novo_valor in regras:

        print(
            f"{formatar_valor(inicio)}"
            f" até "
            f"{formatar_valor(fim)}"
            f" → "
            f"{formatar_valor(novo_valor)}"
        )

    df_resultado = df.copy()

    quantidade_afetada = 0

    for inicio, fim, novo_valor in regras:

        if inicio is None and fim is None:

            mascara = serie_original.notna()

        else:

            mascara = serie_original.notna()

            if inicio is not None:
                mascara &= serie_original >= inicio

            if fim is not None:
                mascara &= serie_original <= fim

        quantidade_afetada += int(mascara.sum())

        df_resultado.loc[
            mascara,
            coluna
        ] = novo_valor

    return df_resultado, quantidade_afetada


# ============================================================
# 2 - NUMÉRICO → CATEGÓRICO
# ============================================================

def numerico_para_categorico(df, coluna):

    serie_original = df[coluna]

    valores_unicos = (
        serie_original
        .dropna()
        .unique()
    )

    if len(valores_unicos) == 0:

        print(
            "\nA coluna não possui valores numéricos válidos."
        )

        return df.copy(), 0

    valores_pendentes = sorted(
        valores_unicos
    )

    print("\n" + "=" * 60)
    print("NUMÉRICO → CATEGÓRICO")
    print("=" * 60)

    print(f"Coluna: {coluna}")

    print(
        "\nTodos os valores numéricos precisam "
        "receber uma categoria."
    )

    regras = []

    while valores_pendentes:

        menor_valor = valores_pendentes[0]

        print("\n" + "-" * 60)
        print("PRÓXIMO VALOR")
        print("-" * 60)

        print(
            f"Menor valor ainda não transformado: "
            f"{formatar_valor(menor_valor)}"
        )

        inicio, fim = ler_intervalo_diretamente(
            menor_valor
        )

        if fim is None:
            descricao_intervalo = (
                f"{formatar_valor(inicio)} ou maior"
            )

        elif inicio == fim:
            descricao_intervalo = (
                formatar_valor(inicio)
            )

        else:
            descricao_intervalo = (
                f"{formatar_valor(inicio)}"
                f" até "
                f"{formatar_valor(fim)}"
            )

        categoria = input(
            f"\nDigite a categoria para "
            f"{descricao_intervalo}: "
        ).strip()

        while not categoria:

            print(
                "Erro: a categoria não pode ser vazia."
            )

            categoria = input(
                "Digite a categoria: "
            ).strip()

        regras.append(
            (inicio, fim, categoria)
        )

        if fim is None:

            valores_pendentes = [
                valor
                for valor in valores_pendentes
                if valor < inicio
            ]

        else:

            valores_pendentes = [
                valor
                for valor in valores_pendentes
                if not (
                    valor >= inicio
                    and valor <= fim
                )
            ]

        print(
            f"\nRegra adicionada: "
            f"{descricao_intervalo}"
            f" → "
            f"{categoria}"
        )

        if valores_pendentes:

            print(
                "\nPróximo menor valor pendente: "
                f"{formatar_valor(valores_pendentes[0])}"
            )

    print("\n" + "-" * 60)
    print("MAPEAMENTO DEFINIDO")
    print("-" * 60)

    for inicio, fim, categoria in regras:

        if fim is None:
            descricao = (
                f"{formatar_valor(inicio)} ou maior"
            )

        elif inicio == fim:
            descricao = formatar_valor(inicio)

        else:
            descricao = (
                f"{formatar_valor(inicio)}"
                f" até "
                f"{formatar_valor(fim)}"
            )

        print(
            f"{descricao} → {categoria}"
        )

    # --------------------------------------------------------
    # CORREÇÃO DE DTYPE
    # --------------------------------------------------------
    # A coluna original pode possuir dtype string.
    # Portanto, não fazemos atribuições numéricas ou de
    # categorias diretamente nela.
    #
    # Construímos uma nova série com os valores transformados
    # e só depois substituímos a coluna.
    # --------------------------------------------------------

    coluna_transformada = serie_original.copy()

    coluna_transformada = coluna_transformada.astype(
        "object"
    )

    coluna_transformada[:] = np.nan

    for inicio, fim, categoria in regras:

        mascara = serie_original.notna()

        if inicio is not None:
            mascara &= serie_original >= inicio

        if fim is not None:
            mascara &= serie_original <= fim

        coluna_transformada.loc[
            mascara
        ] = categoria

    # --------------------------------------------------------
    # GARANTE QUE VALORES NÃO TRANSFORMADOS NÃO SEJAM
    # PERDIDOS POR ACIDENTE
    # --------------------------------------------------------

    df_resultado = df.copy()

    df_resultado[coluna] = coluna_transformada

    quantidade_transformada = int(
        serie_original.notna().sum()
    )

    return df_resultado, quantidade_transformada


# ============================================================
# 3 - CATEGÓRICO → NUMÉRICO
# ============================================================

def categorico_para_numerico(df, coluna):

    serie = df[coluna]

    valores = (
        serie
        .dropna()
        .unique()
    )

    if len(valores) == 0:

        print(
            "\nA coluna não possui categorias válidas."
        )

        return df.copy(), 0

    print("\n" + "=" * 60)
    print("CATEGÓRICO → NUMÉRICO")
    print("=" * 60)

    print(f"Coluna: {coluna}")

    print(
        f"Quantidade de categorias: "
        f"{len(valores)}"
    )

    print("\nComo deseja realizar a transformação?")
    print("1 - Definir os números manualmente")
    print("2 - Gerar os números automaticamente")

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao in {"1", "2"}:
            break

        print("Erro: escolha 1 ou 2.")

    # ========================================================
    # MODO AUTOMÁTICO
    # ========================================================

    if opcao == "2":

        valores_ordenados = sorted(
            valores,
            key=lambda x: str(x).casefold()
        )

        dicionario = {}

        for numero, categoria in enumerate(
            valores_ordenados,
            start=1
        ):

            dicionario[categoria] = numero

        print("\n" + "-" * 60)
        print("MAPEAMENTO GERADO")
        print("-" * 60)

        for categoria, numero in dicionario.items():

            print(
                f"{formatar_valor(categoria)} "
                f"→ "
                f"{numero}"
            )

        print(
            "\nOs números foram atribuídos "
            "em ordem alfabética."
        )

        print(
            "A numeração representa apenas "
            "códigos das categorias."
        )

        print(
            "A ordem numérica NÃO representa "
            "uma relação de maior ou menor."
        )

        while True:

            confirmacao = input(
                "\nDeseja aplicar esse mapeamento? "
                "(s/n): "
            ).strip().lower()

            if confirmacao in {"s", "n"}:
                break

            print("Erro: responda com s ou n.")

        if confirmacao != "s":

            print(
                "\nTransformação cancelada."
            )

            return df.copy(), 0

    # ========================================================
    # MODO MANUAL
    # ========================================================

    else:

        valores_pendentes = list(valores)

        dicionario = {}

        print(
            "\nTodas as categorias precisam "
            "receber um número."
        )

        while valores_pendentes:

            categoria_atual = valores_pendentes[0]

            print("\n" + "-" * 60)
            print("PRÓXIMA CATEGORIA")
            print("-" * 60)

            print(
                f"Categoria: "
                f"{formatar_valor(categoria_atual)}"
            )

            while True:

                entrada = input(
                    "Digite o número que substituirá "
                    "essa categoria: "
                ).strip()

                try:

                    numero = converter_numero(
                        entrada
                    )

                    break

                except ValueError:

                    print(
                        "Erro: informe um número válido."
                    )

            dicionario[
                categoria_atual
            ] = numero

            valores_pendentes.pop(0)

            print(
                f"\nRegra adicionada: "
                f"{formatar_valor(categoria_atual)}"
                f" → "
                f"{formatar_valor(numero)}"
            )

    # ========================================================
    # EXIBE MAPEAMENTO FINAL
    # ========================================================

    print("\n" + "-" * 60)
    print("MAPEAMENTO DEFINIDO")
    print("-" * 60)

    for categoria, numero in dicionario.items():

        print(
            f"{formatar_valor(categoria)} "
            f"→ "
            f"{formatar_valor(numero)}"
        )

    # ========================================================
    # CORREÇÃO DE DTYPE
    # ========================================================
    # Não fazemos:
    #
    # df_resultado.loc[mascara, coluna] = numero
    #
    # pois a coluna original pode ser StringDtype.
    #
    # Em vez disso:
    #
    # 1. Aplicamos o dicionário com map()
    # 2. Convertemos o resultado para numérico
    # 3. Substituímos a coluna inteira
    #
    # Isso evita o TypeError causado pela tentativa de
    # colocar float/int diretamente em uma coluna string.
    # ========================================================

    coluna_transformada = (
        serie
        .map(dicionario)
    )

    coluna_transformada = pd.to_numeric(
        coluna_transformada,
        errors="coerce"
    )

    df_resultado = df.copy()

    df_resultado[coluna] = coluna_transformada

    quantidade_total = int(
        serie.notna().sum()
    )

    return df_resultado, quantidade_total


# ============================================================
# 4 - CATEGÓRICO → CATEGÓRICO
# ============================================================

def categorico_para_categorico(df, coluna):

    serie_original = df[coluna]

    valores_unicos = (
        serie_original
        .dropna()
        .unique()
    )

    if len(valores_unicos) == 0:

        print(
            "\nA coluna não possui categorias válidas."
        )

        return df.copy(), 0

    print("\n" + "=" * 60)
    print("CATEGÓRICO → CATEGÓRICO")
    print("=" * 60)

    print(f"Coluna: {coluna}")

    print(
        f"Quantidade de categorias: "
        f"{len(valores_unicos)}"
    )

    print("\nCategorias existentes:")

    for i, valor in enumerate(
        valores_unicos,
        start=1
    ):

        print(
            f"{i} - {formatar_valor(valor)}"
        )

    regras = []

    while True:

        valor_atual = input(
            "\nDigite o valor que deseja substituir: "
        ).strip()

        encontrados = [
            valor
            for valor in valores_unicos
            if str(valor) == valor_atual
        ]

        if encontrados:

            valor_atual = encontrados[0]
            break

        print(
            "Erro: esse valor não foi encontrado."
        )

    while True:

        novo_valor = input(
            "Digite o novo valor: "
        ).strip()

        if novo_valor:
            break

        print(
            "Erro: o novo valor não pode ser vazio."
        )

    regras.append(
        (valor_atual, novo_valor)
    )

    while True:

        continuar = input(
            "\nDeseja adicionar outra transformação? "
            "(s/n): "
        ).strip().lower()

        if continuar in {"s", "n"}:
            break

        print("Erro: responda com s ou n.")

    while continuar == "s":

        while True:

            valor_atual = input(
                "\nDigite o valor que deseja substituir: "
            ).strip()

            encontrados = [
                valor
                for valor in valores_unicos
                if str(valor) == valor_atual
            ]

            if encontrados:

                valor_atual = encontrados[0]
                break

            print(
                "Erro: esse valor não foi encontrado."
            )

        while True:

            novo_valor = input(
                "Digite o novo valor: "
            ).strip()

            if novo_valor:
                break

            print(
                "Erro: o novo valor não pode ser vazio."
            )

        regras.append(
            (valor_atual, novo_valor)
        )

        while True:

            continuar = input(
                "\nDeseja adicionar outra transformação? "
                "(s/n): "
            ).strip().lower()

            if continuar in {"s", "n"}:
                break

            print("Erro: responda com s ou n.")

    print("\n" + "-" * 60)
    print("TRANSFORMAÇÕES DEFINIDAS")
    print("-" * 60)

    for valor_atual, novo_valor in regras:

        print(
            f"{formatar_valor(valor_atual)}"
            f" → "
            f"{formatar_valor(novo_valor)}"
        )

    df_resultado = df.copy()

    coluna_transformada = serie_original.copy()

    coluna_transformada = coluna_transformada.astype(
        "object"
    )

    for valor_atual, novo_valor in regras:

        mascara = (
            serie_original == valor_atual
        )

        coluna_transformada.loc[
            mascara
        ] = novo_valor

    df_resultado[coluna] = coluna_transformada

    quantidade_afetada = 0

    for valor_atual, _ in regras:

        quantidade_afetada += int(
            (
                serie_original == valor_atual
            ).sum()
        )

    return df_resultado, quantidade_afetada


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("011 - TRANSFORMAR DADOS")
    print("=" * 60)

    # --------------------------------------------------------
    # SELEÇÃO DO ARQUIVO
    # --------------------------------------------------------

    caminho_arquivo = selecionar_arquivo()

    if caminho_arquivo is None:
        return

    print(
        f"\nArquivo selecionado: "
        f"{caminho_arquivo.name}"
    )

    # --------------------------------------------------------
    # CARREGAMENTO
    # --------------------------------------------------------

    df = carregar_arquivo(
        caminho_arquivo
    )

    if df is None:
        return

    print(
        f"\nDataset carregado:"
        f"\nLinhas: {len(df)}"
        f"\nColunas: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # SELEÇÃO DA COLUNA
    # --------------------------------------------------------

    coluna = selecionar_coluna(df)

    serie = df[coluna]

    # --------------------------------------------------------
    # CLASSIFICAÇÃO
    # --------------------------------------------------------

    if eh_numerica(serie):

        tipo_coluna = "NUMÉRICA"

    else:

        tipo_coluna = "CATEGÓRICA / TEXTUAL"

    print("\n" + "-" * 60)

    print(
        f"Coluna selecionada: {coluna}"
    )

    print(
        f"Tipo identificado: {tipo_coluna}"
    )

    print(
        f"dtype original: {serie.dtype}"
    )

    # --------------------------------------------------------
    # ESCOLHA DA TRANSFORMAÇÃO
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TIPO DE TRANSFORMAÇÃO")
    print("=" * 60)

    print("1 - Numérico → Numérico")
    print("2 - Numérico → Categórico")
    print("3 - Categórico → Numérico")
    print("4 - Categórico → Categórico")

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao in {"1", "2", "3", "4"}:
            break

        print("Erro: escolha uma opção de 1 a 4.")

    # --------------------------------------------------------
    # VALIDAÇÃO DO TIPO DA COLUNA
    # --------------------------------------------------------

    if opcao in {"1", "2"} and not eh_numerica(serie):

        print(
            "\nErro: essa transformação exige "
            "uma coluna numérica."
        )

        return

    if opcao in {"3", "4"} and eh_numerica(serie):

        print(
            "\nErro: essa transformação exige "
            "uma coluna categórica/textual."
        )

        return

    # --------------------------------------------------------
    # TRANSFORMAÇÃO
    # --------------------------------------------------------

    if opcao == "1":

        df_resultado, quantidade = (
            numerico_para_numerico(
                df,
                coluna
            )
        )

    elif opcao == "2":

        df_resultado, quantidade = (
            numerico_para_categorico(
                df,
                coluna
            )
        )

    elif opcao == "3":

        df_resultado, quantidade = (
            categorico_para_numerico(
                df,
                coluna
            )
        )

    else:

        df_resultado, quantidade = (
            categorico_para_categorico(
                df,
                coluna
            )
        )

    # --------------------------------------------------------
    # SALVAMENTO
    # --------------------------------------------------------

    nome_saida = (
        f"011_Transformado_"
        f"{caminho_arquivo.stem}.csv"
    )

    caminho_saida = (
        PASTA_SAIDA / nome_saida
    )

    try:

        df_resultado.to_csv(
            caminho_saida,
            index=False,
            encoding="utf-8-sig"
        )

    except Exception as erro:

        print(
            "\nErro ao salvar o arquivo:"
        )

        print(erro)

        return

    # --------------------------------------------------------
    # RESUMO FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRANSFORMAÇÃO CONCLUÍDA")
    print("=" * 60)

    print(
        f"Coluna transformada: {coluna}"
    )

    print(
        f"Valores afetados: {quantidade}"
    )

    print(
        f"dtype antes: {serie.dtype}"
    )

    print(
        f"dtype depois: "
        f"{df_resultado[coluna].dtype}"
    )

    print(
        f"\nArquivo salvo em:"
        f"\n{caminho_saida}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    main()