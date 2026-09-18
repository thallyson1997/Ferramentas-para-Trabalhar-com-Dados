from utilidades import ler_arquivo
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

PASTA_SAIDA.mkdir(
    parents=True,
    exist_ok=True
)


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

        print(
            "\nNenhum arquivo compatível encontrado "
            "em ENTRADA."
        )

        return None

    print("\n" + "=" * 60)
    print("ARQUIVOS DISPONÍVEIS")
    print("=" * 60)

    for i, arquivo in enumerate(
        arquivos,
        start=1
    ):

        print(
            f"{i} - {arquivo.name}"
        )

    while True:

        try:

            opcao = int(
                input(
                    "\nDigite o número do arquivo: "
                ).strip()
            )

            if 1 <= opcao <= len(arquivos):

                return arquivos[opcao - 1]

            print(
                "Erro: escolha um número da lista."
            )

        except ValueError:

            print(
                "Erro: digite apenas um número."
            )


def selecionar_coluna(df):

    colunas = list(df.columns)

    print("\n" + "=" * 60)
    print("COLUNAS DISPONÍVEIS")
    print("=" * 60)

    for i, coluna in enumerate(
        colunas,
        start=1
    ):

        dtype = str(
            df[coluna].dtype
        )

        print(
            f"{i} - {coluna} "
            f"[{dtype}]"
        )

    while True:

        try:

            opcao = int(
                input(
                    "\nDigite o número da coluna: "
                ).strip()
            )

            if 1 <= opcao <= len(colunas):

                return colunas[opcao - 1]

            print(
                "Erro: escolha um número da lista."
            )

        except ValueError:

            print(
                "Erro: digite apenas um número."
            )


def eh_numerica(serie):
    """
    Verifica se a série é numérica.

    Booleanos são tratados separadamente porque
    o pandas pode considerar bool como tipo numérico.
    """

    if pd.api.types.is_bool_dtype(serie):

        return False

    return pd.api.types.is_numeric_dtype(
        serie
    )


def eh_categorica(serie):

    return not eh_numerica(serie)


def formatar_valor(valor):

    if pd.isna(valor):

        return "NaN"

    if isinstance(
        valor,
        (float, np.floating)
    ):

        if valor.is_integer():

            return str(
                int(valor)
            )

        return str(valor)

    return str(valor)


def converter_numero(texto):

    texto = texto.strip()

    if not texto:

        raise ValueError(
            "Valor vazio."
        )

    texto = texto.replace(
        ",",
        "."
    )

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

            # ------------------------------------------------
            # VALOR ÚNICO
            # ------------------------------------------------

            if ":" not in entrada:

                valor = converter_numero(
                    entrada
                )

                return valor, valor

            # ------------------------------------------------
            # INTERVALO
            # ------------------------------------------------

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

            if (
                inicio is None
                and fim is None
            ):

                raise ValueError

            if (
                inicio is not None
                and fim is not None
                and inicio > fim
            ):

                print(
                    "Erro: o início não pode ser "
                    "maior que o fim."
                )

                continue

            return inicio, fim

        except ValueError:

            print(
                "Erro: informe um valor ou "
                "intervalo válido."
            )


# ============================================================
# LEITURA DE INTERVALO PROGRESSIVO
# ============================================================

def ler_intervalo_diretamente(
    valor_minimo
):

    while True:

        entrada = input(
            f"\nDigite um valor ou intervalo "
            f"começando em "
            f"{formatar_valor(valor_minimo)} "
            "(ex.: 18, 18:30, 18:): "
        ).strip()

        try:

            # ------------------------------------------------
            # VALOR ÚNICO
            # ------------------------------------------------

            if ":" not in entrada:

                valor = converter_numero(
                    entrada
                )

                if valor != valor_minimo:

                    print(
                        "Erro: o valor precisa ser "
                        f"{formatar_valor(valor_minimo)}."
                    )

                    continue

                return valor, valor

            # ------------------------------------------------
            # INTERVALO
            # ------------------------------------------------

            partes = entrada.split(":")

            if len(partes) != 2:

                raise ValueError

            inicio_texto = partes[0].strip()
            fim_texto = partes[1].strip()

            if not inicio_texto:

                print(
                    "Erro: o início do intervalo "
                    "é obrigatório."
                )

                continue

            inicio = converter_numero(
                inicio_texto
            )

            # O intervalo precisa começar
            # exatamente no menor valor pendente.

            if inicio != valor_minimo:

                print(
                    "Erro: o intervalo precisa "
                    "começar em "
                    f"{formatar_valor(valor_minimo)}."
                )

                continue

            fim = (
                converter_numero(fim_texto)
                if fim_texto
                else None
            )

            if (
                fim is not None
                and inicio > fim
            ):

                print(
                    "Erro: o início não pode ser "
                    "maior que o fim."
                )

                continue

            return inicio, fim

        except ValueError:

            print(
                "Erro: informe um valor ou "
                "intervalo válido."
            )


# ============================================================
# 1 - NUMÉRICO → NUMÉRICO
# ============================================================

def numerico_para_numerico(
    df,
    coluna
):

    serie_original = df[coluna]

    valores_unicos = (
        serie_original
        .dropna()
        .unique()
    )

    if len(valores_unicos) == 0:

        print(
            "\nA coluna não possui valores "
            "numéricos válidos."
        )

        return df.copy(), 0

    valores_ordenados = sorted(
        valores_unicos
    )

    menor_valor = valores_ordenados[0]
    maior_valor = valores_ordenados[-1]

    regras = []

    print("\n" + "=" * 60)
    print("NUMÉRICO → NUMÉRICO")
    print("=" * 60)

    print(
        f"Coluna: {coluna}"
    )

    print(
        "Menor valor: "
        f"{formatar_valor(menor_valor)}"
    )

    print(
        "Maior valor: "
        f"{formatar_valor(maior_valor)}"
    )

    # ========================================================
    # ESCOLHA DO MODO
    # ========================================================

    print("\n" + "-" * 60)
    print("COMO DESEJA DEFINIR AS REGRAS?")
    print("-" * 60)

    print(
        "1 - Aplicar uma regra em todo o intervalo"
    )

    print(
        "2 - Aplicar regras somente em parte "
        "do intervalo"
    )

    while True:

        opcao_modo = input(
            "\nDigite a opção: "
        ).strip()

        if opcao_modo in {"1", "2"}:

            break

        print(
            "Erro: escolha 1 ou 2."
        )

    # ========================================================
    # MODO 1 - TODO O INTERVALO
    # ========================================================

    if opcao_modo == "1":

        print("\n" + "-" * 60)
        print("REGRA PARA TODO O INTERVALO")
        print("-" * 60)

        print(
            f"\nTodos os valores entre "
            f"{formatar_valor(menor_valor)} "
            f"e "
            f"{formatar_valor(maior_valor)} "
            "serão transformados."
        )

        while True:

            novo_valor_texto = input(
                "\nDigite o novo valor: "
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
            (
                menor_valor,
                maior_valor,
                novo_valor
            )
        )

        print(
            "\nRegra definida: "
            f"{formatar_valor(menor_valor)} "
            "até "
            f"{formatar_valor(maior_valor)} "
            "→ "
            f"{formatar_valor(novo_valor)}"
        )

    # ========================================================
    # MODO 2 - PARTE DO INTERVALO
    # ========================================================

    else:

        print("\n" + "-" * 60)
        print("REGRAS PARCIAIS")
        print("-" * 60)

        print(
            "\nAs regras serão definidas "
            "progressivamente."
        )

        print(
            "O programa sempre informará "
            "o menor valor ainda não transformado."
        )

        print(
            "Cada novo intervalo deve começar "
            "exatamente nesse valor."
        )

        valores_pendentes = list(
            valores_ordenados
        )

        while valores_pendentes:

            menor_valor_pendente = (
                valores_pendentes[0]
            )

            print("\n" + "-" * 60)
            print("PRÓXIMO VALOR")
            print("-" * 60)

            print(
                "Menor valor ainda não "
                "transformado: "
                f"{formatar_valor(menor_valor_pendente)}"
            )

            inicio, fim = (
                ler_intervalo_diretamente(
                    menor_valor_pendente
                )
            )

            # ------------------------------------------------
            # DESCRIÇÃO DO INTERVALO
            # ------------------------------------------------

            if fim is None:

                descricao_intervalo = (
                    f"{formatar_valor(inicio)} "
                    "ou maior"
                )

            elif inicio == fim:

                descricao_intervalo = (
                    formatar_valor(inicio)
                )

            else:

                descricao_intervalo = (
                    f"{formatar_valor(inicio)} "
                    "até "
                    f"{formatar_valor(fim)}"
                )

            # ------------------------------------------------
            # NOVO VALOR
            # ------------------------------------------------

            while True:

                novo_valor_texto = input(
                    "\nDigite o novo valor para "
                    f"{descricao_intervalo}: "
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

            # ------------------------------------------------
            # REGISTRA REGRA
            # ------------------------------------------------

            regras.append(
                (
                    inicio,
                    fim,
                    novo_valor
                )
            )

            print(
                "\nRegra adicionada: "
                f"{descricao_intervalo} "
                "→ "
                f"{formatar_valor(novo_valor)}"
            )

            # ------------------------------------------------
            # ATUALIZA VALORES PENDENTES
            # ------------------------------------------------

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

            # ------------------------------------------------
            # MOSTRA PRÓXIMO VALOR
            # ------------------------------------------------

            if valores_pendentes:

                print(
                    "\nPróximo menor valor pendente: "
                    f"{formatar_valor(valores_pendentes[0])}"
                )

    # ========================================================
    # EXIBE REGRAS DEFINIDAS
    # ========================================================

    print("\n" + "-" * 60)
    print("REGRAS DEFINIDAS")
    print("-" * 60)

    for inicio, fim, novo_valor in regras:

        if fim is None:

            descricao = (
                f"{formatar_valor(inicio)} "
                "ou maior"
            )

        elif inicio == fim:

            descricao = (
                formatar_valor(inicio)
            )

        else:

            descricao = (
                f"{formatar_valor(inicio)} "
                "até "
                f"{formatar_valor(fim)}"
            )

        print(
            f"{descricao} "
            "→ "
            f"{formatar_valor(novo_valor)}"
        )

    # ========================================================
    # APLICA AS REGRAS
    # ========================================================

    df_resultado = df.copy()

    coluna_transformada = (
        serie_original
        .copy()
        .astype("object")
    )

    quantidade_afetada = 0

    for inicio, fim, novo_valor in regras:

        mascara = (
            serie_original.notna()
        )

        if inicio is not None:

            mascara &= (
                serie_original >= inicio
            )

        if fim is not None:

            mascara &= (
                serie_original <= fim
            )

        quantidade_afetada += int(
            mascara.sum()
        )

        coluna_transformada.loc[
            mascara
        ] = novo_valor

    df_resultado[coluna] = (
        coluna_transformada
    )

    return (
        df_resultado,
        quantidade_afetada
    )


# ============================================================
# 2 - NUMÉRICO → CATEGÓRICO
# ============================================================

def numerico_para_categorico(
    df,
    coluna
):

    serie_original = df[coluna]

    valores_unicos = (
        serie_original
        .dropna()
        .unique()
    )

    if len(valores_unicos) == 0:

        print(
            "\nA coluna não possui valores "
            "numéricos válidos."
        )

        return df.copy(), 0

    valores_pendentes = sorted(
        valores_unicos
    )

    print("\n" + "=" * 60)
    print("NUMÉRICO → CATEGÓRICO")
    print("=" * 60)

    print(
        f"Coluna: {coluna}"
    )

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
            "Menor valor ainda não transformado: "
            f"{formatar_valor(menor_valor)}"
        )

        inicio, fim = (
            ler_intervalo_diretamente(
                menor_valor
            )
        )

        if fim is None:

            descricao_intervalo = (
                f"{formatar_valor(inicio)} "
                "ou maior"
            )

        elif inicio == fim:

            descricao_intervalo = (
                formatar_valor(inicio)
            )

        else:

            descricao_intervalo = (
                f"{formatar_valor(inicio)} "
                "até "
                f"{formatar_valor(fim)}"
            )

        categoria = input(
            "\nDigite a categoria para "
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
            (
                inicio,
                fim,
                categoria
            )
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
            "\nRegra adicionada: "
            f"{descricao_intervalo} "
            "→ "
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
                f"{formatar_valor(inicio)} "
                "ou maior"
            )

        elif inicio == fim:

            descricao = (
                formatar_valor(inicio)
            )

        else:

            descricao = (
                f"{formatar_valor(inicio)} "
                "até "
                f"{formatar_valor(fim)}"
            )

        print(
            f"{descricao} → {categoria}"
        )

    # ========================================================
    # CORREÇÃO DE DTYPE
    # ========================================================

    coluna_transformada = (
        serie_original
        .copy()
        .astype("object")
    )

    coluna_transformada[:] = np.nan

    for inicio, fim, categoria in regras:

        mascara = (
            serie_original.notna()
        )

        if inicio is not None:

            mascara &= (
                serie_original >= inicio
            )

        if fim is not None:

            mascara &= (
                serie_original <= fim
            )

        coluna_transformada.loc[
            mascara
        ] = categoria

    df_resultado = df.copy()

    df_resultado[coluna] = (
        coluna_transformada
    )

    quantidade_transformada = int(
        serie_original.notna().sum()
    )

    return (
        df_resultado,
        quantidade_transformada
    )


# ============================================================
# 3 - CATEGÓRICO → NUMÉRICO
# ============================================================

def categorico_para_numerico(
    df,
    coluna
):

    serie = df[coluna]

    valores = (
        serie
        .dropna()
        .unique()
    )

    if len(valores) == 0:

        print(
            "\nA coluna não possui categorias "
            "válidas."
        )

        return df.copy(), 0

    print("\n" + "=" * 60)
    print("CATEGÓRICO → NUMÉRICO")
    print("=" * 60)

    print(
        f"Coluna: {coluna}"
    )

    print(
        f"Quantidade de categorias: "
        f"{len(valores)}"
    )

    print(
        "\nComo deseja realizar a transformação?"
    )

    print(
        "1 - Definir os números manualmente"
    )

    print(
        "2 - Gerar os números automaticamente"
    )

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao in {"1", "2"}:

            break

        print(
            "Erro: escolha 1 ou 2."
        )

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

            dicionario[categoria] = (
                numero
            )

        print("\n" + "-" * 60)
        print("MAPEAMENTO GERADO")
        print("-" * 60)

        for categoria, numero in (
            dicionario.items()
        ):

            print(
                f"{formatar_valor(categoria)} "
                "→ "
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

            print(
                "Erro: responda com s ou n."
            )

        if confirmacao != "s":

            print(
                "\nTransformação cancelada."
            )

            return df.copy(), 0

    # ========================================================
    # MODO MANUAL
    # ========================================================

    else:

        valores_pendentes = list(
            valores
        )

        dicionario = {}

        print(
            "\nTodas as categorias precisam "
            "receber um número."
        )

        while valores_pendentes:

            categoria_atual = (
                valores_pendentes[0]
            )

            print("\n" + "-" * 60)
            print("PRÓXIMA CATEGORIA")
            print("-" * 60)

            print(
                "Categoria: "
                f"{formatar_valor(categoria_atual)}"
            )

            while True:

                entrada = input(
                    "Digite o número que "
                    "substituirá essa categoria: "
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
                "\nRegra adicionada: "
                f"{formatar_valor(categoria_atual)} "
                "→ "
                f"{formatar_valor(numero)}"
            )

    # ========================================================
    # EXIBE MAPEAMENTO FINAL
    # ========================================================

    print("\n" + "-" * 60)
    print("MAPEAMENTO DEFINIDO")
    print("-" * 60)

    for categoria, numero in (
        dicionario.items()
    ):

        print(
            f"{formatar_valor(categoria)} "
            "→ "
            f"{formatar_valor(numero)}"
        )

    # ========================================================
    # APLICA MAPEAMENTO
    # ========================================================

    coluna_transformada = (
        serie.map(dicionario)
    )

    coluna_transformada = (
        pd.to_numeric(
            coluna_transformada,
            errors="coerce"
        )
    )

    df_resultado = df.copy()

    df_resultado[coluna] = (
        coluna_transformada
    )

    quantidade_total = int(
        serie.notna().sum()
    )

    return (
        df_resultado,
        quantidade_total
    )


# ============================================================
# 4 - CATEGÓRICO → CATEGÓRICO
# ============================================================

def categorico_para_categorico(
    df,
    coluna
):

    serie_original = df[coluna]

    valores_unicos = (
        serie_original
        .dropna()
        .unique()
    )

    if len(valores_unicos) == 0:

        print(
            "\nA coluna não possui categorias "
            "válidas."
        )

        return df.copy(), 0

    print("\n" + "=" * 60)
    print("CATEGÓRICO → CATEGÓRICO")
    print("=" * 60)

    print(
        f"Coluna: {coluna}"
    )

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

    # ========================================================
    # PRIMEIRA TRANSFORMAÇÃO
    # ========================================================

    while True:

        valor_atual = input(
            "\nDigite o valor que deseja "
            "substituir: "
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
        (
            valor_atual,
            novo_valor
        )
    )

    # ========================================================
    # DEMAIS TRANSFORMAÇÕES
    # ========================================================

    while True:

        continuar = input(
            "\nDeseja adicionar outra "
            "transformação? (s/n): "
        ).strip().lower()

        if continuar in {"s", "n"}:

            break

        print(
            "Erro: responda com s ou n."
        )

    while continuar == "s":

        while True:

            valor_atual = input(
                "\nDigite o valor que deseja "
                "substituir: "
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
            (
                valor_atual,
                novo_valor
            )
        )

        while True:

            continuar = input(
                "\nDeseja adicionar outra "
                "transformação? (s/n): "
            ).strip().lower()

            if continuar in {"s", "n"}:

                break

            print(
                "Erro: responda com s ou n."
            )

    # ========================================================
    # EXIBE TRANSFORMAÇÕES
    # ========================================================

    print("\n" + "-" * 60)
    print("TRANSFORMAÇÕES DEFINIDAS")
    print("-" * 60)

    for valor_atual, novo_valor in regras:

        print(
            f"{formatar_valor(valor_atual)}"
            " → "
            f"{formatar_valor(novo_valor)}"
        )

    # ========================================================
    # APLICA TRANSFORMAÇÕES
    # ========================================================

    df_resultado = df.copy()

    coluna_transformada = (
        serie_original
        .copy()
        .astype("object")
    )

    for valor_atual, novo_valor in regras:

        mascara = (
            serie_original == valor_atual
        )

        coluna_transformada.loc[
            mascara
        ] = novo_valor

    df_resultado[coluna] = (
        coluna_transformada
    )

    quantidade_afetada = 0

    for valor_atual, _ in regras:

        quantidade_afetada += int(
            (
                serie_original
                == valor_atual
            ).sum()
        )

    return (
        df_resultado,
        quantidade_afetada
    )


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("011 - TRANSFORMAR DADOS")
    print("=" * 60)

    # ========================================================
    # SELEÇÃO DO ARQUIVO
    # ========================================================

    caminho_arquivo = (
        selecionar_arquivo()
    )

    if caminho_arquivo is None:

        return

    print(
        f"\nArquivo selecionado: "
        f"{caminho_arquivo.name}"
    )

    # ========================================================
    # CARREGAMENTO
    # ========================================================

    try:

        df = ler_arquivo(
            caminho_arquivo
        )

    except Exception as erro:

        print(
            "\nErro ao carregar o arquivo:"
        )

        print(erro)

        return

    if df is None:

        return

    print(
        f"\nDataset carregado:"
        f"\nLinhas: {len(df)}"
        f"\nColunas: {len(df.columns)}"
    )

    # ========================================================
    # SELEÇÃO DA COLUNA
    # ========================================================

    coluna = selecionar_coluna(df)

    serie = df[coluna]

    # ========================================================
    # CLASSIFICAÇÃO DA COLUNA
    # ========================================================

    if eh_numerica(serie):

        tipo_coluna = "NUMÉRICA"

    else:

        tipo_coluna = (
            "CATEGÓRICA / TEXTUAL"
        )

    print("\n" + "-" * 60)

    print(
        f"Coluna selecionada: "
        f"{coluna}"
    )

    print(
        f"Tipo identificado: "
        f"{tipo_coluna}"
    )

    print(
        f"dtype original: "
        f"{serie.dtype}"
    )

    # ========================================================
    # ESCOLHA DA TRANSFORMAÇÃO
    # ========================================================

    print("\n" + "=" * 60)
    print("TIPO DE TRANSFORMAÇÃO")
    print("=" * 60)

    print(
        "1 - Numérico → Numérico"
    )

    print(
        "2 - Numérico → Categórico"
    )

    print(
        "3 - Categórico → Numérico"
    )

    print(
        "4 - Categórico → Categórico"
    )

    while True:

        opcao = input(
            "\nDigite a opção: "
        ).strip()

        if opcao in {
            "1",
            "2",
            "3",
            "4"
        }:

            break

        print(
            "Erro: escolha uma opção de 1 a 4."
        )

    # ========================================================
    # VALIDAÇÃO DO TIPO
    # ========================================================

    if (
        opcao in {"1", "2"}
        and not eh_numerica(serie)
    ):

        print(
            "\nErro: essa transformação exige "
            "uma coluna numérica."
        )

        return

    if (
        opcao in {"3", "4"}
        and eh_numerica(serie)
    ):

        print(
            "\nErro: essa transformação exige "
            "uma coluna categórica/textual."
        )

        return

    # ========================================================
    # TRANSFORMAÇÃO
    # ========================================================

    if opcao == "1":

        (
            df_resultado,
            quantidade
        ) = numerico_para_numerico(
            df,
            coluna
        )

    elif opcao == "2":

        (
            df_resultado,
            quantidade
        ) = numerico_para_categorico(
            df,
            coluna
        )

    elif opcao == "3":

        (
            df_resultado,
            quantidade
        ) = categorico_para_numerico(
            df,
            coluna
        )

    else:

        (
            df_resultado,
            quantidade
        ) = categorico_para_categorico(
            df,
            coluna
        )

    # ========================================================
    # SALVAMENTO
    # ========================================================

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

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("TRANSFORMAÇÃO CONCLUÍDA")
    print("=" * 60)

    print(
        f"Coluna transformada: "
        f"{coluna}"
    )

    print(
        f"Valores afetados: "
        f"{quantidade}"
    )

    print(
        f"dtype antes: "
        f"{serie.dtype}"
    )

    print(
        f"dtype depois: "
        f"{df_resultado[coluna].dtype}"
    )

    print(
        "\nArquivo salvo em:"
        f"\n{caminho_saida}"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    main()