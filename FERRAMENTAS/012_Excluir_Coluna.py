from pathlib import Path
from utilidades import ler_arquivo
import pandas as pd


# ============================================================
# CONFIGURAÇÃO DOS DIRETÓRIOS
# ============================================================

PASTA_FERRAMENTAS = Path(__file__).resolve().parent
PASTA_REPOSITORIO = PASTA_FERRAMENTAS.parent

PASTA_ENTRADA = PASTA_REPOSITORIO / "ENTRADA"
PASTA_SAIDA = PASTA_REPOSITORIO / "SAIDA"

PASTA_SAIDA.mkdir(exist_ok=True)


# ============================================================
# LISTAR ARQUIVOS
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


# ============================================================
# SELECIONAR ARQUIVO
# ============================================================

def selecionar_arquivo():

    arquivos = listar_arquivos()

    if not arquivos:

        print(
            "\nNenhum arquivo compatível "
            "encontrado em ENTRADA."
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

# ============================================================
# MOSTRAR COLUNAS
# ============================================================

def mostrar_colunas(df):

    print("\n" + "=" * 60)
    print("COLUNAS DISPONÍVEIS")
    print("=" * 60)

    if len(df.columns) == 0:

        print(
            "\nO dataset não possui nenhuma coluna."
        )

        return

    for i, coluna in enumerate(
        df.columns,
        start=1
    ):

        print(
            f"{i} - {coluna} "
            f"[{df[coluna].dtype}]"
        )


# ============================================================
# SELECIONAR COLUNA
# ============================================================

def selecionar_coluna(df):

    while True:

        try:

            opcao = int(
                input(
                    "\nDigite o número da coluna "
                    "que deseja excluir: "
                ).strip()
            )

            if 1 <= opcao <= len(df.columns):

                return df.columns[opcao - 1]

            print(
                "Erro: escolha um número da lista."
            )

        except ValueError:

            print(
                "Erro: digite apenas um número."
            )


# ============================================================
# FUNÇÃO PRINCIPAL
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("012 - EXCLUIR COLUNAS")
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

    df = ler_arquivo(
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
    # EXCLUSÃO DAS COLUNAS
    # --------------------------------------------------------

    while True:

        # Verifica se ainda existem colunas
        if len(df.columns) == 0:

            print(
                "\nTodas as colunas foram excluídas."
            )

            break

        mostrar_colunas(df)

        coluna = selecionar_coluna(df)

        print(
            f"\nColuna selecionada: {coluna}"
        )

        # ----------------------------------------------------
        # CONFIRMAÇÃO
        # ----------------------------------------------------

        while True:

            confirmacao = input(
                f"\nTem certeza que deseja excluir "
                f"a coluna '{coluna}'? (s/n): "
            ).strip().lower()

            if confirmacao in {"s", "n"}:
                break

            print(
                "Erro: responda com s ou n."
            )

        if confirmacao == "n":

            print(
                "\nExclusão cancelada."
            )

        else:

            df = df.drop(
                columns=[coluna]
            )

            print(
                f"\nColuna '{coluna}' excluída "
                "com sucesso."
            )

            print(
                f"Colunas restantes: "
                f"{len(df.columns)}"
            )

        # ----------------------------------------------------
        # VERIFICA SE DESEJA EXCLUIR OUTRA
        # ----------------------------------------------------

        if len(df.columns) == 0:
            break

        while True:

            continuar = input(
                "\nDeseja excluir outra coluna? "
                "(s/n): "
            ).strip().lower()

            if continuar in {"s", "n"}:
                break

            print(
                "Erro: responda com s ou n."
            )

        if continuar == "n":
            break

    # --------------------------------------------------------
    # SALVAMENTO
    # --------------------------------------------------------

    nome_saida = (
        f"012_Sem_Colunas_"
        f"{caminho_arquivo.stem}.csv"
    )

    caminho_saida = (
        PASTA_SAIDA / nome_saida
    )

    try:

        df.to_csv(
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
    print("PROCESSO CONCLUÍDO")
    print("=" * 60)

    print(
        f"Linhas finais: {len(df)}"
    )

    print(
        f"Colunas finais: {len(df.columns)}"
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