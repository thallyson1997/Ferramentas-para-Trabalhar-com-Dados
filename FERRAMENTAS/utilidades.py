from pathlib import Path
import csv
import pandas as pd


# ============================================================
# IDENTIFICAÇÃO DO SEPARADOR CSV
# ============================================================

def identificar_separador(caminho):
    """
    Tenta identificar automaticamente o separador de um
    arquivo CSV.

    Separadores analisados:
    - vírgula (,)
    - ponto e vírgula (;)
    - tabulação (\t)
    - barra vertical (|)
    """

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1"
    ]

    for encoding in encodings:

        try:

            with open(
                caminho,
                "r",
                encoding=encoding,
                newline=""
            ) as arquivo:

                amostra = arquivo.read(8192)

            dialeto = csv.Sniffer().sniff(
                amostra,
                delimiters=",;\t|"
            )

            return dialeto.delimiter, encoding

        except (UnicodeDecodeError, csv.Error):
            continue

    return None, None


# ============================================================
# ESCOLHA MANUAL DO SEPARADOR
# ============================================================

def escolher_separador():

    print()
    print("=" * 70)
    print("ESCOLHA DO SEPARADOR")
    print("=" * 70)

    print("1 - Vírgula (,)")
    print("2 - Ponto e vírgula (;)")
    print("3 - Tabulação (\\t)")
    print("4 - Barra vertical (|)")
    print("5 - Outro")

    while True:

        escolha = input(
            "\nDigite o número do separador: "
        ).strip()

        if escolha == "1":
            return ","

        elif escolha == "2":
            return ";"

        elif escolha == "3":
            return "\t"

        elif escolha == "4":
            return "|"

        elif escolha == "5":

            separador = input(
                "Digite o separador utilizado no arquivo: "
            )

            if separador:
                return separador

            print(
                "O separador não pode ser vazio."
            )

        else:

            print(
                "Opção inválida. Escolha uma das opções listadas."
            )


# ============================================================
# LEITURA DE ARQUIVO
# ============================================================

def ler_arquivo(caminho, mostrar_detalhes=True):
    """
    Lê arquivos CSV, XLSX, XLS, JSON e Parquet.

    Para CSV:
    - tenta identificar automaticamente o separador;
    - tenta identificar a codificação;
    - solicita confirmação ao usuário;
    - permite escolha manual caso necessário.

    Retorna:
        DataFrame
    """

    caminho = Path(caminho)
    extensao = caminho.suffix.lower()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if extensao == ".csv":

        separador, encoding = identificar_separador(
            caminho
        )

        if separador is None:

            print()
            print(
                "Não foi possível identificar "
                "automaticamente o separador."
            )

            separador = escolher_separador()

            encoding = "utf-8"

        else:

            nomes_separadores = {
                ",": "vírgula (,)",
                ";": "ponto e vírgula (;)",
                "\t": "tabulação (\\t)",
                "|": "barra vertical (|)"
            }

            nome_separador = nomes_separadores.get(
                separador,
                repr(separador)
            )

            if mostrar_detalhes:

                print()
                print("=" * 70)
                print("CSV DETECTADO")
                print("=" * 70)

                print(
                    f"Separador identificado: {nome_separador}"
                )

                print(
                    f"Codificação identificada: {encoding}"
                )

                while True:

                    resposta = input(
                        "\nDeseja utilizar este separador? [S/n]: "
                    ).strip().lower()

                    if resposta in ["", "s", "sim"]:
                        break

                    elif resposta in ["n", "nao", "não"]:

                        separador = escolher_separador()
                        break

                    else:

                        print(
                            "Digite S para sim ou N para não."
                        )

        try:

            return pd.read_csv(
                caminho,
                sep=separador,
                encoding=encoding
            )

        except Exception as erro:

            raise ValueError(
                f"Não foi possível ler o CSV: {erro}"
            )

    # --------------------------------------------------------
    # EXCEL
    # --------------------------------------------------------

    elif extensao in [".xlsx", ".xls"]:

        try:

            return pd.read_excel(caminho)

        except Exception as erro:

            raise ValueError(
                f"Não foi possível ler o arquivo Excel: {erro}"
            )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    elif extensao == ".json":

        try:

            return pd.read_json(caminho)

        except Exception as erro:

            raise ValueError(
                f"Não foi possível ler o arquivo JSON: {erro}"
            )

    # --------------------------------------------------------
    # PARQUET
    # --------------------------------------------------------

    elif extensao == ".parquet":

        try:

            return pd.read_parquet(caminho)

        except Exception as erro:

            raise ValueError(
                f"Não foi possível ler o arquivo Parquet: {erro}"
            )

    # --------------------------------------------------------
    # FORMATO NÃO SUPORTADO
    # --------------------------------------------------------

    else:

        raise ValueError(
            f"Formato de arquivo não suportado: {extensao}"
        )