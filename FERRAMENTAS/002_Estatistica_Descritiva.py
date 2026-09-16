from utilidades import ler_arquivo
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
# RELATÓRIO
# ============================================================

relatorio = []


def exibir(texto=""):
    """Exibe no terminal e armazena para o relatório TXT."""
    print(texto)
    relatorio.append(str(texto))


# ============================================================
# FORMATAÇÃO DE VALORES
# ============================================================

def formatar_numero(valor):
    if pd.isna(valor):
        return "-"

    if isinstance(valor, float):
        return f"{valor:,.4f}"

    return f"{valor:,}"


def obter_moda(serie):
    serie_sem_nulos = serie.dropna()

    if serie_sem_nulos.empty:
        return "-", 0

    moda = serie_sem_nulos.mode()

    if moda.empty:
        return "-", 0

    valor_moda = moda.iloc[0]
    frequencia = (serie_sem_nulos == valor_moda).sum()

    return valor_moda, frequencia


# ============================================================
# INÍCIO
# ============================================================

exibir("=" * 70)
exibir("ESTATÍSTICA DESCRITIVA DO DATASET")
exibir("=" * 70)

# ------------------------------------------------------------
# Verifica arquivos disponíveis
# ------------------------------------------------------------

arquivos = [
    arquivo
    for arquivo in PASTA_ENTRADA.iterdir()
    if arquivo.is_file()
]

if not arquivos:
    exibir("\nNenhum arquivo encontrado na pasta ENTRADA.")
    raise SystemExit


exibir("\nARQUIVOS DISPONÍVEIS:")
exibir("-" * 70)

for i, arquivo in enumerate(arquivos, start=1):
    exibir(f"{i}. {arquivo.name}")


# ------------------------------------------------------------
# Escolha do arquivo
# ------------------------------------------------------------

while True:
    try:
        escolha = int(
            input("\nDigite o número do arquivo que deseja analisar: ")
        )

        if 1 <= escolha <= len(arquivos):
            break

        print("Número inválido. Escolha um arquivo da lista.")

    except ValueError:
        print("Digite apenas o número correspondente ao arquivo.")


arquivo_selecionado = arquivos[escolha - 1]

exibir(
    f"\nOk, vamos trabalhar com o documento: "
    f"{arquivo_selecionado.name}"
)


# ============================================================
# LEITURA DO DATASET
# ============================================================

try:
    df = ler_arquivo(arquivo_selecionado)

except Exception as erro:
    exibir("\nERRO AO LER O ARQUIVO:")
    exibir(str(erro))
    raise SystemExit


# ============================================================
# INFORMAÇÕES GERAIS
# ============================================================

exibir("\n" + "=" * 70)
exibir("INFORMAÇÕES GERAIS")
exibir("=" * 70)

exibir(f"Arquivo: {arquivo_selecionado.name}")
exibir(f"Linhas: {df.shape[0]:,}")
exibir(f"Colunas: {df.shape[1]:,}")


# ============================================================
# ESTATÍSTICA POR COLUNA
# ============================================================

for nome_coluna in df.columns:

    serie = df[nome_coluna]

    count = serie.count()
    valores_unicos = serie.nunique(dropna=True)
    nulos = serie.isna().sum()

    if len(serie) > 0:
        percentual_nulos = (nulos / len(serie)) * 100
    else:
        percentual_nulos = 0

    moda, frequencia_moda = obter_moda(serie)

    # --------------------------------------------------------
    # IDENTIFICAÇÃO DO TIPO
    # --------------------------------------------------------

    if pd.api.types.is_bool_dtype(serie):
        tipo = "Booleano"

    elif pd.api.types.is_datetime64_any_dtype(serie):
        tipo = "Data/Hora"

    elif pd.api.types.is_numeric_dtype(serie):
        tipo = "Numérico"

    else:
        tipo = "Categórico/Textual"


    # --------------------------------------------------------
    # CABEÇALHO DA COLUNA
    # --------------------------------------------------------

    exibir("\n" + "=" * 70)
    exibir(f"COLUNA: {nome_coluna}")
    exibir("=" * 70)

    exibir(f"Tipo de dado: {tipo}")
    exibir(f"Count: {count:,}")
    exibir(f"Valores únicos: {valores_unicos:,}")
    exibir(f"Valores ausentes: {nulos:,}")
    exibir(f"% de valores ausentes: {percentual_nulos:.2f}%")
    exibir(f"Moda: {moda}")
    exibir(f"Frequência da moda: {frequencia_moda:,}")


    # ========================================================
    # BOOLEANO
    # ========================================================

    if pd.api.types.is_bool_dtype(serie):

        verdadeiros = (serie == True).sum()
        falsos = (serie == False).sum()

        total_validos = verdadeiros + falsos

        if total_validos > 0:
            percentual_true = (
                verdadeiros / total_validos
            ) * 100

            percentual_false = (
                falsos / total_validos
            ) * 100
        else:
            percentual_true = 0
            percentual_false = 0

        exibir("\n--- ESTATÍSTICAS BOOLEANAS ---")

        exibir(f"True: {verdadeiros:,}")
        exibir(f"False: {falsos:,}")
        exibir(f"% True: {percentual_true:.2f}%")
        exibir(f"% False: {percentual_false:.2f}%")

        exibir("\n--- ESTATÍSTICAS NUMÉRICAS ---")

        exibir("Média: -")
        exibir("Mediana: -")
        exibir("Desvio padrão: -")
        exibir("Variância: -")
        exibir("Mínimo: -")
        exibir("Q1 (25%): -")
        exibir("Q2 (50%): -")
        exibir("Q3 (75%): -")
        exibir("Máximo: -")
        exibir("Amplitude: -")
        exibir("IQR: -")
        exibir("Coeficiente de variação: -")
        exibir("Assimetria (Skewness): -")
        exibir("Curtose (Kurtosis): -")


    # ========================================================
    # NUMÉRICO
    # ========================================================

    elif pd.api.types.is_numeric_dtype(serie):

        valores = serie.dropna()

        if not valores.empty:

            media = valores.mean()
            mediana = valores.median()
            desvio_padrao = valores.std()
            variancia = valores.var()

            minimo = valores.min()
            q1 = valores.quantile(0.25)
            q2 = valores.quantile(0.50)
            q3 = valores.quantile(0.75)
            maximo = valores.max()

            amplitude = maximo - minimo
            iqr = q3 - q1

            # Coeficiente de variação
            if media != 0:
                coef_variacao = (
                    desvio_padrao / abs(media)
                ) * 100
            else:
                coef_variacao = None

            assimetria = valores.skew()
            curtose = valores.kurtosis()

        else:
            media = mediana = desvio_padrao = None
            variancia = minimo = q1 = q2 = q3 = None
            maximo = amplitude = iqr = None
            coef_variacao = assimetria = curtose = None


        exibir("\n--- ESTATÍSTICAS NUMÉRICAS ---")

        exibir(
            f"Média: "
            f"{formatar_numero(media)}"
        )

        exibir(
            f"Mediana: "
            f"{formatar_numero(mediana)}"
        )

        exibir(
            f"Desvio padrão: "
            f"{formatar_numero(desvio_padrao)}"
        )

        exibir(
            f"Variância: "
            f"{formatar_numero(variancia)}"
        )

        exibir(
            f"Mínimo: "
            f"{formatar_numero(minimo)}"
        )

        exibir(
            f"Q1 (25%): "
            f"{formatar_numero(q1)}"
        )

        exibir(
            f"Q2 (50%): "
            f"{formatar_numero(q2)}"
        )

        exibir(
            f"Q3 (75%): "
            f"{formatar_numero(q3)}"
        )

        exibir(
            f"Máximo: "
            f"{formatar_numero(maximo)}"
        )

        exibir(
            f"Amplitude: "
            f"{formatar_numero(amplitude)}"
        )

        exibir(
            f"IQR: "
            f"{formatar_numero(iqr)}"
        )

        if coef_variacao is not None:
            exibir(
                f"Coeficiente de variação: "
                f"{coef_variacao:.2f}%"
            )
        else:
            exibir(
                "Coeficiente de variação: -"
            )

        exibir(
            f"Assimetria (Skewness): "
            f"{formatar_numero(assimetria)}"
        )

        exibir(
            f"Curtose (Kurtosis): "
            f"{formatar_numero(curtose)}"
        )


    # ========================================================
    # DATA / HORA
    # ========================================================

    elif pd.api.types.is_datetime64_any_dtype(serie):

        valores = serie.dropna()

        exibir("\n--- ESTATÍSTICAS DE DATA/HORA ---")

        if not valores.empty:

            data_minima = valores.min()
            data_maxima = valores.max()
            mediana_data = valores.median()

            periodo = data_maxima - data_minima

            exibir(
                f"Data mínima: "
                f"{data_minima}"
            )

            exibir(
                f"Data máxima: "
                f"{data_maxima}"
            )

            exibir(
                f"Mediana: "
                f"{mediana_data}"
            )

            exibir(
                f"Período abrangido: "
                f"{periodo}"
            )

        else:

            exibir("Data mínima: -")
            exibir("Data máxima: -")
            exibir("Mediana: -")
            exibir("Período abrangido: -")


    # ========================================================
    # CATEGÓRICO / TEXTUAL
    # ========================================================

    else:

        exibir("\n--- ESTATÍSTICAS CATEGÓRICAS/TEXTUAIS ---")

        exibir(
            f"Categoria mais frequente: "
            f"{moda}"
        )

        exibir(
            f"Frequência da categoria mais frequente: "
            f"{frequencia_moda:,}"
        )

        exibir(
            f"Quantidade de categorias únicas: "
            f"{valores_unicos:,}"
        )

        exibir("Média: -")
        exibir("Mediana: -")
        exibir("Desvio padrão: -")
        exibir("Variância: -")
        exibir("Mínimo: -")
        exibir("Q1 (25%): -")
        exibir("Q2 (50%): -")
        exibir("Q3 (75%): -")
        exibir("Máximo: -")
        exibir("Amplitude: -")
        exibir("IQR: -")
        exibir("Coeficiente de variação: -")
        exibir("Assimetria (Skewness): -")
        exibir("Curtose (Kurtosis): -")


# ============================================================
# SALVAR RELATÓRIO
# ============================================================

nome_saida = (
    f"002_Estatistica_Descritiva_"
    f"{arquivo_selecionado.stem}.txt"
)

caminho_saida = PASTA_SAIDA / nome_saida

with open(
    caminho_saida,
    "w",
    encoding="utf-8"
) as arquivo:

    arquivo.write("\n".join(relatorio))


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\n" + "=" * 70)
print("ANÁLISE CONCLUÍDA")
print("=" * 70)

print(
    f"Relatório salvo em:\n{caminho_saida}"
)