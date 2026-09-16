from utilidades import ler_arquivo
from pathlib import Path
import shutil
import time


# ============================================================
# CONFIGURAÇÃO DOS CAMINHOS
# ============================================================

# Pasta onde este programa está localizado
PASTA_FERRAMENTAS = Path(__file__).resolve().parent

# Pasta principal do projeto
PASTA_PROJETO = PASTA_FERRAMENTAS.parent

# Pastas de entrada e saída
PASTA_ENTRADA = PASTA_PROJETO / "ENTRADA"
PASTA_SAIDA = PASTA_PROJETO / "SAIDA"


# ============================================================
# LEITURA DOS DOCUMENTOS
# ============================================================

documentos = [
    arquivo
    for arquivo in PASTA_ENTRADA.iterdir()
    if arquivo.is_file()
]


# ============================================================
# VERIFICAÇÃO
# ============================================================

if not documentos:
    print("Nenhum documento foi encontrado na pasta ENTRADA.")
    exit()


# ============================================================
# LISTAGEM DOS DOCUMENTOS
# ============================================================

print("\nDocumentos disponíveis:\n")

for numero, documento in enumerate(documentos, start=1):
    print(f"{numero} - {documento.name}")


# ============================================================
# ESCOLHA DO DOCUMENTO
# ============================================================

while True:

    try:
        escolha = int(input("\nDigite o número do documento que deseja trabalhar: "))

        if 1 <= escolha <= len(documentos):
            break

        print("Número inválido. Escolha um dos documentos listados.")

    except ValueError:
        print("Digite apenas o número correspondente ao documento.")


documento_selecionado = documentos[escolha - 1]


# ============================================================
# PROCESSAMENTO
# ============================================================

print(f"\nOk, vamos trabalhar com esse documento: {documento_selecionado.name}")

time.sleep(5)


# ============================================================
# SALVAR NA PASTA SAIDA
# ============================================================

PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

destino = PASTA_SAIDA / documento_selecionado.name

shutil.copy2(documento_selecionado, destino)


# ============================================================
# FINALIZAÇÃO
# ============================================================

print("\nDocumento copiado com sucesso!")
print(f"Arquivo salvo em: {destino}")