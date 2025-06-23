"""
Módulo de inicialização da aplicação Flask e configuração de logging.
"""

import os
import logging
import click
from flask import Flask

# Importa blueprint de rotas e registra endpoints no app
from .api import bp as api_bp
from .builder import build_index
from .model import MODEL_NAME, DATA_DIR 

# Cria a instância da aplicação Flask
app = Flask(__name__)
app.register_blueprint(api_bp)

# Configura logging para saída em console
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
handler = logging.StreamHandler()
handler.setLevel(LOG_LEVEL)
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
handler.setFormatter(logging.Formatter(LOG_FORMAT))
app.logger.setLevel(LOG_LEVEL)
app.logger.addHandler(handler)


@app.cli.command("build-index")
@click.option("--force", is_flag=True, help="Força a reconstrução do índice FAISS.")
def build_index_command(force):
    """Gera o índice FAISS e os labels (usado na inicialização do modelo)."""

    idx_path = os.path.join(DATA_DIR, f"{MODEL_NAME}.faiss")
    lbl_path = os.path.join(DATA_DIR, f"{MODEL_NAME}_labels.json")

    if force:
        for path in [idx_path, lbl_path]:
            try:
                os.remove(path)
                print(f"Removido: {path}")
            except FileNotFoundError:
                pass

    print("Gerando índice FAISS e labels...")
    idx_path, lbl_path = build_index()
    print(f"Index criado: {idx_path}")
    print(f"Labels salvos: {lbl_path}")
