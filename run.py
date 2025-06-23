"""
Entrypoint da API Flask para o serviço de detecção de intents.
"""

import os
from app import app
from app.model import load_model

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

    app.logger.info("Starting server at http://%s:%s (debug=%s)", host, port, debug)

    print("Inicializando recursos da API (modelo, índice FAISS, labels)...")
    load_model()
    print("Recursos carregados com sucesso. Servidor pronto para receber requisições.")

    app.run(host=host, port=port, debug=debug)
