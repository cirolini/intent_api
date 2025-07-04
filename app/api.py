"""
API Blueprint para endpoints de health check, predição de intenções e construção do índice FAISS.
"""

import traceback

from flask import Blueprint, current_app, request, jsonify

from .model import predict
from .config import DEFAULT_THRESHOLD
from .utils import preprocess
from .builder import build_index

bp = Blueprint("api", __name__)


@bp.route("/health", methods=["GET"])
def health():
    """Endpoint de health check para verificação de disponibilidade."""
    current_app.logger.debug("Health check endpoint called")
    return jsonify(status="ok")


@bp.route("/predict", methods=["POST"])
def predict_route():
    """
    Endpoint de predição de intenções.
    Espera JSON com 'text' (obrigatório), 'top_k' e 'threshold' (opcionais).
    """
    try:
        payload = request.get_json(force=True)
        current_app.logger.debug("Received payload: %s", payload)

        text = payload.get("text", "").strip()
        if not text:
            current_app.logger.warning("Missing 'text' parameter in request")
            return jsonify(error="Missing 'text' parameter"), 400

        text = preprocess(text)
        top_k = int(payload.get("top_k", 5))
        threshold = float(payload.get("threshold", DEFAULT_THRESHOLD))

        current_app.logger.info(
            'Predicting intent for text="%s", top_k=%d, threshold=%s',
            text,
            top_k,
            threshold,
        )

        result = predict(text, top_k=top_k, threshold=threshold)
        current_app.logger.info("Prediction result: %s", result)

        return jsonify(result)

    except Exception as e:  # pylint: disable=broad-exception-caught
        # Justificado: captura ampla para retornar erro HTTP controlado em produção
        current_app.logger.exception("Erro ao processar /predict")
        excerpt = "\n".join(traceback.format_exc().splitlines()[-3:])
        return (
            jsonify(error="Internal server error", detail=str(e), excerpt=excerpt),
            500,
        )


@bp.route("/build_index", methods=["POST"])
def build_index_route():
    """
    Endpoint para construção (ou reconstrução) do índice FAISS.
    Retorna os caminhos dos arquivos gerados.
    """
    current_app.logger.info("Iniciando geração do índice FAISS e labels")
    try:
        index_path, labels_path = build_index()
        current_app.logger.info(
            "Index gerado em %s, labels em %s", index_path, labels_path
        )
        return jsonify(index_path=index_path, labels_path=labels_path), 201

    except Exception as e:  # pylint: disable=broad-exception-caught
        current_app.logger.exception("Erro ao gerar índice")
        return jsonify(error="Falha ao gerar índice", detail=str(e)), 500
