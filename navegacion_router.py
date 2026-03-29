"""
navegacion_router.py
====================
Módulo de routing híbrido para el Agente Tecnológico.
Se integra en main.py reemplazando la función search_qdrant
en el flujo de consultas SIFODS.

USO EN main.py:
    from navegacion_router import NavegacionRouter

    # Al startup (una sola vez):
    nav_router = NavegacionRouter(qdrant_client, settings)

    # En procesar_consulta_sifods:
    chunks = nav_router.search(mensaje)
"""

import json
import logging
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# CONFIG DEL ROUTER (independiente de settings.py)
# ══════════════════════════════════════════════════════════════════════

ROUTER_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
ROUTER_TREE_PATH       = "estructura_rag.json"

SIMILARITY_THRESHOLD   = 0.82   # score alto → fast path directo
DIFF_THRESHOLD         = 0.03   # diff #1-#2 mínima → si menor, activa LLM
LOW_SCORE_THRESHOLD    = 0.60   # score muy bajo → LLM siempre
TOP_K_CANDIDATES       = 3

# Campos del payload en Qdrant
# Ajusta si tus colecciones usan nombres distintos
PAYLOAD_TEXT    = "text"
PAYLOAD_SECTION = "filename"   # en tu metadata está filename, no section
PAYLOAD_SOURCE  = "filename"
PAYLOAD_CHUNK   = "chunk"


PROMPT_ROUTER = """\
[ROL]
Eres un clasificador de intenciones para la plataforma SIFODS \
(plataforma educativa para docentes del Ministerio de Educación del Perú).

[TAREA]
Elige el candidato que mejor responde a la intención del usuario.
Considera que pueden mencionar regiones peruanas (Ancash, Puno, Cusco, Lima, etc.)
o secciones de la plataforma por su nombre coloquial.

[CONSULTA]
{query}

[CANDIDATOS]
{candidates}

[FORMATO]
Responde ÚNICAMENTE con el knowledge_base elegido. Ejemplo: kb_login_acceso
"""


# ══════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class NavegacionRouter:
    """
    Router híbrido que:
      1. Usa embeddings multilingual-e5 para routing rápido
      2. Cae a LLM Gemini si hay ambigüedad (empate o score bajo)
      3. Busca en la colección Qdrant correcta según el nodo elegido

    Se instancia UNA SOLA VEZ al startup del agente y se reutiliza
    en cada request (el índice de nodos se mantiene en memoria).
    """

    def __init__(self, qdrant_client, settings, gemini_client=None):
        """
        Args:
            qdrant_client : instancia de QdrantClient ya inicializada en main.py
            settings      : objeto Settings de config/settings.py
            gemini_client : instancia de genai.Client ya inicializada en main.py
                            Si es None, el fallback LLM queda deshabilitado
                            (solo se usará el fast path por embeddings).
        """
        self.qdrant   = qdrant_client
        self.settings = settings
        self.gemini   = gemini_client
        self.top_k    = settings.qdrant.top_k

        # Carga el modelo de embeddings para routing
        # (distinto al embedding_model de settings que usa bge-base-en)
        logger.info(f"[NavRouter] Cargando modelo de routing: {ROUTER_EMBEDDING_MODEL}")
        self._embed_model = SentenceTransformer(ROUTER_EMBEDDING_MODEL)

        # Construye el índice de nodos en memoria
        self._node_index: list = []
        self._build_index(ROUTER_TREE_PATH)

    # ──────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DEL ÍNDICE
    # ──────────────────────────────────────────────────────────────────

    def _build_embed_text(self, node: dict) -> str:
        base = (
            f"{node['name']}. "
            f"{node['description']}. "
            f"Palabras clave: {', '.join(node.get('keywords', []))}."
        )
        questions = node.get("hypothetical_questions", [])
        if questions:
            base += " Preguntas que responde: " + " | ".join(questions)
        return base

    def _extract_leaves(self, node: dict, path: list = []) -> list:
        current_path = path + [node["name"]]
        leaves = []
        if node.get("knowledge_base"):
            leaves.append({
                "name":           node["name"],
                "description":    node["description"],
                "keywords":       node.get("keywords", []),
                "knowledge_base": node["knowledge_base"],
                "path":           " > ".join(current_path),
                "embed_text":     self._build_embed_text(node),
            })
        for child in node.get("children", []):
            leaves.extend(self._extract_leaves(child, current_path))
        return leaves

    def _build_index(self, tree_path: str) -> None:
        try:
            with open(tree_path, "r", encoding="utf-8") as f:
                tree = json.load(f)
        except FileNotFoundError:
            logger.error(f"[NavRouter] ❌ No se encontró {tree_path}")
            return

        # Entramos directo a Navegabilidad del Sistema
        navegabilidad = tree["children"][0]
        leaves = self._extract_leaves(navegabilidad)

        # multilingual-e5 requiere prefijo "passage:" en documentos
        texts = [f"passage: {leaf['embed_text']}" for leaf in leaves]
        embeddings = self._embed_model.encode(texts, normalize_embeddings=True)

        for i, leaf in enumerate(leaves):
            leaf["embedding"] = np.array(embeddings[i])

        self._node_index = leaves
        logger.info(f"[NavRouter] ✅ Índice construido — {len(leaves)} nodos hoja")
        for leaf in leaves:
            logger.debug(f"[NavRouter]   • {leaf['knowledge_base']}")

    # ──────────────────────────────────────────────────────────────────
    # ROUTING
    # ──────────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        # multilingual-e5 requiere prefijo "query:" en las consultas
        emb = self._embed_model.encode(f"query: {query}", normalize_embeddings=True)
        return np.array(emb)

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    def _top_candidates(self, query_emb: np.ndarray) -> list:
        scored = [
            {**node, "score": self._cosine(query_emb, node["embedding"])}
            for node in self._node_index
        ]
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:TOP_K_CANDIDATES]

    def _decide(self, candidates: list) -> tuple[bool, str]:
        """Retorna (usar_fast_path, motivo)."""
        s1   = candidates[0]["score"]
        s2   = candidates[1]["score"] if len(candidates) > 1 else 0.0
        diff = s1 - s2

        if s1 >= SIMILARITY_THRESHOLD:
            return True,  f"score alto ({s1:.4f})"
        if s1 < LOW_SCORE_THRESHOLD:
            return False, f"score bajo ({s1:.4f})"
        if diff < DIFF_THRESHOLD:
            return False, f"empate (diff={diff:.4f})"
        return True, f"score ok ({s1:.4f})"

    def _llm_route(self, query: str, candidates: list) -> Optional[str]:
        """Usa Gemini para desambiguar. Retorna kb_name o None si falla."""
        if not self.gemini:
            logger.warning("[NavRouter] Gemini no disponible, usando best embedding")
            return None
        try:
            formatted = "\n".join(
                f"- {c['knowledge_base']} | {c['name']} | score: {c['score']:.4f}\n"
                f"  {c['description']}"
                for c in candidates
            )
            prompt = PROMPT_ROUTER.format(query=query, candidates=formatted)
            resp   = self.gemini.models.generate_content(
                model    = self.settings.llm.modelo_fallback,
                contents = prompt,
                config   = {"temperature": 0.0},
            )
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"[NavRouter] LLM router falló: {e}")
            return None

    def route(self, query: str) -> dict:
        """
        Retorna el nodo seleccionado con metadatos de routing.
        Útil para logging y trazabilidad.
        """
        if not self._node_index:
            logger.error("[NavRouter] Índice vacío — verifica estructura_rag_v2.json")
            return {"kb": self.settings.qdrant.coleccion, "node": "fallback", "method": "sin_indice"}

        query_emb  = self._embed_query(query)
        candidates = self._top_candidates(query_emb)
        fast_path, motivo = self._decide(candidates)

        if fast_path:
            selected = candidates[0]
            method   = f"embeddings ⚡ ({motivo})"
        else:
            kb_name  = self._llm_route(query, candidates)
            selected = (
                next((n for n in self._node_index if n["knowledge_base"] == kb_name), candidates[0])
                if kb_name else candidates[0]
            )
            method = f"llm 🤖 ({motivo})"

        logger.info(
            f"[NavRouter] '{query[:60]}' → {selected['knowledge_base']} "
            f"| {method} | score={candidates[0]['score']:.4f}"
        )

        return {
            "kb":         selected["knowledge_base"],
            "node":       selected["name"],
            "path":       selected["path"],
            "method":     method,
            "score":      round(candidates[0]["score"], 4),
            "candidates": [
                {"rank": i+1, "kb": c["knowledge_base"], "score": round(c["score"], 4)}
                for i, c in enumerate(candidates)
            ],
        }

    # ──────────────────────────────────────────────────────────────────
    # RETRIEVAL DESDE QDRANT
    # ──────────────────────────────────────────────────────────────────

    def _search_collection(self, collection_name: str, query: str) -> list[dict]:
        """Busca en una colección Qdrant específica."""
        query_vector = self._embed_query(query).tolist()
        try:
            result = self.qdrant.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=self.top_k,
                with_payload=True,
            )

            return [
                {
                    "text": p.payload.get(PAYLOAD_TEXT, ""),
                    "filename": p.payload.get(PAYLOAD_SECTION, ""),
                    "chunk": p.payload.get(PAYLOAD_CHUNK, 0),
                    "score": round(p.score, 4),
                }
                for p in result.points
            ]

        except Exception as e:
            logger.error(f"[NavRouter] Error buscando en '{collection_name}': {e}")
            return []

    # ──────────────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL — reemplaza search_qdrant en main.py
    # ──────────────────────────────────────────────────────────────────

    def search(self, query: str) -> list[dict]:
        """
        Pipeline completo: routing → colección correcta → chunks.

        Reemplaza directamente a search_qdrant(query) en main.py.
        Retorna el mismo formato que search_qdrant para compatibilidad:
          [{"text": ..., "score": ..., "filename": ..., "chunk": ...}]
        """
        routing = self.route(query)
        chunks  = self._search_collection(routing["kb"], query)

        # Si la colección está vacía o no existe, intenta la colección
        # general como fallback (settings.qdrant.coleccion)
        if not chunks:
            fallback_col = self.settings.qdrant.coleccion
            if fallback_col != routing["kb"]:
                logger.warning(
                    f"[NavRouter] '{routing['kb']}' sin resultados → "
                    f"fallback a '{fallback_col}'"
                )
                chunks = self._search_collection(fallback_col, query)

        logger.info(
            f"[NavRouter] {len(chunks)} chunks desde '{routing['kb']}'"
        )
        return chunks
