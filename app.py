"""
🤖 BACKEND DEL AGENTE TECNOLÓGICO - DIFODS
==========================================

Sistema independiente para el Agente Tecnológico con 2 tareas:
1. Consultas sobre plataforma SIFODS (Qdrant)
2. Recomendación de cursos (Filtro Colaborativo Híbrido - una sola tabla)

CAMBIOS v2.0:
- El recomendador ahora trabaja sobre una sola tabla PostgreSQL
- El DataFrame se carga al inicio del servidor (una vez en memoria)
- Se eliminaron las tablas separadas: cursos, inscripciones, usuarios_perfil
- Nuevo endpoint /api/admin/refrescar-recomendador para actualizar datos
- El fallback de recomendación ahora lo maneja internamente HybridRecommender
"""

import os
import json
import logging
import pandas as pd
import psycopg2
import tiktoken

from typing import Optional, List, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── Sistema de recomendación (nueva versión una-sola-tabla) ───────────────────
from sistema_recomendacion import HybridRecommender, crear_recomendador

# ── Configuración del agente ──────────────────────────────────────────────────
from agente_tecnologico_config import (
    AGENTE_CONFIG,
    TAREA_SIFODS,
    TAREA_RECOMENDACION,
    PARAMETROS_GLOBALES,
    MENSAJES_AYUDA,
    PROMPT_BASE
)

import ssl
import certifi
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

# ══════════════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN INICIAL
# ══════════════════════════════════════════════════════════════════════════════

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=AGENTE_CONFIG["nombre"],
    version=AGENTE_CONFIG["version"],
    description=AGENTE_CONFIG["descripcion"]
)

templates = Jinja2Templates(directory="templates")

# ── Clientes externos ─────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_model  = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

logger.info(f"GEMINI KEY: {'OK' if os.getenv('GEMINI_API_KEY') else 'NO DEFINIDA'}")
logger.info(f"OPENAI KEY: {'OK' if os.getenv('OPENAI_API_KEY') else 'NO DEFINIDA'}")

qdrant_client   = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY")
)
embedding_model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
)
tokenizer       = tiktoken.encoding_for_model("gpt-4o-mini")
LIMA_TZ         = ZoneInfo("America/Lima")

# ── Recomendador global (se instancia en startup) ────────────────────────────
# Se carga una sola vez al iniciar el servidor para no repetir el cálculo
# de la matriz usuario-curso en cada request.
recomendador: Optional[HybridRecommender] = None


# ══════════════════════════════════════════════════════════════════════════════
# 📊 MODELOS DE DATOS (Pydantic)
# ══════════════════════════════════════════════════════════════════════════════

class ConsultaRequest(BaseModel):
    """Request para consultas generales."""
    mensaje:        str
    usuario:        str
    nombre_usuario: Optional[str] = None
    metadata:       Optional[Dict] = None


class RecomendacionRequest(BaseModel):
    """Request para recomendación de cursos."""
    usuario:        str
    nombre_usuario: Optional[str] = None
    top_k:          int = 5
    # nivel_educativo, especialidad y region ya NO son necesarios:
    # el sistema los lee directamente del historial del docente en el DataFrame.


# ══════════════════════════════════════════════════════════════════════════════
# 💾 BASE DE DATOS — CONEXIÓN Y TABLAS
# ══════════════════════════════════════════════════════════════════════════════

def get_db_connection():
    """Obtiene conexión a PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "agente_tecnologico"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            client_encoding="UTF8"
        )
        return conn
    except Exception as e:
        logger.error(f"Error de conexión a BD: {e}")
        raise


def crear_tablas():
    """Crea las tablas de conversaciones y métricas si no existen."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversaciones (
                id              SERIAL PRIMARY KEY,
                usuario         VARCHAR(100) NOT NULL,
                nombre_usuario  VARCHAR(200),
                mensaje         TEXT NOT NULL,
                respuesta       TEXT NOT NULL,
                tarea           VARCHAR(20) NOT NULL,
                fuente_datos    VARCHAR(50),
                tokens_entrada  INTEGER,
                tokens_salida   INTEGER,
                latencia_ms     INTEGER,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metricas (
                id                  SERIAL PRIMARY KEY,
                fecha               DATE NOT NULL,
                tarea               VARCHAR(20) NOT NULL,
                total_consultas     INTEGER DEFAULT 0,
                latencia_promedio_ms INTEGER,
                tokens_totales      INTEGER,
                fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(fecha, tarea)
            )
        """)

        conn.commit()
        logger.info("✅ Tablas de conversaciones y métricas verificadas")

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error creando tablas: {e}")
    finally:
        if conn:
            conn.close()


def normalizar_texto(texto: str) -> str:
    """Normaliza texto para almacenar en BD sin errores de encoding."""
    return texto.encode("utf-8", errors="ignore").decode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# 🔄 CARGA DEL RECOMENDADOR
# ══════════════════════════════════════════════════════════════════════════════

def cargar_dataframe_recomendacion() -> pd.DataFrame:
    """
    Lee la tabla única de inscripciones/cursos desde PostgreSQL.

    Ajusta el nombre de la tabla ("inscripciones_cursos") al nombre
    real que uses en tu base de datos.
    """
    conn = get_db_connection()
    try:
        # ⚠️  AJUSTA el nombre de la tabla a tu esquema real
        df = pd.read_sql(
            "SELECT * FROM inscripciones_cursos",
            conn
        )
        logger.info(f"📊 DataFrame cargado: {len(df)} registros, {len(df.columns)} columnas")
        return df
    except Exception as e:
        logger.error(f"Error cargando DataFrame: {e}")
        raise
    finally:
        conn.close()


def inicializar_recomendador() -> Optional[HybridRecommender]:
    """
    Instancia HybridRecommender con el DataFrame cargado desde PostgreSQL.
    Devuelve None si falla (el servidor sigue funcionando sin recomendador).
    """
    try:
        df = cargar_dataframe_recomendacion()
        rec = crear_recomendador(df)
        logger.info("✅ Recomendador híbrido cargado correctamente")
        return rec
    except Exception as e:
        logger.warning(f"⚠️  Recomendador no pudo iniciarse: {e}")
        logger.warning("    El agente funcionará sin recomendación de cursos.")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 🤖 LLM CON FALLBACK (GPT → Gemini)
# ══════════════════════════════════════════════════════════════════════════════

async def llamar_llm_con_fallback(prompt: str, model_params: dict) -> str:
    """
    Intenta OpenAI GPT primero; si falla, usa Gemini como backup.

    Args:
        prompt:       Prompt completo a enviar.
        model_params: Dict con claves max_tokens y temperature.

    Returns:
        Texto de respuesta del LLM.
    """
    # INTENTO 1: OpenAI GPT
    try:
        logger.info("🔵 Intentando con OpenAI GPT...")
        response = openai_client.chat.completions.create(
            model=PARAMETROS_GLOBALES["modelo_llm"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=model_params["max_tokens"],
            temperature=model_params["temperature"]
        )
        respuesta = response.choices[0].message.content.strip()
        logger.info("✅ OpenAI GPT respondió correctamente")
        return respuesta

    except Exception as e:
        logger.warning(f"⚠️  OpenAI falló: {e} → cambiando a Gemini...")

    # INTENTO 2: Gemini como fallback
    try:
        response = gemini_model.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "temperature":      model_params["temperature"],
                "max_output_tokens": model_params["max_tokens"],
            }
        )
        respuesta = response.text.strip()
        logger.info("✅ Gemini respondió correctamente")
        return respuesta

    except Exception as e2:
        logger.error(f"❌ Gemini también falló: {e2}")
        raise HTTPException(
            status_code=503,
            detail="Ambos modelos (GPT y Gemini) no están disponibles"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 🔍 QDRANT — BÚSQUEDA SEMÁNTICA
# ══════════════════════════════════════════════════════════════════════════════

def search_qdrant(query: str, collection_name: str = "Curso_0", k: int = 5) -> List[Dict]:
    """
    Busca en Qdrant usando embeddings semánticos.

    Args:
        query:           Consulta del usuario.
        collection_name: Colección Qdrant (siempre "Curso_0" para este agente).
        k:               Número de chunks a recuperar.

    Returns:
        Lista de chunks con texto y metadatos.
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()

        search_result = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=k
        )

        chunks = []
        for point in search_result.points:
            chunks.append({
                "text":         point.payload.get("text", ""),
                "score":        point.score,
                "filename":     point.payload.get("filename", ""),
                "chunk":        point.payload.get("chunk", 0),
                "total_chunks": point.payload.get("total_chunks", 0),
                "metadata":     point.payload
            })

        return chunks

    except Exception as e:
        logger.error(f"Error en búsqueda Qdrant: {e}")
        return []


def formatear_chunk_para_contexto(chunk: Dict) -> str:
    """Formatea un chunk para incluirlo en el prompt como contexto."""
    filename = chunk.get("filename", "Documento")
    text     = chunk.get("text", "")
    return f"[{filename}]\n{text}\n"


# ══════════════════════════════════════════════════════════════════════════════
# 🛠️ UTILIDADES DE PROMPT Y CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def obtener_prompt_para_tarea(tarea: str, context: str, question: str) -> str:
    """Genera el prompt completo según la tarea."""
    if tarea == "sifods":
        config = TAREA_SIFODS
    elif tarea == "recomendacion":
        config = TAREA_RECOMENDACION
    else:
        raise ValueError(f"Tarea inválida: {tarea}")

    prompt  = PROMPT_BASE.format(context=context, question=question)
    prompt += "\n\n" + config["prompt_especializado"]
    return prompt


def obtener_parametros_modelo(tarea: str) -> dict:
    """Devuelve los parámetros LLM según la tarea."""
    if tarea == "sifods":
        return TAREA_SIFODS["parametros_modelo"]
    elif tarea == "recomendacion":
        return TAREA_RECOMENDACION["parametros_modelo"]
    return PARAMETROS_GLOBALES


def clasificar_consulta(pregunta: str, tarea_forzada: str = None) -> tuple:
    """
    Clasifica la consulta del usuario.

    Si la UI ya indicó la tarea (tarea_forzada), se usa directamente.
    Si no, se clasifica por keywords.

    Returns:
        (categoria, confianza) — categoria: "sifods" | "recomendacion"
    """
    if tarea_forzada and tarea_forzada in ["sifods", "recomendacion"]:
        return (tarea_forzada, 1.0)

    pregunta_lower = pregunta.lower()

    if any(kw in pregunta_lower for kw in TAREA_SIFODS["keywords_deteccion"]):
        return ("sifods", 0.85)

    if any(kw in pregunta_lower for kw in TAREA_RECOMENDACION["keywords_deteccion"]):
        return ("recomendacion", 0.85)

    # Por defecto SIFODS si es ambiguo
    return ("sifods", 0.5)


# ══════════════════════════════════════════════════════════════════════════════
# 📚 TAREA 1: CONSULTAS SIFODS
# ══════════════════════════════════════════════════════════════════════════════

async def procesar_consulta_sifods(mensaje: str, usuario: str) -> Dict:
    """
    Procesa consultas sobre la plataforma SIFODS usando RAG sobre Qdrant.

    Flujo:
        1. Embedding de la pregunta
        2. Búsqueda semántica en Qdrant (colección "Curso_0")
        3. Construcción del prompt con contexto
        4. Llamada al LLM (con fallback)

    Args:
        mensaje: Pregunta del usuario.
        usuario: ID del docente.

    Returns:
        Dict con respuesta, referencias y métricas.
    """
    timestamp_inicio = datetime.now(LIMA_TZ)

    try:
        # 1. Buscar en Qdrant
        chunks = search_qdrant(
            query=mensaje,
            collection_name=TAREA_SIFODS["coleccion_qdrant"],
            k=PARAMETROS_GLOBALES["limite_contexto"]
        )

        if not chunks:
            return {
                "respuesta":    MENSAJES_AYUDA["sin_resultados_sifods"],
                "tarea":        "sifods",
                "fuente_datos": "ninguna",
                "referencias":  []
            }

        # 2. Formatear contexto y prompt
        context = "\n\n".join(formatear_chunk_para_contexto(c) for c in chunks)
        prompt  = obtener_prompt_para_tarea("sifods", context, mensaje)

        # 3. Llamar al LLM
        model_params = obtener_parametros_modelo("sifods")
        respuesta    = await llamar_llm_con_fallback(prompt, model_params)

        # 4. Métricas
        timestamp_fin = datetime.now(LIMA_TZ)
        latencia_ms   = int((timestamp_fin - timestamp_inicio).total_seconds() * 1000)
        tokens_entrada = len(tokenizer.encode(prompt))
        tokens_salida  = len(tokenizer.encode(respuesta))

        referencias = [
            {"fuente": c["filename"], "relevancia": c["score"]}
            for c in chunks[:3]
        ]

        return {
            "respuesta":     respuesta,
            "tarea":         "sifods",
            "fuente_datos":  "qdrant",
            "referencias":   referencias,
            "tokens_entrada": tokens_entrada,
            "tokens_salida":  tokens_salida,
            "latencia_ms":    latencia_ms
        }

    except Exception as e:
        logger.error(f"Error en consulta SIFODS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 TAREA 2: RECOMENDACIÓN DE CURSOS
# ══════════════════════════════════════════════════════════════════════════════

def _formatear_respuesta_recomendaciones(recomendaciones: List[Dict]) -> str:
    """
    Convierte la lista de recomendaciones en texto markdown amigable.

    Usa los campos que devuelve el nuevo HybridRecommender:
        CURSO, HORAS_PROGRAMA, TASA_APROBACION, justificacion, score_final
    """
    if not recomendaciones:
        return MENSAJES_AYUDA["sin_recomendaciones"]

    texto = "📚 **Cursos Recomendados para Ti:**\n\n"

    for idx, rec in enumerate(recomendaciones, 1):
        nombre          = rec.get("CURSO", "Curso sin nombre")
        justificacion   = rec.get("justificacion", "Relevante para tu perfil")
        horas           = rec.get("HORAS_PROGRAMA", 0)
        tasa_aprobacion = rec.get("TASA_APROBACION", 0)
        año             = rec.get("AÑO", "")
        score           = rec.get("score_final", 0)

        texto += f"**{idx}. {nombre}**\n"
        texto += f"   🎯 {justificacion}\n"
        texto += f"   ⏱️ Duración: {horas} horas"
        if año:
            texto += f"  |  📅 Año: {año}"
        texto += "\n"
        texto += f"   ✅ Tasa de aprobación: {tasa_aprobacion * 100:.0f}%\n"
        texto += f"   📊 Score de relevancia: {score:.2f}\n\n"

    texto += "\n💡 **Tip:** Puedes inscribirte directamente desde tu panel de cursos en SIFODS."
    return texto


async def procesar_recomendacion_cursos(request: RecomendacionRequest) -> Dict:
    """
    Genera recomendaciones de cursos usando el filtro colaborativo híbrido.

    Usa el recomendador global cargado en memoria al iniciar el servidor.
    Si el recomendador no está disponible, devuelve un mensaje de error claro.

    Args:
        request: Datos del usuario (solo se necesita usuario y top_k).

    Returns:
        Dict con cursos recomendados y respuesta formateada.
    """
    global recomendador
    timestamp_inicio = datetime.now(LIMA_TZ)

    logger.info(f"🎯 Generando recomendaciones para docente: {request.usuario}")

    # ── Verificar que el recomendador esté disponible ─────────────────────────
    if recomendador is None:
        logger.warning("⚠️  Recomendador no disponible. Devolviendo mensaje de error.")
        return {
            "respuesta": (
                "⚠️ El sistema de recomendación no está disponible en este momento. "
                "Por favor, intenta más tarde o contacta al administrador."
            ),
            "tarea":        "recomendacion",
            "fuente_datos": "no_disponible",
            "recomendaciones": [],
            "tokens_entrada":  0,
            "tokens_salida":   0,
            "latencia_ms":     0
        }

    try:
        # ── Llamar al recomendador ────────────────────────────────────────────
        # El fallback interno (cursos populares) ya lo maneja HybridRecommender
        # si no hay suficientes datos del docente.
        recomendaciones = recomendador.recomendar_hibrido(
            user_id=str(request.usuario),
            top_k=request.top_k,
            incluir_justificacion=True
        )

        # ── Formatear respuesta ───────────────────────────────────────────────
        respuesta_texto = _formatear_respuesta_recomendaciones(recomendaciones)

        # ── Métricas ──────────────────────────────────────────────────────────
        timestamp_fin  = datetime.now(LIMA_TZ)
        latencia_ms    = int((timestamp_fin - timestamp_inicio).total_seconds() * 1000)
        tokens_salida  = sum(
            len(tokenizer.encode(r.get("justificacion", "")))
            for r in recomendaciones
        )

        # Extraer métodos usados para el log
        metodos_usados = list({
            m for r in recomendaciones
            for m in r.get("metodos_usados", [])
        })

        return {
            "respuesta":        respuesta_texto,
            "tarea":            "recomendacion",
            "fuente_datos":     "modelo_recomendacion_hibrido",
            "recomendaciones":  recomendaciones,
            "tokens_entrada":   0,          # El recomendador no usa tokens de entrada propios
            "tokens_salida":    tokens_salida,
            "latencia_ms":      latencia_ms,
            "metadata": {
                "total_recomendaciones": len(recomendaciones),
                "algoritmos_usados":     metodos_usados
            }
        }

    except Exception as e:
        logger.error(f"Error inesperado en recomendación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 💾 PERSISTENCIA — GUARDAR CONVERSACIONES
# ══════════════════════════════════════════════════════════════════════════════

def guardar_conversacion(request: ConsultaRequest, resultado: Dict):
    """Guarda una conversación general en la tabla conversaciones."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO conversaciones
            (usuario, nombre_usuario, mensaje, respuesta, tarea, fuente_datos,
             tokens_entrada, tokens_salida, latencia_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            normalizar_texto(request.usuario),
            normalizar_texto(request.nombre_usuario or ""),
            normalizar_texto(request.mensaje),
            normalizar_texto(resultado["respuesta"]),
            resultado["tarea"],
            resultado.get("fuente_datos"),
            resultado.get("tokens_entrada"),
            resultado.get("tokens_salida"),
            resultado.get("latencia_ms")
        ))

        conn.commit()
        logger.info(f"✅ Conversación guardada para usuario {request.usuario}")

    except Exception as e:
        if conn:
            conn.rollback()
        logger.warning(f"No se pudo guardar la conversación: {e}")
    finally:
        if conn:
            conn.close()


def guardar_conversacion_recomendacion(request: RecomendacionRequest, resultado: Dict):
    """Guarda una conversación de recomendación en la tabla conversaciones."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        mensaje_log = f"Recomendación de cursos solicitada (top {request.top_k})"

        cursor.execute("""
            INSERT INTO conversaciones
            (usuario, nombre_usuario, mensaje, respuesta, tarea, fuente_datos,
             tokens_entrada, tokens_salida, latencia_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            normalizar_texto(request.usuario),
            normalizar_texto(request.nombre_usuario or ""),
            normalizar_texto(mensaje_log),
            normalizar_texto(resultado["respuesta"]),
            "recomendacion",
            resultado.get("fuente_datos", "modelo_recomendacion"),
            resultado.get("tokens_entrada"),
            resultado.get("tokens_salida"),
            resultado.get("latencia_ms")
        ))

        conn.commit()

    except Exception as e:
        if conn:
            conn.rollback()
        logger.warning(f"No se pudo guardar la recomendación: {e}")
    finally:
        if conn:
            conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# 🌐 ENDPOINTS DE LA API
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """
    Inicializa el sistema al arranque del servidor:
        1. Crea tablas de conversaciones y métricas si no existen.
        2. Carga el DataFrame de inscripciones e instancia el recomendador.
    """
    global recomendador

    # 1. Tablas de conversaciones
    crear_tablas()

    # 2. Recomendador híbrido
    logger.info("🔄 Cargando recomendador de cursos...")
    recomendador = inicializar_recomendador()

    logger.info(
        f"🚀 {AGENTE_CONFIG['nombre']} v{AGENTE_CONFIG['version']} iniciado | "
        f"Puerto: {os.getenv('PORT', 7002)} | "
        f"Recomendador: {'✅ activo' if recomendador else '⚠️  no disponible'}"
    )


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Página principal del agente."""
    return templates.TemplateResponse(
        "agente_tecnologico.html",
        {"request": request, "agente": AGENTE_CONFIG}
    )


@app.get("/health")
async def health_check():
    """Health check del servicio."""
    return {
        "status":               "healthy",
        "agente":               AGENTE_CONFIG["id_agente"],
        "version":              AGENTE_CONFIG["version"],
        "recomendador_activo":  recomendador is not None,
        "timestamp":            datetime.now(LIMA_TZ).isoformat()
    }


@app.post("/api/consulta")
async def procesar_consulta(request: ConsultaRequest):
    """
    Endpoint principal: clasifica la consulta y la rutea a la tarea correcta.

    La tarjeta de la UI envía tarea_forzada en metadata cuando el usuario
    clickea una tarjeta específica. Si no viene, se clasifica por keywords.
    """
    try:
        # 1. Clasificar
        tarea_forzada = None
        if request.metadata and "tarea_forzada" in request.metadata:
            tarea_forzada = request.metadata["tarea_forzada"]

        categoria, confianza = clasificar_consulta(request.mensaje, tarea_forzada)
        logger.info(f"📋 Consulta clasificada: {categoria} (confianza: {confianza:.2f})")

        # 2. Rutear a la tarea correspondiente
        if categoria == "sifods":
            resultado = await procesar_consulta_sifods(
                mensaje=request.mensaje,
                usuario=request.usuario
            )

        elif categoria == "recomendacion":
            rec_request = RecomendacionRequest(
                usuario=request.usuario,
                nombre_usuario=request.nombre_usuario
            )
            resultado = await procesar_recomendacion_cursos(rec_request)

        else:
            resultado = {
                "respuesta":    MENSAJES_AYUDA["consulta_ambigua"],
                "tarea":        "aclaracion",
                "fuente_datos": "ninguna"
            }

        # 3. Guardar en BD
        guardar_conversacion(request, resultado)

        # 4. Responder
        return {
            "respuesta":        resultado["respuesta"],
            "tarea":            resultado["tarea"],
            "fuente_datos":     resultado.get("fuente_datos"),
            "referencias":      resultado.get("referencias", []),
            "recomendaciones":  resultado.get("recomendaciones", []),
            "metadata": {
                "tokens_entrada":           resultado.get("tokens_entrada"),
                "tokens_salida":            resultado.get("tokens_salida"),
                "latencia_ms":              resultado.get("latencia_ms"),
                "confianza_clasificacion":  confianza
            }
        }

    except Exception as e:
        logger.error(f"Error procesando consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recomendar")
async def recomendar_cursos(request: RecomendacionRequest):
    """
    Endpoint dedicado para recomendación de cursos.

    Útil cuando se quiere llamar directamente desde la UI
    sin pasar por el clasificador.
    """
    try:
        resultado = await procesar_recomendacion_cursos(request)
        guardar_conversacion_recomendacion(request, resultado)

        return {
            "recomendaciones":   resultado["recomendaciones"],
            "respuesta_formateada": resultado["respuesta"],
            "metadata": {
                "tokens_salida":         resultado.get("tokens_salida"),
                "latencia_ms":           resultado.get("latencia_ms"),
                "algoritmos_usados":     resultado.get("metadata", {}).get("algoritmos_usados", []),
                "total_recomendaciones": resultado.get("metadata", {}).get("total_recomendaciones", 0)
            }
        }

    except Exception as e:
        logger.error(f"Error en endpoint /api/recomendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/refrescar-recomendador")
async def refrescar_recomendador():
    """
    Recarga el DataFrame desde PostgreSQL y reinicia el recomendador.

    Usar cuando lleguen nuevos datos (nuevas inscripciones, aprobaciones, etc.)
    sin necesidad de reiniciar el servidor.

    ⚠️  Proteger este endpoint con autenticación en producción.
    """
    global recomendador
    try:
        logger.info("🔄 Refrescando recomendador...")
        df_nuevo   = cargar_dataframe_recomendacion()
        recomendador = crear_recomendador(df_nuevo)
        logger.info("✅ Recomendador actualizado")
        return {
            "status":  "ok",
            "mensaje": "Recomendador actualizado correctamente",
            "registros_cargados": len(df_nuevo),
            "timestamp": datetime.now(LIMA_TZ).isoformat()
        }
    except Exception as e:
        logger.error(f"Error refrescando recomendador: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 🚀 EJECUTAR SERVIDOR
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7002)),
        reload=False   # reload=False porque el recomendador está en memoria global
    )
