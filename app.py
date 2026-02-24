"""
🤖 BACKEND DEL AGENTE TECNOLÓGICO - DIFODS
==========================================
v2.2 — Correcciones de escala y robustez

Correcciones respecto a v2.1:
    - Connection pooling (psycopg2.pool.ThreadedConnectionPool)
    - gemini_model inicializado condicionalmente (no falla si key es None)
    - refrescar_recomendador usa swap seguro (instancia nueva)
    - EXCEL_SHEET_NAME tipado correctamente como int si es numérico
    - _fuente_datos guardada en el recomendador para /health
"""

import os
import logging
import pandas as pd
import tiktoken

from typing import Optional, List, Dict
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from google import genai

import psycopg2
from psycopg2 import pool as pg_pool

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from sistema_recomendacion import HybridRecommender, crear_recomendador

from agente_tecnologico_config import (
    AGENTE_CONFIG, TAREA_SIFODS, TAREA_RECOMENDACION,
    PARAMETROS_GLOBALES, MENSAJES_AYUDA, PROMPT_BASE
)

import ssl, certifi
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

import warnings
warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable"
)
# ══════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN INICIAL
# ══════════════════════════════════════════════════════════════════════

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=AGENTE_CONFIG["nombre"],
    version=AGENTE_CONFIG["version"],
    description=AGENTE_CONFIG["descripcion"]
)

templates = Jinja2Templates(directory="templates")

_openai_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=_openai_key) if _openai_key else None

if not openai_client:
    logger.warning("⚠️  OPENAI_API_KEY no definida → se usará solo Gemini")

# Gemini: inicializar condicionalmente para no crashear si la key no está
_gemini_key   = os.getenv("GEMINI_API_KEY")
gemini_model  = genai.Client(api_key=_gemini_key) if _gemini_key else None
if not gemini_model:
    logger.warning("⚠️  GEMINI_API_KEY no definida → sin fallback LLM para SIFODS")

qdrant_client   = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None
)
embedding_model = SentenceTransformer(
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
)
tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
LIMA_TZ   = ZoneInfo("America/Lima")

# Recomendador global (instanciado en startup)
recomendador: Optional[HybridRecommender] = None

# ──────────────────────────────────────────────────────────────────────
# Columnas esperadas en la tabla/Excel de recomendación
# ──────────────────────────────────────────────────────────────────────
COLUMNAS_RECOMENDACION = [
    "AÑO", "TIPO_CONSTANCIA", "CURSO", "NOMBRE_DRE", "NOMBRE_UGEL",
    "USUARIO_DOCUMENTO", "NOMBRE_COMPLETO", "NIVELNEXUS",
    "APROBACION", "ID_OFERTA_FORMATIVA", "ID_GRUPO",
    "FECHA_NACIMIENTO", "ES_FOCALIZADO", "HORAS_PROGRAMA",
    "CALIFICACIONES",       # rating del curso (0-5)
    "PROPOSITO", "ACTIVO", "PUBLICO_OBJETIVO", "CURSO_CULMINADO", "EDAD",
]


# ══════════════════════════════════════════════════════════════════════
# 📊 MODELOS DE DATOS (Pydantic)
# ══════════════════════════════════════════════════════════════════════

class ConsultaRequest(BaseModel):
    mensaje:        str
    usuario:        str
    nombre_usuario: Optional[str] = None
    metadata:       Optional[Dict] = None


class RecomendacionRequest(BaseModel):
    usuario:        str
    nombre_usuario: Optional[str] = None
    top_k:          int = 3


# ══════════════════════════════════════════════════════════════════════
# 💾 CONNECTION POOL (reemplaza conexión-por-request)
# ══════════════════════════════════════════════════════════════════════

_db_pool: Optional[pg_pool.ThreadedConnectionPool] = None


def _get_pool_kwargs() -> dict:
    return dict(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "agente_tecnologico"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        port=int(os.getenv("DB_PORT", 5432)),
        client_encoding="UTF8"
    )


def inicializar_pool():
    """Crea el pool de conexiones al arrancar el servidor."""
    global _db_pool
    try:
        _db_pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **_get_pool_kwargs()
        )
        logger.info("✅ Connection pool PostgreSQL creado (2-10 conexiones)")
    except Exception as e:
        logger.error(f"❌ No se pudo crear connection pool: {e}")
        _db_pool = None


def get_db_connection():
    """
    Obtiene una conexión del pool.
    Usar con contextmanager o devolver con devolver_conexion().
    """
    if _db_pool:
        return _db_pool.getconn()
    # Fallback a conexión directa si el pool no está disponible
    return psycopg2.connect(**_get_pool_kwargs())


def devolver_conexion(conn):
    """Devuelve la conexión al pool (o cierra si es conexión directa)."""
    if _db_pool and conn:
        _db_pool.putconn(conn)
    elif conn:
        conn.close()


def normalizar_texto(texto: str) -> str:
    return (texto or "").encode("utf-8", errors="ignore").decode("utf-8")


def crear_tablas():
    """Crea las tablas operacionales del agente si no existen."""
    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversaciones_agente (
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
                timestamp       TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS metricas (
                id                   SERIAL PRIMARY KEY,
                fecha                DATE NOT NULL,
                tarea                VARCHAR(20) NOT NULL,
                total_consultas      INTEGER DEFAULT 0,
                latencia_promedio_ms INTEGER,
                tokens_totales       INTEGER,
                fecha_actualizacion  TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(fecha, tarea)
            )
        """)
        conn.commit()
        logger.info("✅ Tablas operacionales verificadas/creadas")
    except Exception as e:
        if conn: conn.rollback()
        logger.error(f"Error creando tablas: {e}")
    finally:
        devolver_conexion(conn)


# ══════════════════════════════════════════════════════════════════════
# 📥 CARGA DE DATOS PARA EL RECOMENDADOR
# ══════════════════════════════════════════════════════════════════════

def _seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    cols_presentes = [c for c in COLUMNAS_RECOMENDACION if c in df.columns]
    cols_faltantes = [c for c in COLUMNAS_RECOMENDACION if c not in df.columns]
    if cols_faltantes:
        logger.warning(f"⚠️  Columnas no encontradas en la fuente: {cols_faltantes}")
    return df[cols_presentes]


def cargar_desde_postgres() -> pd.DataFrame:
    tabla  = os.getenv("DB_TABLE_RECOMENDACION")
    schema = os.getenv("DB_SCHEMA", "public")

    if not tabla:
        raise EnvironmentError(
            "DB_TABLE_RECOMENDACION no definida. "
            "Ej: DB_TABLE_RECOMENDACION=inscripciones_cursos"
        )

    conn = get_db_connection()
    try:
        query = f'SELECT * FROM "{schema}"."{tabla}"'
        df    = pd.read_sql(query, conn)
        df.columns = [c.strip().upper() for c in df.columns]
        logger.info(f"✅ PostgreSQL → {len(df):,} registros desde {schema}.{tabla}")
        return _seleccionar_columnas(df)
    finally:
        devolver_conexion(conn)


def cargar_desde_excel() -> pd.DataFrame:
    path_env = os.getenv("EXCEL_FALLBACK_PATH")

    if not path_env:
        raise EnvironmentError("EXCEL_FALLBACK_PATH no definida.")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, path_env)

    logger.info(f"📂 Buscando Excel en: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel no encontrado: {path}")

    sheet_raw = os.getenv("EXCEL_SHEET_NAME", "0")
    try:
        sheet = int(sheet_raw)
    except ValueError:
        sheet = sheet_raw

    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [c.strip().upper() for c in df.columns]

    logger.info(f"✅ Excel → {len(df):,} registros desde {path} (hoja: {sheet})")
    return _seleccionar_columnas(df)


def cargar_dataframe_recomendacion() -> tuple[pd.DataFrame, str]:
    """
    Retorna (df, fuente) donde fuente es 'postgresql' o 'excel'.
    """
    try:
        logger.info("🔄 Cargando datos desde PostgreSQL...")
        df = cargar_desde_postgres()
        return df, "postgresql"
    except Exception as e_pg:
        logger.warning(f"⚠️  PostgreSQL falló: {e_pg}")

    try:
        logger.info("🔄 Fallback: cargando desde Excel...")
        df = cargar_desde_excel()
        return df, "excel"
    except Exception as e_xl:
        logger.error(f"❌ Excel también falló: {e_xl}")

    raise RuntimeError(
        "No se pudo cargar datos desde PostgreSQL ni desde Excel. "
        "Revisa DB_TABLE_RECOMENDACION y EXCEL_FALLBACK_PATH en .env"
    )


def inicializar_recomendador() -> Optional[HybridRecommender]:
    try:
        df, fuente = cargar_dataframe_recomendacion()
        rec = crear_recomendador(df)
        rec._fuente_datos = fuente   # guardar para /health
        return rec
    except Exception as e:
        logger.warning(
            f"⚠️  Recomendador no disponible: {e}\n"
            "    Las consultas de SIFODS seguirán funcionando."
        )
        return None


# ══════════════════════════════════════════════════════════════════════
# 🤖 LLM CON FALLBACK (GPT → Gemini) — para consultas SIFODS
# ══════════════════════════════════════════════════════════════════════

async def llamar_llm_con_fallback(prompt: str, model_params: dict) -> str:
    # Intento 1: OpenAI
    if openai_client:
        try:
            resp = openai_client.chat.completions.create(
                model=PARAMETROS_GLOBALES["modelo_llm"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=model_params["max_tokens"],
                temperature=model_params["temperature"]
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"⚠️  OpenAI falló: {e} → Gemini...")

    # Intento 2: Gemini (solo si está disponible)
    if gemini_model:
        try:
            resp = gemini_model.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "temperature":       model_params["temperature"],
                    "max_output_tokens": model_params["max_tokens"],
                }
            )
            return resp.text.strip()
        except Exception as e2:
            logger.error(f"❌ Gemini también falló: {e2}")

    raise HTTPException(status_code=503, detail="LLM no disponible temporalmente")


# ══════════════════════════════════════════════════════════════════════
# 🔍 QDRANT
# ══════════════════════════════════════════════════════════════════════

def search_qdrant(query: str, collection_name: str = "Curso_0", k: int = 10) -> List[Dict]:
    try:
        emb    = embedding_model.encode(query).tolist()
        result = qdrant_client.query_points(
            collection_name=collection_name,
            query=emb,
            limit=k
        )
        return [
            {
                "text":     p.payload.get("text", ""),
                "score":    p.score,
                "filename": p.payload.get("filename", ""),
                "chunk":    p.payload.get("chunk", 0),
            }
            for p in result.points
        ]
    except Exception as e:
        logger.error(f"Error en Qdrant: {e}")
        return []


def formatear_chunk_para_contexto(chunk: Dict) -> str:
    return f"[{chunk.get('filename', 'Documento')}]\n{chunk.get('text', '')}\n"


# ══════════════════════════════════════════════════════════════════════
# 🛠️ PROMPT Y CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════════════

def obtener_prompt_para_tarea(tarea: str, context: str, question: str) -> str:
    config = TAREA_SIFODS if tarea == "sifods" else TAREA_RECOMENDACION
    return PROMPT_BASE.format(context=context, question=question) + "\n\n" + config["prompt_especializado"]


def obtener_parametros_modelo(tarea: str) -> dict:
    if tarea == "sifods":        return TAREA_SIFODS["parametros_modelo"]
    if tarea == "recomendacion": return TAREA_RECOMENDACION["parametros_modelo"]
    return PARAMETROS_GLOBALES


def clasificar_consulta(pregunta: str, tarea_forzada: str = None) -> tuple:
    if tarea_forzada and tarea_forzada in ("sifods", "recomendacion"):
        return (tarea_forzada, 1.0)
    q = pregunta.lower()
    if any(kw in q for kw in TAREA_SIFODS["keywords_deteccion"]):
        return ("sifods", 0.85)
    if any(kw in q for kw in TAREA_RECOMENDACION["keywords_deteccion"]):
        return ("recomendacion", 0.85)
    return ("sifods", 0.5)


# ══════════════════════════════════════════════════════════════════════
# 📚 TAREA 1: CONSULTAS SIFODS
# ══════════════════════════════════════════════════════════════════════

async def procesar_consulta_sifods(mensaje: str, usuario: str) -> Dict:
    ts     = datetime.now(LIMA_TZ)
    chunks = search_qdrant(mensaje, TAREA_SIFODS["coleccion_qdrant"], PARAMETROS_GLOBALES["limite_contexto"])

    if not chunks:
        return {
            "respuesta":    MENSAJES_AYUDA["sin_resultados_sifods"],
            "tarea":        "sifods",
            "fuente_datos": "ninguna",
            "referencias":  [],
        }

    context   = "\n\n".join(formatear_chunk_para_contexto(c) for c in chunks)
    prompt    = obtener_prompt_para_tarea("sifods", context, mensaje)
    respuesta = await llamar_llm_con_fallback(prompt, obtener_parametros_modelo("sifods"))
    latencia  = int((datetime.now(LIMA_TZ) - ts).total_seconds() * 1000)

    return {
        "respuesta":      respuesta,
        "tarea":          "sifods",
        "fuente_datos":   "qdrant",
        "referencias":    [{"fuente": c["filename"], "relevancia": c["score"]} for c in chunks[:3]],
        "tokens_entrada": len(tokenizer.encode(prompt)),
        "tokens_salida":  len(tokenizer.encode(respuesta)),
        "latencia_ms":    latencia,
    }


# ══════════════════════════════════════════════════════════════════════
# 🎯 TAREA 2: RECOMENDACIÓN DE CURSOS
# ══════════════════════════════════════════════════════════════════════

def _formatear_respuesta_recomendaciones(recs: List[Dict]) -> str:
    if not recs:
        return MENSAJES_AYUDA["sin_recomendaciones"]

    texto = "📚 **Cursos Recomendados para Ti:**\n\n"
    for i, r in enumerate(recs, 1):
        cal      = r.get("CALIFICACION_PROM", 0)
        text_cal = f"⭐ {cal:.1f}/5.0" if cal > 0 else ""
        texto   += f"**{i}. {r.get('CURSO', 'Curso')}**\n"
        texto   += f"   🎯 {r.get('justificacion', 'Relevante para tu perfil')}\n"
        texto   += f"   ⏱️ {r.get('HORAS_PROGRAMA', 0)}h  |  "
        texto   += f"✅ {r.get('TASA_CULMINACION', 0)*100:.0f}% culminación"
        if text_cal:
            texto += f"  |  {text_cal}"
        texto += "\n"
        if r.get("PUBLICO_OBJETIVO"):
            texto += f"   👥 Dirigido a: {r['PUBLICO_OBJETIVO']}\n"
        texto += f"   📊 Score: {r.get('score_final', 0):.2f}\n\n"

    texto += "💡 Inscríbete directamente desde tu panel de cursos en SIFODS."
    return texto


async def procesar_recomendacion_cursos(request: RecomendacionRequest) -> Dict:
    global recomendador
    ts = datetime.now(LIMA_TZ)

    if recomendador is None:
        return {
            "respuesta":       (
                "⚠️ El sistema de recomendación no está disponible. "
                "Verifica la conexión a la base de datos o el Excel de respaldo."
            ),
            "tarea":           "recomendacion",
            "fuente_datos":    "no_disponible",
            "recomendaciones": [],
            "tokens_entrada":  0,
            "tokens_salida":   0,
            "latencia_ms":     0,
        }

    try:
        recs = recomendador.recomendar_hibrido(
            user_id=str(request.usuario),
            top_k=request.top_k,
            incluir_justificacion=True
        )

        latencia      = int((datetime.now(LIMA_TZ) - ts).total_seconds() * 1000)
        tokens_salida = sum(len(tokenizer.encode(r.get("justificacion", ""))) for r in recs)
        metodos       = list({m for r in recs for m in r.get("metodos_usados", [])})

        return {
            "respuesta":       _formatear_respuesta_recomendaciones(recs),
            "tarea":           "recomendacion",
            "fuente_datos":    "modelo_recomendacion_hibrido",
            "recomendaciones": recs,
            "tokens_entrada":  0,
            "tokens_salida":   tokens_salida,
            "latencia_ms":     latencia,
            "metadata":        {"total": len(recs), "algoritmos": metodos},
        }

    except Exception as e:
        logger.error(f"Error en recomendación: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════
# 💾 PERSISTENCIA (usa connection pool)
# ══════════════════════════════════════════════════════════════════════

def _guardar(usuario, nombre, mensaje, respuesta, tarea, fuente, te, ts_tok, lat):
    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO conversaciones_agente
            (usuario, nombre_usuario, mensaje, respuesta, tarea,
             fuente_datos, tokens_entrada, tokens_salida, latencia_ms)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            normalizar_texto(usuario),
            normalizar_texto(nombre or ""),
            normalizar_texto(mensaje),
            normalizar_texto(respuesta),
            tarea, fuente, te, ts_tok, lat
        ))
        conn.commit()
    except Exception as e:
        if conn:
            try: conn.rollback()
            except: pass
        logger.warning(f"No se pudo guardar conversación: {e}")
    finally:
        devolver_conexion(conn)


def guardar_conversacion(request: ConsultaRequest, resultado: Dict):
    _guardar(
        request.usuario, request.nombre_usuario,
        request.mensaje, resultado["respuesta"],
        resultado["tarea"], resultado.get("fuente_datos"),
        resultado.get("tokens_entrada"), resultado.get("tokens_salida"),
        resultado.get("latencia_ms")
    )


def guardar_conversacion_recomendacion(request: RecomendacionRequest, resultado: Dict):
    _guardar(
        request.usuario, request.nombre_usuario,
        f"Recomendación (top {request.top_k})", resultado["respuesta"],
        "recomendacion", resultado.get("fuente_datos"),
        resultado.get("tokens_entrada"), resultado.get("tokens_salida"),
        resultado.get("latencia_ms")
    )


# ══════════════════════════════════════════════════════════════════════
# 🌐 ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    global recomendador
    inicializar_pool()
    crear_tablas()
    logger.info("🔄 Inicializando recomendador...")
    recomendador = inicializar_recomendador()
    logger.info(
        f"🚀 {AGENTE_CONFIG['nombre']} v{AGENTE_CONFIG['version']} | "
        f"Puerto: {os.getenv('PORT', 7002)} | "
        f"Recomendador: {'✅ activo' if recomendador else '⚠️  no disponible'} | "
        f"Gemini: {'✅' if gemini_model else '⚠️  no configurado'}"
    )


@app.on_event("shutdown")
async def shutdown_event():
    if _db_pool:
        _db_pool.closeall()
        logger.info("🔒 Connection pool cerrado")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "agente_tecnologico.html", {"request": request, "agente": AGENTE_CONFIG}
    )


@app.get("/health")
async def health_check():
    return {
        "status":              "healthy",
        "agente":              AGENTE_CONFIG["id_agente"],
        "version":             AGENTE_CONFIG["version"],
        "recomendador_activo": recomendador is not None,
        "fuente_datos":        getattr(recomendador, "_fuente_datos", "sin_datos"),
        "gemini_disponible":   gemini_model is not None,
        "db_pool_activo":      _db_pool is not None,
        "timestamp":           datetime.now(LIMA_TZ).isoformat()
    }


@app.post("/api/consulta")
async def procesar_consulta(request: ConsultaRequest):
    try:
        tarea_forzada       = (request.metadata or {}).get("tarea_forzada")
        categoria, confianza = clasificar_consulta(request.mensaje, tarea_forzada)

        if categoria == "sifods":
            resultado = await procesar_consulta_sifods(request.mensaje, request.usuario)
        elif categoria == "recomendacion":
            resultado = await procesar_recomendacion_cursos(
                RecomendacionRequest(
                    usuario=request.usuario,
                    nombre_usuario=request.nombre_usuario
                )
            )
        else:
            resultado = {
                "respuesta":    MENSAJES_AYUDA["consulta_ambigua"],
                "tarea":        "aclaracion",
                "fuente_datos": "ninguna",
            }

        guardar_conversacion(request, resultado)

        return {
            "respuesta":       resultado["respuesta"],
            "tarea":           resultado["tarea"],
            "fuente_datos":    resultado.get("fuente_datos"),
            "referencias":     resultado.get("referencias", []),
            "recomendaciones": resultado.get("recomendaciones", []),
            "metadata": {
                "tokens_entrada":          resultado.get("tokens_entrada"),
                "tokens_salida":           resultado.get("tokens_salida"),
                "latencia_ms":             resultado.get("latencia_ms"),
                "confianza_clasificacion": confianza,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /api/consulta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recomendar")
async def recomendar_cursos(request: RecomendacionRequest):
    logger.info(f"📨 Request recibido → usuario: {request.usuario} | top_k: {request.top_k}")
    try:
        resultado = await procesar_recomendacion_cursos(request)
        guardar_conversacion_recomendacion(request, resultado)
        return {
            "recomendaciones":      resultado["recomendaciones"],
            "respuesta_formateada": resultado["respuesta"],
            "metadata": {
                "latencia_ms": resultado.get("latencia_ms"),
                "algoritmos":  resultado.get("metadata", {}).get("algoritmos", []),
                "total":       resultado.get("metadata", {}).get("total", 0),
                "fuente_datos": resultado.get("fuente_datos"),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /api/recomendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/refrescar-recomendador")
async def refrescar_recomendador():
    """
    Recarga datos sin reiniciar el servidor.
    Usa swap seguro: crea nueva instancia antes de reemplazar.
    ⚠️  Proteger con autenticación en producción.
    """
    global recomendador
    try:
        df, fuente   = cargar_dataframe_recomendacion()
        nuevo_rec    = crear_recomendador(df)
        nuevo_rec._fuente_datos = fuente
        recomendador = nuevo_rec   # swap atómico
        return {
            "status":      "ok",
            "registros":   len(df),
            "fuente":      fuente,
            "timestamp":   datetime.now(LIMA_TZ).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════════
# 🚀 SERVIDOR
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 7002)),
        reload=False   # False: el recomendador vive en memoria global
    )