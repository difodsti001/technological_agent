"""
APLICACIÓN DEL AGENTE TECNOLÓGICO DIFODS
======================================
FastAPI principal. Orquesta los dos módulos:
  1. SIFODS   → RAG sobre Qdrant (consultas de plataforma)
  2. Cursos   → Filtro colaborativo híbrido (recomendación)

Toda la configuración viene de config/settings.py → .env
"""

import logging
import ssl
import certifi
import warnings
import asyncio
from datetime import datetime
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

import pandas as pd
import tiktoken
import psycopg2
from psycopg2 import pool as pg_pool

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI
from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from config.settings import settings
from config.prompts import (
    PROMPT_BASE,
    PROMPT_SIFODS,
    MENSAJES_AYUDA,
)

from sistema_recomendacion import HybridRecommender, crear_recomendador

ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy connectable")

logging.basicConfig(level=settings.servidor.log_level)
logger = logging.getLogger(__name__)

LIMA_TZ = ZoneInfo("America/Lima")

# ══════════════════════════════════════════════════════════════════════
# COLUMNAS ESPERADAS EN LA TABLA
# ══════════════════════════════════════════════════════════════════════

COLUMNAS_RECOMENDACION = [
    "AÑO", "TIPO_CONSTANCIA", "CURSO", "NOMBRE_DRE", "NOMBRE_UGEL",
    "USUARIO_DOCUMENTO", "NOMBRE_COMPLETO", "NIVELNEXUS",
    "APROBACION", "ID_OFERTA_FORMATIVA", "ID_GRUPO",
    "FECHA_NACIMIENTO", "ES_FOCALIZADO", "HORAS_PROGRAMA",
    "CALIFICACIONES", "PROPOSITO", "ACTIVO", "PUBLICO_OBJETIVO",
    "CURSO_CULMINADO", "EDAD",
]

# ══════════════════════════════════════════════════════════════════════
# MODELOS PYDANTIC
# ══════════════════════════════════════════════════════════════════════
class SifodsRequest(BaseModel):
    mensaje:        str
    usuario:        str
    nombre_usuario: Optional[str] = None

class RecomendacionRequest(BaseModel):
    usuario:        str
    nombre_usuario: Optional[str] = None
    top_k:          Optional[int] = None

# ══════════════════════════════════════════════════════════════════════
# APLICACIÓN FASTAPI
# ══════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = settings.agente.nombre,
    version     = settings.agente.version,
    description = settings.agente.descripcion,
)

templates = Jinja2Templates(directory="templates")

# ══════════════════════════════════════════════════════════════════════
# CLIENTES EXTERNOS
# ══════════════════════════════════════════════════════════════════════

openai_client: Optional[OpenAI] = (
    OpenAI(api_key=settings.llm.openai_api_key) if settings.llm.tiene_openai else None
)

gemini_model = None
if settings.llm.tiene_gemini:
    try:
        gemini_model = genai.Client(api_key=settings.llm.gemini_api_key)
    except Exception as e:
        logger.warning(f"⚠️  Gemini no disponible: {e}")

qdrant_client = QdrantClient(
    url     = settings.qdrant.url,
    api_key = settings.qdrant.api_key or None,
)

embedding_model = SentenceTransformer(settings.llm.embedding_model)
tokenizer       = tiktoken.encoding_for_model("gpt-4o-mini")

recomendador: Optional[HybridRecommender] = None


# ══════════════════════════════════════════════════════════════════════
# CONNECTION POOL POSTGRESQL
# ══════════════════════════════════════════════════════════════════════

_db_pool: Optional[pg_pool.ThreadedConnectionPool] = None


def inicializar_pool() -> None:
    global _db_pool
    try:
        _db_pool = pg_pool.ThreadedConnectionPool(
            minconn = settings.db.pool_min,
            maxconn = settings.db.pool_max,
            **settings.db.as_dict(),
        )
        logger.info(
            f"✅ Connection pool PostgreSQL ({settings.db.pool_min}–{settings.db.pool_max})"
        )
    except Exception as e:
        logger.error(f"❌ No se pudo crear connection pool: {e}")
        _db_pool = None


def get_db_connection():
    if _db_pool:
        return _db_pool.getconn()
    return psycopg2.connect(**settings.db.as_dict())


def devolver_conexion(conn) -> None:
    if _db_pool and conn:
        _db_pool.putconn(conn)
    elif conn:
        conn.close()


def normalizar_texto(texto: str) -> str:
    return (texto or "").encode("utf-8", errors="ignore").decode("utf-8")

# ══════════════════════════════════════════════════════════════════════
# CARGA DE DATOS PARA EL RECOMENDADOR
# ══════════════════════════════════════════════════════════════════════

_COLUMN_ALIASES = {
    "ANIO":             "AÑO",
    "ANO":              "AÑO",
    "YEAR":             "AÑO",
    "USUARIO_DOC":      "USUARIO_DOCUMENTO",
    "DNI":              "USUARIO_DOCUMENTO",
    "DOCUMENTO":        "USUARIO_DOCUMENTO",
    "NOMBRE":           "NOMBRE_COMPLETO",
    "NIVEL":            "NIVELNEXUS",
    "DRE":              "NOMBRE_DRE",
    "UGEL":             "NOMBRE_UGEL",
    "ID_OFERTA":        "ID_OFERTA_FORMATIVA",
    "APROBADO":         "APROBACION",
    "CULMINADO":        "CURSO_CULMINADO",
    "HORAS":            "HORAS_PROGRAMA",
    "CALIFICACION":     "CALIFICACIONES",
    "RATING":           "CALIFICACIONES",
}


def _aplicar_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas a mayúsculas y aplica aliases."""
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    cols = set(df.columns)
    for alias, canonico in _COLUMN_ALIASES.items():
        if alias in cols and canonico not in cols:
            df.rename(columns={alias: canonico}, inplace=True)
            logger.info(f"🔄 Columna '{alias}' → '{canonico}'")
            cols = set(df.columns)
    return df


def _seleccionar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = _aplicar_aliases(df)
    cols_presentes = [c for c in COLUMNAS_RECOMENDACION if c in df.columns]
    cols_faltantes = [c for c in COLUMNAS_RECOMENDACION if c not in df.columns]
    if cols_faltantes:
        logger.warning(f"⚠️  Columnas no encontradas: {cols_faltantes}")
    return df[cols_presentes]


def cargar_desde_postgres() -> pd.DataFrame:
    tabla  = settings.db.tabla_recomendacion
    schema = settings.db.schema

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
    import os
    path_env  = settings.db.excel_fallback_path
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    path      = os.path.join(base_dir, path_env)

    logger.info(f"📂 Buscando Excel en: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel no encontrado: {path}")

    sheet_raw = settings.db.excel_sheet_name
    try:
        sheet = int(sheet_raw)
    except ValueError:
        sheet = sheet_raw

    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    df.columns = [c.strip().upper() for c in df.columns]
    logger.info(f"✅ Excel → {len(df):,} registros (hoja: {sheet})")
    return _seleccionar_columnas(df)


def cargar_dataframe_recomendacion() -> tuple[pd.DataFrame, str]:
    """Retorna (df, fuente) probando primero PostgreSQL, luego Excel."""
    try:
        logger.info("🔄 Cargando datos desde PostgreSQL...")
        return cargar_desde_postgres(), "postgresql"
    except Exception as e_pg:
        logger.warning(f"⚠️  PostgreSQL falló: {e_pg}")

    try:
        logger.info("🔄 Fallback: cargando desde Excel...")
        return cargar_desde_excel(), "excel"
    except Exception as e_xl:
        logger.error(f"❌ Excel también falló: {e_xl}")

    raise RuntimeError(
        "No se pudo cargar datos desde PostgreSQL ni desde Excel. "
        f"Revisa DB_TABLE_RECOMENDACION='{settings.db.tabla_recomendacion}' "
        f"y EXCEL_FALLBACK_PATH='{settings.db.excel_fallback_path}' en .env"
    )


def inicializar_recomendador() -> Optional[HybridRecommender]:
    try:
        df, fuente = cargar_dataframe_recomendacion()
        rec = crear_recomendador(df)
        rec._fuente_datos = fuente
        return rec
    except Exception as e:
        logger.warning(
            f"⚠️  Recomendador no disponible: {e}\n"
            "    Las consultas de SIFODS seguirán funcionando."
        )
        return None


# ══════════════════════════════════════════════════════════════════════
# LLM CON FALLBACK (GPT → Gemini)
# ══════════════════════════════════════════════════════════════════════

async def llamar_llm_con_fallback(prompt: str, model_params: dict) -> str:
    loop = asyncio.get_event_loop()

    if openai_client:
        try:
            def _openai():
                return openai_client.chat.completions.create(
                    model       = settings.llm.modelo_principal,
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = model_params["max_tokens"],
                    temperature = model_params["temperature"],
                )
            resp  = await loop.run_in_executor(None, _openai)
            texto = resp.choices[0].message.content.strip()
            logger.info(f"✅ OpenAI | tokens_salida: {resp.usage.completion_tokens} | finish: {resp.choices[0].finish_reason}")
            return texto
        except Exception as e:
            logger.warning(f"⚠️  OpenAI falló: {e} → Gemini...")

    if gemini_model:
        try:
            from google.genai import types as genai_types

            def _gemini():
                cfg = genai_types.GenerateContentConfig(
                    temperature       = model_params["temperature"],
                    max_output_tokens = model_params["max_tokens"],
                    top_p             = model_params.get("top_p") or None,
                )
                return gemini_model.models.generate_content(
                    model    = settings.llm.modelo_fallback,
                    contents = prompt,
                    config   = cfg,
                )
            logger.info(f"🔍 Gemini config → max_output_tokens: {model_params['max_tokens']} | temperature: {model_params['temperature']}")
            resp  = await loop.run_in_executor(None, _gemini)
            texto = resp.text.strip()
            finish = getattr(resp.candidates[0], "finish_reason", "?") if resp.candidates else "?"
            logger.info(f"✅ Gemini | chars: {len(texto)} | finish: {finish}")
            return texto
        except Exception as e2:
            logger.error(f"❌ Gemini también falló: {e2}")

    raise HTTPException(status_code=503, detail="LLM no disponible temporalmente")



# ══════════════════════════════════════════════════════════════════════
# QDRANT
# ══════════════════════════════════════════════════════════════════════

def search_qdrant(query: str) -> List[Dict]:
    try:
        emb    = embedding_model.encode(query).tolist()
        result = qdrant_client.query_points(
            collection_name = settings.qdrant.coleccion,
            query           = emb,
            limit           = settings.qdrant.top_k,
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


def _formatear_chunk(chunk: Dict) -> str:
    return f"[{chunk.get('filename', 'Documento')}]\n{chunk.get('text', '')}\n"


# ══════════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE TAREAS
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# TAREA 1: CONSULTAS SIFODS
# ══════════════════════════════════════════════════════════════════════

async def procesar_consulta_sifods(mensaje: str, usuario: str) -> Dict:
    ts     = datetime.now(LIMA_TZ)
    chunks = search_qdrant(mensaje)

    if not chunks:
        return {
            "respuesta":    MENSAJES_AYUDA["sin_resultados_sifods"],
            "tarea":        "sifods",
            "fuente_datos": "ninguna",
            "referencias":  [],
        }

    context   = "\n\n".join(_formatear_chunk(c) for c in chunks)
    prompt    = PROMPT_BASE.format(context=context, question=mensaje) + "\n\n" + PROMPT_SIFODS
    respuesta = await llamar_llm_con_fallback(prompt, settings.sifods.parametros_modelo)
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
# TAREA 2: RECOMENDACIÓN DE CURSOS
# ══════════════════════════════════════════════════════════════════════

def _formatear_respuesta_recomendaciones(recs: List[Dict]) -> str:
    if not recs:
        return MENSAJES_AYUDA["sin_recomendaciones"]

    texto = "📚 **Cursos Recomendados para Ti:**\n\n"
    for i, r in enumerate(recs, 1):
        cal      = r.get("CALIFICACION_PROM", 0)
        text_cal = f"⭐ {cal:.1f}/5.0" if cal > 0 else ""
        texto   += f"**{i}. {r.get('CURSO', 'Curso')}**\n"
        texto   += f" 🎯 {r.get('justificacion', 'Relevante para tu perfil')}\n"
        texto   += f" ⏱️ {r.get('HORAS_PROGRAMA', 0)}h  |  "
        texto   += f"✅ {r.get('TASA_CULMINACION', 0) * 100:.0f}% culminación"
        if text_cal:
            texto += f"  |  {text_cal}"
        texto += "\n"
        if r.get("PUBLICO_OBJETIVO"):
            texto += f" 👥 Dirigido a: {r['PUBLICO_OBJETIVO']}\n"
        texto += f" 📊 Score: {r.get('score_final', 0):.2f}\n\n"

    texto += "💡 Inscríbete directamente desde tu panel de cursos en SIFODS."
    return texto


async def procesar_recomendacion_cursos(request: RecomendacionRequest) -> Dict:
    ts = datetime.now(LIMA_TZ)

    if recomendador is None:
        return {
            "respuesta":       MENSAJES_AYUDA["recomendador_no_disponible"],
            "tarea":           "recomendacion",
            "fuente_datos":    "no_disponible",
            "recomendaciones": [],
            "tokens_entrada":  0,
            "tokens_salida":   0,
            "latencia_ms":     0,
        }

    try:
        top_k = request.top_k if request.top_k and request.top_k > 0 else settings.recomendacion.top_k

        recs = recomendador.recomendar_hibrido(
            user_id              = str(request.usuario),
            top_k                = top_k,
            incluir_justificacion= True,
        )

        latencia     = int((datetime.now(LIMA_TZ) - ts).total_seconds() * 1000)
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
# PERSISTENCIA
# ══════════════════════════════════════════════════════════════════════

def _guardar(usuario, nombre, mensaje, respuesta, tarea, fuente, te, ts_tok, lat) -> Optional[int]:
    """
    Inserta en conversaciones_agente y retorna el id generado.
    Retorna None si guardar_conversaciones=False o si falla.
    """
    if not settings.servidor.guardar_conversaciones:
        return None
    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute("""
            INSERT INTO conversaciones_agente
            (usuario, nombre_usuario, mensaje, respuesta, tarea,
             fuente_datos, tokens_entrada, tokens_salida, latencia_ms)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
        """, (
            normalizar_texto(usuario),
            normalizar_texto(nombre or ""),
            normalizar_texto(mensaje),
            normalizar_texto(respuesta),
            tarea, fuente, te, ts_tok, lat,
        ))
        conversacion_id = cur.fetchone()[0]
        conn.commit()
        return conversacion_id
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        logger.warning(f"No se pudo guardar conversación: {e}")
        return None
    finally:
        devolver_conexion(conn)


def _guardar_detalle_recomendaciones(
    conversacion_id: int,
    usuario: str,
    recomendaciones: List[Dict],
) -> None:
    """
    Inserta una fila en recomendaciones_detalle por cada curso recomendado.
    Guarda scores individuales y la justificación del LLM.
    """
    if not conversacion_id or not recomendaciones:
        return
    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        for i, rec in enumerate(recomendaciones, 1):
            scores = rec.get("scores_detalle", {})
            cur.execute("""
                INSERT INTO recomendaciones_detalle (
                    conversacion_id, usuario, posicion,
                    id_oferta_formativa, curso, horas_programa, publico_objetivo,
                    tasa_culminacion, tasa_aprobacion, calificacion_prom,
                    score_final, score_colaborativo, score_popularidad,
                    score_historial, score_novedad, algoritmos_usados,
                    justificacion
                ) VALUES (
                    %s,%s,%s, %s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,%s,%s, %s
                )
            """, (
                conversacion_id,
                normalizar_texto(usuario),
                i,
                rec.get("ID_OFERTA_FORMATIVA"),
                normalizar_texto(rec.get("CURSO", "")),
                rec.get("HORAS_PROGRAMA"),
                normalizar_texto(rec.get("PUBLICO_OBJETIVO", "")),
                rec.get("TASA_CULMINACION"),
                rec.get("TASA_APROBACION"),
                rec.get("CALIFICACION_PROM"),
                rec.get("score_final"),
                scores.get("colaborativo"),
                scores.get("popularidad"),
                scores.get("historial"),
                scores.get("novedad"),
                rec.get("metodos_usados", []),
                normalizar_texto(rec.get("justificacion", "")),
            ))
        conn.commit()
        logger.info(f"✅ Detalle guardado: {len(recomendaciones)} cursos para conversacion {conversacion_id}")
    except Exception as e:
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        logger.warning(f"No se pudo guardar detalle de recomendaciones: {e}")
    finally:
        devolver_conexion(conn)


def guardar_conversacion_sifods(usuario: str, nombre: str, mensaje: str, resultado: Dict) -> None:
    _guardar(
        usuario, nombre,
        mensaje, resultado["respuesta"],
        resultado["tarea"], resultado.get("fuente_datos"),
        resultado.get("tokens_entrada"), resultado.get("tokens_salida"),
        resultado.get("latencia_ms"),
    )


def guardar_conversacion_recomendacion(request: RecomendacionRequest, resultado: Dict) -> None:
    top_k = request.top_k or settings.recomendacion.top_k
    conversacion_id = _guardar(
        request.usuario, request.nombre_usuario,
        f"Recomendación (top {top_k})", resultado["respuesta"],
        "recomendacion", resultado.get("fuente_datos"),
        resultado.get("tokens_entrada"), resultado.get("tokens_salida"),
        resultado.get("latencia_ms"),
    )
    _guardar_detalle_recomendaciones(
        conversacion_id  = conversacion_id,
        usuario          = request.usuario,
        recomendaciones  = resultado.get("recomendaciones", []),
    )



# ══════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ══════════════════════════════════════════════════════════════════════
def verificar_schema() -> bool:
    """
    Verifica que las tablas operacionales del agente existan en la BD.
    NO crea ni modifica nada — solo informa.

    Las tablas de negocio (inscripciones, cursos, docentes) son externas
    y se configuran en .env — no son responsabilidad de este check.

    Retorna True si todo está OK, False si falta algo.
    """
    TABLAS_AGENTE  = ["conversaciones_agente", "recomendaciones_detalle"]
    VISTAS_AGENTE  = ["v_metricas_diarias", "v_historial_recomendaciones"]

    conn = None
    try:
        conn = get_db_connection()
        cur  = conn.cursor()

        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = %s",
            (settings.db.schema,)
        )
        tablas_existentes = {row[0] for row in cur.fetchall()}

        cur.execute(
            "SELECT viewname FROM pg_views WHERE schemaname = %s",
            (settings.db.schema,)
        )
        vistas_existentes = {row[0] for row in cur.fetchall()}

        tablas_faltantes = [t for t in TABLAS_AGENTE if t not in tablas_existentes]
        vistas_faltantes = [v for v in VISTAS_AGENTE  if v not in vistas_existentes]
        todo_faltante    = tablas_faltantes + vistas_faltantes

        if not todo_faltante:
            logger.info("✅ BD verificada — tablas operacionales OK")
            return True

        logger.error(
            f"❌ Faltan objetos en la BD: {todo_faltante}\n"
            f"   Ejecuta el schema antes de iniciar el agente:\n"
            f"   psql -U {settings.db.user} -d {settings.db.name} "
            f"-f schema_recomendacion.sql"
        )
        return False

    except Exception as e:
        logger.error(f"❌ No se pudo verificar la BD: {e}")
        return False
    finally:
        devolver_conexion(conn)
        
@app.on_event("startup")
async def startup_event():
    global recomendador
    inicializar_pool()
    logger.info("🔄 Inicializando recomendador...")
    recomendador = inicializar_recomendador()
    logger.info(
        f"🚀 {settings.agente.nombre} v{settings.agente.version} | "
        f"Puerto: {settings.servidor.port} | "
        f"Recomendador: {'✅ activo' if recomendador else '⚠️  no disponible'} | "
        f"Gemini: {'✅' if gemini_model else '⚠️  no configurado'} | "
        f"top_k default: {settings.recomendacion.top_k}"
    )
    asyncio.create_task(_auto_refrescar_recomendador())


async def _auto_refrescar_recomendador() -> None:
    """
    Tarea background que recarga el recomendador automáticamente cada
    CACHE_TTL_SEGUNDOS (configurable en .env, default 3600 = 1 hora).

    Flujo seguro:
      1. Espera el TTL sin bloquear requests
      2. Construye el nuevo modelo en RAM (proceso pesado en thread pool)
      3. Solo hace swap atómico si la carga fue exitosa
      4. Los requests en curso usan el modelo viejo hasta el swap
    """
    ttl = settings.servidor.cache_ttl_segundos
    logger.info(f"⏰ Auto-refresco del recomendador cada {ttl}s ({ttl//3600}h {(ttl%3600)//60}m)")

    while True:
        await asyncio.sleep(ttl)
        logger.info("🔄 Auto-refresco: recargando datos desde PostgreSQL...")
        try:
            loop    = asyncio.get_event_loop()
            nuevo   = await loop.run_in_executor(None, inicializar_recomendador)
            if nuevo:
                global recomendador
                recomendador = nuevo         
                logger.info(
                    f"✅ Auto-refresco completado | "
                    f"Registros: {len(nuevo.df_raw):,} | "
                    f"Próximo refresco en {ttl}s"
                )
            else:
                logger.warning("⚠️  Auto-refresco falló — se mantiene el modelo anterior")
        except Exception as e:
            logger.error(f"❌ Error en auto-refresco: {e} — se mantiene el modelo anterior")


@app.on_event("shutdown")
async def shutdown_event():
    if _db_pool:
        _db_pool.closeall()
        logger.info("🔒 Connection pool cerrado")

# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

@app.get("/agente_tecnologico", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "agente_tecnologico.html",
        {"request": request, "agente": settings.agente.__dict__},
    )


@app.get("/health")
async def health_check():
    return {
        "status":              "healthy",
        **settings.resumen(),
        "recomendador_activo": recomendador is not None,
        "fuente_datos":        getattr(recomendador, "_fuente_datos", "sin_datos"),
        "db_pool_activo":      _db_pool is not None,
        "timestamp":           datetime.now(LIMA_TZ).isoformat(),
    }


@app.post("/api/sifods")
async def consulta_sifods(request:SifodsRequest):
    """
    Módulo RAG — responde preguntas sobre la plataforma SIFODS.
    Busca contexto en Qdrant y genera respuesta con el LLM.
    """
    try:
        resultado = await procesar_consulta_sifods(request.mensaje, request.usuario)
        guardar_conversacion_sifods(request.usuario, request.nombre_usuario, request.mensaje, resultado)
        return {
            "respuesta":    resultado["respuesta"],
            "fuente_datos": resultado.get("fuente_datos"),
            "referencias":  resultado.get("referencias", []),
            "metadata": {
                "tokens_entrada": resultado.get("tokens_entrada"),
                "tokens_salida":  resultado.get("tokens_salida"),
                "latencia_ms":    resultado.get("latencia_ms"),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /api/sifods: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recomendar")
async def recomendar_cursos(request: RecomendacionRequest):
    logger.info(
        f"📨 /api/recomendar → usuario: {request.usuario} | "
        f"top_k: {request.top_k or settings.recomendacion.top_k}"
    )
    try:
        resultado = await procesar_recomendacion_cursos(request)
        guardar_conversacion_recomendacion(request, resultado)
        return {
            "recomendaciones":      resultado["recomendaciones"],
            "respuesta_formateada": resultado["respuesta"],
            "metadata": {
                "latencia_ms":  resultado.get("latencia_ms"),
                "algoritmos":   resultado.get("metadata", {}).get("algoritmos", []),
                "total":        resultado.get("metadata", {}).get("total", 0),
                "fuente_datos": resultado.get("fuente_datos"),
                "top_k_usado":  len(resultado.get("recomendaciones", [])),
            },
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
        recomendador = nuevo_rec
        return {
            "status":    "ok",
            "registros": len(df),
            "fuente":    fuente,
            "top_k":     settings.recomendacion.top_k,
            "timestamp": datetime.now(LIMA_TZ).isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
async def ver_config():
    """
    Muestra la configuración activa (sin secrets).
    Útil para depuración en desarrollo.
    """
    return {
        **settings.resumen(),
        "sifods": {
            "max_tokens":  settings.sifods.max_tokens,
            "temperature": settings.sifods.temperature,
        },
        "justificacion": {
            "max_tokens":  settings.justificacion.max_tokens,
            "temperature": settings.justificacion.temperature,
        },
        "db": {
            "host":   settings.db.host,
            "name":   settings.db.name,
            "schema": settings.db.schema,
            "tabla":  settings.db.tabla_recomendacion,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# SERVIDOR
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host   = "0.0.0.0",
        port   = settings.servidor.port,
        reload = False,
    )
