"""
CONFIGURACIÓN DEL AGENTE TECNOLÓGICO DIFODS
====================================================
Cómo funciona:
  1. Lee las variables desde .env
  2. Expone objetos tipados para cada subsistema
  3. Valida que las variables críticas estén presentes al importar

Uso:
    from config.settings import settings
    settings.llm.modelo_principal   → "gpt-4o-mini"
    settings.recomendacion.top_k    → 3
    settings.db.host                → "localhost"
"""

import os
import logging
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"⚠️  {key} no es entero válido → usando default {default}")
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"⚠️  {key} no es float válido → usando default {default}")
        return default

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).strip().lower()
    return val in ("1", "true", "yes", "on")


# ══════════════════════════════════════════════════════════════════════
# SECCIONES DE CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

@dataclass
class AgenteConfig:
    nombre:    str = "Agente Tecnológico"
    id_agente: str = "tecnologico"
    emoji:     str = "🔧"
    descripcion: str = (
        "Asistente especializado en navegación de la plataforma SIFODS "
        "y recomendación de cursos"
    )
    version: str = "1.0.0"


@dataclass
class LLMConfig:
    openai_api_key:   str = field(default_factory=lambda: _env_str("OPENAI_API_KEY"))
    gemini_api_key:   str = field(default_factory=lambda: _env_str("GEMINI_API_KEY"))
    modelo_principal: str = field(default_factory=lambda: _env_str("LLM_PRINCIPAL", "gpt-4o-mini"))
    modelo_fallback:  str = field(default_factory=lambda: _env_str("LLM_FALLBACK",  "gemini-2.5-flash"))
    embedding_model:  str = field(default_factory=lambda: _env_str(
        "EMBEDDING_MODEL",
        "BAAI/bge-base-en-v1.5"
    ))

    @property
    def tiene_openai(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def tiene_gemini(self) -> bool:
        return bool(self.gemini_api_key)


@dataclass
class QdrantConfig:
    url:        str = field(default_factory=lambda: _env_str("QDRANT_URL", "http://localhost:6333"))
    api_key:    str = field(default_factory=lambda: _env_str("QDRANT_API_KEY"))
    coleccion:  str = field(default_factory=lambda: _env_str("QDRANT_COLLECTION", "Curso_0"))
    top_k:      int = field(default_factory=lambda: _env_int("QDRANT_TOP_K", 10))


@dataclass
class DBConfig:
    host:     str  = field(default_factory=lambda: _env_str("DB_HOST", "localhost"))
    port:     int  = field(default_factory=lambda: _env_int("DB_PORT", 5432))
    name:     str  = field(default_factory=lambda: _env_str("DB_NAME", "agente_tecnologico"))
    user:     str  = field(default_factory=lambda: _env_str("DB_USER", "postgres"))
    password: str  = field(default_factory=lambda: _env_str("DB_PASSWORD"))
    schema:   str  = field(default_factory=lambda: _env_str("DB_SCHEMA", "public"))
    pool_min: int  = field(default_factory=lambda: _env_int("DB_POOL_MIN", 2))
    pool_max: int  = field(default_factory=lambda: _env_int("DB_POOL_MAX", 10))
    tabla_recomendacion: str = field(
        default_factory=lambda: _env_str("DB_TABLE_RECOMENDACION", "v_inscripciones_recomendacion")
    )
    excel_fallback_path: str = field(
        default_factory=lambda: _env_str("EXCEL_FALLBACK_PATH", "data/DATA_RECOM.xlsx")
    )
    excel_sheet_name: str = field(
        default_factory=lambda: _env_str("EXCEL_SHEET_NAME", "0")
    )

    def as_dict(self) -> dict:
        """Kwargs para psycopg2.connect / ThreadedConnectionPool."""
        return dict(
            host=self.host,
            port=self.port,
            database=self.name,
            user=self.user,
            password=self.password,
            client_encoding="UTF8",
        )


@dataclass
class RecomendacionConfig:
    # El parámetro top_k tiene prioridad cuando el cliente lo envía explícitamente
    top_k:               int   = field(default_factory=lambda: _env_int("REC_TOP_K", 3))
    min_score:           float = field(default_factory=lambda: _env_float("REC_MIN_SCORE", 0.05))
    top_k_similar_users: int   = field(default_factory=lambda: _env_int("REC_TOP_K_SIMILAR_USERS", 20))
    max_usuarios_matriz: int   = field(default_factory=lambda: _env_int("REC_MAX_USUARIOS_MATRIZ", 15_000))
    max_llm_workers:     int   = field(default_factory=lambda: _env_int("REC_MAX_LLM_WORKERS", 3))

    # Pesos del modelo híbrido
    peso_colaborativo: float = field(default_factory=lambda: _env_float("REC_PESO_COLABORATIVO", 0.40))
    peso_popularidad:  float = field(default_factory=lambda: _env_float("REC_PESO_POPULARIDAD",  0.30))
    peso_historial:    float = field(default_factory=lambda: _env_float("REC_PESO_HISTORIAL",     0.20))
    peso_novedad:      float = field(default_factory=lambda: _env_float("REC_PESO_NOVEDAD",       0.10))

    @property
    def pesos(self) -> dict:
        return {
            "colaborativo": self.peso_colaborativo,
            "popularidad":  self.peso_popularidad,
            "historial":    self.peso_historial,
            "novedad":      self.peso_novedad,
        }

    def validar_pesos(self) -> None:
        total = sum(self.pesos.values())
        if not (0.99 < total < 1.01):
            raise ValueError(
                f"❌ Los pesos del recomendador suman {total:.2f}, deben sumar 1.0. "
                f"Revisa REC_PESO_* en el .env"
            )


@dataclass
class SIFODSConfig:
    """Parámetros del módulo RAG para consultas de plataforma."""
    max_tokens:  int   = field(default_factory=lambda: _env_int("SIFODS_MAX_TOKENS", 1500))
    temperature: float = field(default_factory=lambda: _env_float("SIFODS_TEMPERATURE", 0.45))
    top_p:       float = field(default_factory=lambda: _env_float("SIFODS_TOP_P", 0.9))

    fuentes_datos: list = field(default_factory=lambda: [
        "DOCENTE AL DÍA",
        "CENTRO DE RECURSOS",
        "ASISTENCIA VIRTUAL DOCENTE",
        "CANAL DE YOUTUBE",
        "PREGUNTAS FRECUENTES",
    ])

    keywords_deteccion: list = field(default_factory=lambda: [
        "cómo", "dónde", "acceder", "entrar", "iniciar sesión",
        "no puedo", "error", "no carga", "no funciona",
        "tutorial", "ayuda", "guía", "manual",
        "plataforma", "sifods", "recursos", "youtube",
    ])

    @property
    def parametros_modelo(self) -> dict:
        return {
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "top_p":       self.top_p,
        }


@dataclass
class JustificacionLLMConfig:
    """Parámetros del LLM al generar justificaciones de cursos."""
    max_tokens:  int   = field(default_factory=lambda: _env_int("JUSTIF_MAX_TOKENS", 250))
    temperature: float = field(default_factory=lambda: _env_float("JUSTIF_TEMPERATURE", 0.75))


@dataclass
class ServidorConfig:
    port:      int  = field(default_factory=lambda: _env_int("PORT", 7002))
    log_level: str  = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))
    guardar_conversaciones: bool = field(
        default_factory=lambda: _env_bool("GUARDAR_CONVERSACIONES", True)
    )
    guardar_metricas: bool = field(
        default_factory=lambda: _env_bool("GUARDAR_METRICAS", True)
    )
    cache_ttl_segundos: int = field(
        default_factory=lambda: _env_int("CACHE_TTL_SEGUNDOS", 3600)
    )


# ══════════════════════════════════════════════════════════════════════
# OBJETO GLOBAL
# ══════════════════════════════════════════════════════════════════════

class Settings:
    """
    Punto de acceso único a toda la configuración.
    Instanciar una sola vez al importar el módulo.

    Uso:
        from config.settings import settings
        settings.llm.modelo_principal
    """
    def __init__(self):
        self.agente        = AgenteConfig()
        self.llm           = LLMConfig()
        self.qdrant        = QdrantConfig()
        self.db            = DBConfig()
        self.recomendacion = RecomendacionConfig()
        self.sifods        = SIFODSConfig()
        self.justificacion = JustificacionLLMConfig()
        self.servidor      = ServidorConfig()
        self._validar()

    def _validar(self) -> None:
        """Advertencias al arrancar si faltan claves críticas."""
        if not self.llm.tiene_openai:
            logger.warning("⚠️  OPENAI_API_KEY no definida → se usará solo Gemini como LLM")
        if not self.llm.tiene_gemini:
            logger.warning("⚠️  GEMINI_API_KEY no definida → sin fallback LLM")
        if not self.llm.tiene_openai and not self.llm.tiene_gemini:
            logger.error("❌ No hay ningún LLM configurado. El agente no podrá responder.")
        try:
            self.recomendacion.validar_pesos()
        except ValueError as e:
            logger.error(str(e))

    def resumen(self) -> dict:
        """Resumen legible para el endpoint /health."""
        return {
            "agente":          self.agente.nombre,
            "version":         self.agente.version,
            "llm_principal":   self.llm.modelo_principal,
            "llm_fallback":    self.llm.modelo_fallback,
            "openai":          self.llm.tiene_openai,
            "gemini":          self.llm.tiene_gemini,
            "qdrant_url":      self.qdrant.url,
            "qdrant_coleccion":self.qdrant.coleccion,
            "db_host":         self.db.host,
            "db_tabla_rec":    self.db.tabla_recomendacion,
            "rec_top_k":       self.recomendacion.top_k,
            "rec_pesos":       self.recomendacion.pesos,
            "puerto":          self.servidor.port,
        }

settings = Settings()
