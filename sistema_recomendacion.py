"""
🎯 SISTEMA DE RECOMENDACIÓN DE CURSOS - FILTRO COLABORATIVO HÍBRIDO
====================================================================
DIFODS - Agente Tecnológico v2.2

Tabla única en PostgreSQL (vista o tabla plana) con los campos reales:

    AÑO, TIPO_CONSTANCIA, CURSO, NOMBRE_DRE, NOMBRE_UGEL,
    USUARIO_DOCUMENTO, NOMBRE_COMPLETO, NIVELNEXUS,
    APROBACION, ID_OFERTA_FORMATIVA, ID_GRUPO,
    FECHA_NACIMIENTO, ES_FOCALIZADO, HORAS_PROGRAMA,
    CALIFICACIONES,   ← rating del CURSO (0-5), NO del docente
    PROPOSITO, ACTIVO, PUBLICO_OBJETIVO, CURSO_CULMINADO, EDAD

Componentes del modelo híbrido:
    1. Colaborativo  (40%) → producto matricial numpy (NO doble loop)
    2. Popularidad   (30%) → tasa_culminacion × calificacion_norm × log(inscritos)
    3. Historial     (20%) → co-ocurrencia con peso PUBLICO_OBJETIVO
    4. Novedad       (10%) → score por AÑO más reciente

CORRECCIONES v2.2:
    - RATING_COMPUESTO vectorizado con np.where (sin df.apply)
    - Colaborativo con producto matricial numpy (O(U*C) en vez de O(C*V))
    - Justificaciones concurrentes con ThreadPoolExecutor
    - Similitud: sparse matrix + limitación a 10k docentes para RAM segura
    - refrescar_datos() crea instancia nueva (no corrompe estado)
    - _normalizar_binario maneja numpy.bool_ correctamente
    - gemini_client sin acceso a atributo privado ._api_key
    - CALIFICACIONES escala 0-5 documentada y guardada para refrescar
"""

import os
import logging
import concurrent.futures
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from openai import OpenAI
from google import genai

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# ⚙️ CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════

class RecommenderConfig:
    PESOS = {
        "colaborativo": 0.40,
        "popularidad":  0.30,
        "historial":    0.20,
        "novedad":      0.10,
    }

    TOP_K_SIMILAR_USERS   = 20
    TOP_K_RECOMENDACIONES = 3
    MIN_SCORE             = 0.05

    # Límite de docentes para la matriz de similitud (protección RAM)
    # Para 10k docentes × 10k docentes × float32 = ~400MB (seguro)
    # Para 50k docentes la matrix sería ~10GB → usar sparse en su lugar
    MAX_USUARIOS_MATRIZ   = 15_000

    # Justificaciones concurrentes (máx threads simultáneos al LLM)
    MAX_LLM_WORKERS       = 3

    COLUMNAS_REQUERIDAS = {
        "USUARIO_DOCUMENTO", "ID_OFERTA_FORMATIVA", "CURSO",
        "APROBACION", "ACTIVO", "NIVELNEXUS", "AÑO"
    }

    COLUMNAS_OPCIONALES = {
        "PORCENTAJE_AVANCE":  0.0,
        "CALIFICACIONES":     np.nan,   # rating del curso (0-5)
        "CURSO_CULMINADO":    0,
        "NOMBRE_DRE":         "",
        "NOMBRE_UGEL":        "",
        "HORAS_PROGRAMA":     0,
        "PROPOSITO":          "",
        "NOMBRE_COMPLETO":    "",
        "PUBLICO_OBJETIVO":   "",
        "ES_FOCALIZADO":      0,
        "ID_GRUPO":           "",
        "EDAD":               np.nan,
    }


# ══════════════════════════════════════════════════════════════════════
# 🔧 CLASE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class HybridRecommender:
    """
    Sistema de recomendación híbrido para cursos SIFODS.
    Optimizado para gran escala: operaciones vectorizadas, sparse matrices,
    LLM concurrente.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        openai_api_key: str,
        gemini_api_key: str = None
    ):
        # ── Clientes LLM ──────────────────────────────────────────────
        self.openai_client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)

        # Guardar keys (NO usar atributos privados de la SDK)
        self._openai_key = openai_api_key
        self._gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")

        self.gemini_client = None
        if self._gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=self._gemini_key)
                logger.info("✅ Gemini configurado como fallback LLM")
            except Exception as e:
                logger.warning(f"⚠️  Gemini no disponible: {e}")

        # ── Preparar DataFrame ────────────────────────────────────────
        df = df.copy()
        df.columns = [c.strip().upper() for c in df.columns]

        missing = RecommenderConfig.COLUMNAS_REQUERIDAS - set(df.columns)
        if missing:
            raise ValueError(f"❌ Faltan columnas requeridas: {missing}")

        for col, default in RecommenderConfig.COLUMNAS_OPCIONALES.items():
            if col not in df.columns:
                logger.warning(f"⚠️  '{col}' no encontrada → default: {default!r}")
                df[col] = default

        self.df_raw = df

        # ── Pipeline de construcción ──────────────────────────────────
        self._preprocesar()
        self.df_cursos    = self._construir_catalogo_cursos()
        self.df_perfil    = self._construir_perfil_docentes()
        self.matriz_uc, self.user_index, self.curso_index = self._construir_matriz_usuario_curso()
        self.sim_matrix   = self._calcular_similitud_usuarios()

        logger.info(
            f"✅ HybridRecommender v2.2 | "
            f"Registros: {len(self.df_raw):,} | "
            f"Cursos activos: {(self.df_cursos['ACTIVO']==1).sum()} | "
            f"Docentes: {len(self.df_perfil):,}"
        )

    # ══════════════════════════════════════════════════════════════════
    # 📐 PRE-PROCESAMIENTO (VECTORIZADO)
    # ══════════════════════════════════════════════════════════════════

    def _preprocesar(self):
        df = self.df_raw

        # Numéricos
        df["APROBACION"]        = pd.to_numeric(df["APROBACION"],        errors="coerce").fillna(0).astype(int)
        df["ACTIVO"]            = pd.to_numeric(df["ACTIVO"],            errors="coerce").fillna(0).astype(int)
        df["PORCENTAJE_AVANCE"] = pd.to_numeric(df["PORCENTAJE_AVANCE"], errors="coerce").fillna(0.0).clip(0, 1)
        df["HORAS_PROGRAMA"]    = pd.to_numeric(df["HORAS_PROGRAMA"],    errors="coerce").fillna(0)
        df["AÑO"]               = pd.to_numeric(df["AÑO"],               errors="coerce").fillna(0).astype(int)
        df["EDAD"]              = pd.to_numeric(df["EDAD"],               errors="coerce")

        # CALIFICACIONES: rating del CURSO (0-5)
        df["CALIFICACIONES"] = pd.to_numeric(df["CALIFICACIONES"], errors="coerce").clip(0, 5)

        # CURSO_CULMINADO vectorizado (maneja str, int, numpy bool_)
        df["CURSO_CULMINADO"] = self._normalizar_binario_series(df["CURSO_CULMINADO"])

        # IDs como string limpio
        df["USUARIO_DOCUMENTO"]   = df["USUARIO_DOCUMENTO"].astype(str).str.strip()
        df["ID_OFERTA_FORMATIVA"] = df["ID_OFERTA_FORMATIVA"].astype(str).str.strip()

        # ── RATING_COMPUESTO vectorizado (sin df.apply) ───────────────
        #
        #  +2.5  si CURSO_CULMINADO == 1  (señal más fuerte)
        #  +1.5  si APROBACION == 1 Y CURSO_CULMINADO == 0
        #  +1.0  si CURSO_CULMINADO==0 Y PORCENTAJE_AVANCE >= 0.8
        #  +0.5  si CURSO_CULMINADO==0 Y 0.5 <= PORCENTAJE_AVANCE < 0.8
        #
        culmino  = df["CURSO_CULMINADO"] == 1
        aprobo   = df["APROBACION"]       == 1
        sin_cul  = ~culmino
        avance   = df["PORCENTAJE_AVANCE"]

        rating = (
            culmino.astype(float) * 2.5
            + (sin_cul & aprobo).astype(float) * 1.5
            + (sin_cul & (avance >= 0.8)).astype(float) * 1.0
            + (sin_cul & (avance >= 0.5) & (avance < 0.8)).astype(float) * 0.5
        ).clip(0, 5).round(4)

        df["RATING_COMPUESTO"] = rating
        self.df_raw = df

    @staticmethod
    def _normalizar_binario_series(serie: pd.Series) -> pd.Series:
        """
        Vectorizado. Convierte SI/NO/1/0/True/False/numpy.bool_ → int 0 ó 1.
        Más rápido que apply() fila por fila.
        """
        # Intentar conversión numérica directa (maneja int, float, numpy.bool_)
        numerico = pd.to_numeric(serie, errors="coerce")
        resultado = (numerico > 0).astype(int)

        # Para los que no son numéricos (strings "SI", "NO", etc.)
        es_string = numerico.isna() & serie.notna()
        if es_string.any():
            positivos = serie[es_string].astype(str).str.strip().str.upper().isin(
                {"SI", "SÍ", "S", "TRUE", "YES", "Y", "1"}
            )
            resultado[es_string] = positivos.astype(int)

        return resultado.fillna(0).astype(int)

    # ══════════════════════════════════════════════════════════════════
    # 📚 CATÁLOGO DE CURSOS
    # ══════════════════════════════════════════════════════════════════

    def _construir_catalogo_cursos(self) -> pd.DataFrame:
        """
        Una fila por ID_OFERTA_FORMATIVA.
        CALIFICACIONES (0-5) → CALIFICACION_PROM → entrada al score de popularidad.
        """
        df = self.df_raw

        agg = df.groupby("ID_OFERTA_FORMATIVA").agg(
            CURSO             = ("CURSO",             "first"),
            HORAS_PROGRAMA    = ("HORAS_PROGRAMA",    "first"),
            PROPOSITO         = ("PROPOSITO",          "first"),
            PUBLICO_OBJETIVO  = ("PUBLICO_OBJETIVO",   "first"),
            AÑO_MAX           = ("AÑO",                "max"),
            ACTIVO            = ("ACTIVO",             "max"),
            TOTAL_INSCRITOS   = ("USUARIO_DOCUMENTO",  "count"),
            TOTAL_CULMINADOS  = ("CURSO_CULMINADO",    "sum"),
            TOTAL_APROBADOS   = ("APROBACION",         "sum"),
            CALIFICACION_PROM = ("CALIFICACIONES",     "mean"),  # 0-5, promedio del curso
        ).reset_index()

        n = agg["TOTAL_INSCRITOS"].replace(0, np.nan)

        agg["TASA_CULMINACION"]  = (agg["TOTAL_CULMINADOS"] / n).fillna(0).clip(0, 1)
        agg["TASA_APROBACION"]   = (agg["TOTAL_APROBADOS"]  / n).fillna(0).clip(0, 1)
        agg["CALIFICACION_NORM"] = (agg["CALIFICACION_PROM"].fillna(0) / 5.0).clip(0, 1)

        # Score popularidad: cursos que mucha gente TERMINA y están bien CALIFICADOS
        agg["SCORE_POPULARIDAD"] = (
            agg["TASA_CULMINACION"] *
            agg["CALIFICACION_NORM"] *
            np.log1p(agg["TOTAL_INSCRITOS"])
        ).round(4)

        año_max_global = int(agg["AÑO_MAX"].max())
        agg["SCORE_NOVEDAD"] = (año_max_global - agg["AÑO_MAX"]).map(
            lambda d: 1.0 if d == 0 else (0.6 if d == 1 else (0.3 if d == 2 else 0.1))
        )

        return agg

    # ══════════════════════════════════════════════════════════════════
    # 👤 PERFIL DE DOCENTES
    # ══════════════════════════════════════════════════════════════════

    def _construir_perfil_docentes(self) -> pd.DataFrame:
        df = self.df_raw

        perfil = df.groupby("USUARIO_DOCUMENTO").agg(
            NOMBRE_COMPLETO   = ("NOMBRE_COMPLETO",     "first"),
            NIVELNEXUS        = ("NIVELNEXUS",           "first"),
            NOMBRE_DRE        = ("NOMBRE_DRE",           "first"),
            NOMBRE_UGEL       = ("NOMBRE_UGEL",          "first"),
            EDAD_APROX        = ("EDAD",                 "first"),
            TOTAL_CURSOS      = ("ID_OFERTA_FORMATIVA",  "count"),
            CURSOS_CULMINADOS = ("CURSO_CULMINADO",      "sum"),
            CURSOS_APROBADOS  = ("APROBACION",           "sum"),
        ).reset_index()

        perfil["TASA_COMPLETITUD"] = (
            perfil["CURSOS_CULMINADOS"] / perfil["TOTAL_CURSOS"].replace(0, np.nan)
        ).fillna(0).round(4)

        return perfil

    # ══════════════════════════════════════════════════════════════════
    # 📊 MATRIZ USUARIO-CURSO (SPARSE) Y SIMILITUD
    # ══════════════════════════════════════════════════════════════════

    def _construir_matriz_usuario_curso(self) -> Tuple[csr_matrix, pd.Index, pd.Index]:
        """
        Construye matriz sparse usuario × curso.
        Retorna (matriz_sparse, user_index, curso_index).

        Para grandes volúmenes usa sparse matrix en lugar de dense DataFrame
        para ahorrar RAM significativamente.
        """
        pivot = self.df_raw.pivot_table(
            index="USUARIO_DOCUMENTO",
            columns="ID_OFERTA_FORMATIVA",
            values="RATING_COMPUESTO",
            aggfunc="max",
            fill_value=0.0
        )

        user_index  = pivot.index
        curso_index = pivot.columns

        # Convertir a sparse float32 (ahorra ~50% RAM vs float64 dense)
        sparse = csr_matrix(pivot.values.astype(np.float32))

        logger.info(
            f"📊 Matriz usuario-curso: {sparse.shape} | "
            f"Densidad: {sparse.nnz / (sparse.shape[0]*sparse.shape[1]):.2%} | "
            f"RAM estimada: {sparse.data.nbytes / 1e6:.1f} MB"
        )
        return sparse, user_index, curso_index

    def _calcular_similitud_usuarios(self) -> np.ndarray:
        """
        Calcula similitud coseno entre usuarios.

        Si hay más de MAX_USUARIOS_MATRIZ usuarios, usa solo los más activos
        para evitar OOM. Los usuarios fuera del top son manejados por el
        componente de historial (que no requiere similitud global).
        """
        n_usuarios = self.matriz_uc.shape[0]
        max_u = RecommenderConfig.MAX_USUARIOS_MATRIZ

        if n_usuarios > max_u:
            logger.warning(
                f"⚠️  {n_usuarios:,} docentes superan el límite ({max_u:,}) "
                f"para la matriz de similitud. "
                f"Usando los {max_u:,} más activos. "
                f"El resto usa componente historial."
            )
            # Seleccionar los más activos (más cursos tomados)
            actividad = np.array(self.matriz_uc.sum(axis=1)).flatten()
            top_idx   = np.argsort(actividad)[::-1][:max_u]
            sub_matrix = self.matriz_uc[top_idx]
            self._sim_user_index = self.user_index[top_idx]
        else:
            sub_matrix = self.matriz_uc
            self._sim_user_index = self.user_index

        sim = cosine_similarity(sub_matrix, dense_output=True).astype(np.float32)
        logger.info(f"📐 Matriz similitud: {sim.shape} | RAM: {sim.nbytes / 1e6:.1f} MB")
        return sim

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 1: COLABORATIVO (40%) — MATRICIAL
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_colaborativo(
        self, user_id: str, cursos_excluir: List[str], top_k: int
    ) -> Dict[str, float]:
        """
        Producto matricial numpy: sim_vector @ rating_matrix.
        O(U) en lugar de O(C × V) del doble loop anterior.
        """
        sim_idx_series = pd.Index(self._sim_user_index)

        if user_id not in sim_idx_series:
            # Usuario fuera de la submatriz (poco activo) o sin historial
            logger.warning(f"⚠️  {user_id} no está en submatriz similitud → colaborativo omitido")
            return {}

        user_pos = sim_idx_series.get_loc(user_id)
        sim_vec  = self.sim_matrix[user_pos].copy()  # shape (N_sim,)
        sim_vec[user_pos] = 0.0  # excluir el propio usuario

        # Top K vecinos
        top_vecinos = np.argsort(sim_vec)[::-1][:RecommenderConfig.TOP_K_SIMILAR_USERS]
        sim_top     = sim_vec[top_vecinos]  # (K,)

        # Submatriz de vecinos en el espacio global de cursos
        # Necesitamos mapear _sim_user_index → user_index (puede ser subconjunto)
        global_idx = [
            self.user_index.get_loc(self._sim_user_index[i])
            for i in top_vecinos
            if self._sim_user_index[i] in self.user_index
        ]
        valid_sim = sim_top[:len(global_idx)]

        if not global_idx:
            return {}

        # sub_matrix: (K, C) sparse → dense para operación vectorial
        vecinos_matrix = self.matriz_uc[global_idx].toarray()  # (K, C)

        # Predicción ponderada: (K,) @ (K, C) = (C,)
        denom = valid_sim.sum()
        if denom == 0:
            return {}

        predicted = (valid_sim @ vecinos_matrix) / denom  # (C,)

        # Construir dict curso_id → score
        excluir_set = set(cursos_excluir)
        scores = {}
        for j, curso_id in enumerate(self.curso_index):
            if curso_id in excluir_set:
                continue
            v = float(predicted[j])
            if v > 0:
                scores[str(curso_id)] = v

        # Normalizar 0-1
        if scores:
            max_s = max(scores.values())
            if max_s > 0:
                scores = {k: round(v / max_s, 4) for k, v in scores.items()}

        return scores

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 2: POPULARIDAD (30%)
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_popularidad(self, cursos_excluir: List[str]) -> Dict[str, float]:
        df = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ]

        if df.empty:
            return {}

        max_s = df["SCORE_POPULARIDAD"].max()
        if max_s == 0:
            return {}

        return dict(zip(
            df["ID_OFERTA_FORMATIVA"],
            (df["SCORE_POPULARIDAD"] / max_s).round(4)
        ))

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 3: HISTORIAL POR SIMILITUD (20%)
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_historial(
        self, user_id: str, cursos_excluir: List[str], top_k: int
    ) -> Dict[str, float]:
        """
        Cursos culminados por docentes del mismo NIVELNEXUS + DRE.
        Boost si PUBLICO_OBJETIVO menciona el nivel del docente.
        """
        perfil_row = self.df_perfil[self.df_perfil["USUARIO_DOCUMENTO"] == user_id]
        if perfil_row.empty:
            return {}

        p     = perfil_row.iloc[0]
        nivel = str(p["NIVELNEXUS"]).upper()
        dre   = str(p["NOMBRE_DRE"])

        mask_nivel = self.df_perfil["NIVELNEXUS"].astype(str).str.upper() == nivel
        mask_dre   = self.df_perfil["NOMBRE_DRE"] == dre
        mask_self  = self.df_perfil["USUARIO_DOCUMENTO"] != user_id

        similares = self.df_perfil[mask_nivel & mask_dre & mask_self]["USUARIO_DOCUMENTO"].tolist()

        if len(similares) < 5:
            similares = self.df_perfil[mask_nivel & mask_self]["USUARIO_DOCUMENTO"].tolist()

        if not similares:
            return {}

        excluir_set = set(cursos_excluir)
        df_sim = self.df_raw[
            self.df_raw["USUARIO_DOCUMENTO"].isin(similares) &
            (self.df_raw["CURSO_CULMINADO"] == 1) &
            (self.df_raw["ACTIVO"] == 1) &
            ~self.df_raw["ID_OFERTA_FORMATIVA"].isin(excluir_set)
        ].copy()

        if df_sim.empty:
            return {}

        # Boost vectorizado según PUBLICO_OBJETIVO
        pub_upper = df_sim["PUBLICO_OBJETIVO"].fillna("").astype(str).str.upper()
        df_sim["PESO"] = np.where(pub_upper.str.contains(nivel, regex=False), 1.5, 0.8)
        # Sin restricción de público → peso neutro 1.0
        sin_publico = df_sim["PUBLICO_OBJETIVO"].fillna("").astype(str).str.strip() == ""
        df_sim.loc[sin_publico, "PESO"] = 1.0

        freq     = df_sim.groupby("ID_OFERTA_FORMATIVA")["PESO"].sum()
        max_freq = freq.max()
        if max_freq == 0:
            return {}

        return (freq / max_freq).round(4).to_dict()

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 4: NOVEDAD (10%)
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_novedad(self, cursos_excluir: List[str]) -> Dict[str, float]:
        df = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ]
        return dict(zip(df["ID_OFERTA_FORMATIVA"], df["SCORE_NOVEDAD"]))

    # ══════════════════════════════════════════════════════════════════
    # 🏆 RECOMENDACIÓN HÍBRIDA FINAL
    # ══════════════════════════════════════════════════════════════════

    def recomendar_hibrido(
        self,
        user_id: str,
        top_k: int = RecommenderConfig.TOP_K_RECOMENDACIONES,
        incluir_justificacion: bool = True
    ) -> List[Dict]:
        logger.info(f"Recomendando para: {user_id}")
        logger.info(f"top_k recibido: {top_k} | config default: {RecommenderConfig.TOP_K_RECOMENDACIONES}")

        cursos_tomados = set(
            self.df_raw[self.df_raw["USUARIO_DOCUMENTO"] == user_id]
            ["ID_OFERTA_FORMATIVA"].unique()
        )

        expand  = top_k * 4
        s_colab = self._recomendar_colaborativo(user_id, list(cursos_tomados), expand)
        s_popul = self._recomendar_popularidad(list(cursos_tomados))
        s_hist  = self._recomendar_historial(user_id, list(cursos_tomados), expand)
        s_novel = self._recomendar_novedad(list(cursos_tomados))

        todos = set(s_colab) | set(s_popul) | set(s_hist) | set(s_novel)

        if not todos:
            logger.warning(f"Sin candidatos para {user_id} → fallback populares")
            return self._fallback_populares(list(cursos_tomados), top_k)

        pesos = RecommenderConfig.PESOS
        resultados = []

        for curso_id in todos:
            sc = s_colab.get(curso_id, 0.0) * pesos["colaborativo"]
            sp = s_popul.get(curso_id, 0.0) * pesos["popularidad"]
            sh = s_hist.get(curso_id, 0.0)  * pesos["historial"]
            sn = s_novel.get(curso_id, 0.0) * pesos["novedad"]
            score_final = sc + sp + sh + sn

            if score_final < RecommenderConfig.MIN_SCORE:
                continue

            metodos = [
                m for m, s in [
                    ("colaborativo", s_colab.get(curso_id, 0)),
                    ("popularidad",  s_popul.get(curso_id, 0)),
                    ("historial",    s_hist.get(curso_id, 0)),
                    ("novedad",      s_novel.get(curso_id, 0)),
                ] if s > 0
            ]

            resultados.append({
                "ID_OFERTA_FORMATIVA": curso_id,
                "score_final":         round(score_final, 4),
                "scores_detalle": {
                    "colaborativo": round(sc, 4),
                    "popularidad":  round(sp, 4),
                    "historial":    round(sh, 4),
                    "novedad":      round(sn, 4),
                },
                "metodos_usados": metodos,
            })

        if not resultados:
            return self._fallback_populares(list(cursos_tomados), top_k)

        resultados.sort(key=lambda x: x["score_final"], reverse=True)
        top_r = self._enriquecer_con_catalogo(resultados[:top_k])

        if incluir_justificacion:
            top_r = self._generar_justificaciones_concurrente(top_r, user_id)

        logger.info(f"✅ {len(top_r)} cursos recomendados para {user_id}")
        return top_r

    # ══════════════════════════════════════════════════════════════════
    # 🛟 FALLBACK
    # ══════════════════════════════════════════════════════════════════

    def _fallback_populares(self, cursos_excluir: List[str], top_k: int) -> List[Dict]:
        logger.info("🔄 Fallback: cursos más populares")
        df = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ].nlargest(top_k, "SCORE_POPULARIDAD")

        return [
            {
                "ID_OFERTA_FORMATIVA": str(row["ID_OFERTA_FORMATIVA"]),
                "CURSO":              str(row["CURSO"]),
                "HORAS_PROGRAMA":     int(row["HORAS_PROGRAMA"]),
                "PROPOSITO":          str(row["PROPOSITO"])[:300],
                "PUBLICO_OBJETIVO":   str(row["PUBLICO_OBJETIVO"]),
                "AÑO":                int(row["AÑO_MAX"]),
                "TOTAL_INSCRITOS":    int(row["TOTAL_INSCRITOS"]),
                "TASA_CULMINACION":   round(float(row["TASA_CULMINACION"]), 2),
                "TASA_APROBACION":    round(float(row["TASA_APROBACION"]), 2),
                "CALIFICACION_PROM":  round(float(row["CALIFICACION_PROM"] or 0), 2),
                "score_final":        round(float(row["SCORE_POPULARIDAD"]), 4),
                "scores_detalle":     {},
                "metodos_usados":     ["popularidad_fallback"],
                "justificacion":      "Curso muy valorado por docentes de la plataforma.",
            }
            for _, row in df.iterrows()
        ]

    # ══════════════════════════════════════════════════════════════════
    # 🗂️ ENRIQUECIMIENTO CON CATÁLOGO
    # ══════════════════════════════════════════════════════════════════

    def _enriquecer_con_catalogo(self, resultados: List[Dict]) -> List[Dict]:
        idx = self.df_cursos.set_index("ID_OFERTA_FORMATIVA")
        for rec in resultados:
            cid = rec["ID_OFERTA_FORMATIVA"]
            if cid in idx.index:
                f = idx.loc[cid]
                rec.update({
                    "CURSO":             str(f["CURSO"]),
                    "HORAS_PROGRAMA":    int(f["HORAS_PROGRAMA"]),
                    "PROPOSITO":         str(f["PROPOSITO"])[:300],
                    "PUBLICO_OBJETIVO":  str(f["PUBLICO_OBJETIVO"]),
                    "AÑO":               int(f["AÑO_MAX"]),
                    "TOTAL_INSCRITOS":   int(f["TOTAL_INSCRITOS"]),
                    "TASA_CULMINACION":  round(float(f["TASA_CULMINACION"]), 2),
                    "TASA_APROBACION":   round(float(f["TASA_APROBACION"]), 2),
                    "CALIFICACION_PROM": round(float(f["CALIFICACION_PROM"] or 0), 2),
                })
            else:
                rec.update({
                    "CURSO": "Curso no encontrado", "HORAS_PROGRAMA": 0,
                    "PROPOSITO": "", "PUBLICO_OBJETIVO": "", "AÑO": 0,
                    "TOTAL_INSCRITOS": 0, "TASA_CULMINACION": 0.0,
                    "TASA_APROBACION": 0.0, "CALIFICACION_PROM": 0.0,
                })
        return resultados

    # ══════════════════════════════════════════════════════════════════
    # 🤖 JUSTIFICACIONES CONCURRENTES (GPT → Gemini fallback)
    # ══════════════════════════════════════════════════════════════════

    def _generar_justificaciones_concurrente(
        self, recomendaciones: List[Dict], user_id: str
    ) -> List[Dict]:
        """
        Llama al LLM en paralelo para todas las justificaciones.
        Reduce latencia de top_k * T_llm → max(T_llm) en la mayoría de casos.
        """
        perfil_row = self.df_perfil[self.df_perfil["USUARIO_DOCUMENTO"] == user_id]

        if perfil_row.empty:
            perfil_txt = "Docente de la plataforma SIFODS"
        else:
            p = perfil_row.iloc[0]
            perfil_txt = (
                f"Docente de nivel {p['NIVELNEXUS']}, "
                f"DRE {p['NOMBRE_DRE']}, UGEL {p['NOMBRE_UGEL']}. "
                f"Ha culminado {int(p['CURSOS_CULMINADOS'])} cursos "
                f"({p['TASA_COMPLETITUD']*100:.0f}% tasa de completitud)."
            )

        def _generar_una(rec: Dict) -> str:
            cal      = rec.get("CALIFICACION_PROM", 0)
            tasa_cul = rec.get("TASA_CULMINACION", 0)
            tasa_apr = rec.get("TASA_APROBACION", 0)
            horas    = rec.get("HORAS_PROGRAMA", 0)
            metodos  = rec.get("metodos_usados", [])
            ranking  = rec.get("_ranking_pos", "")  # posición en el top (ver nota abajo)

            # Traducir los métodos a lenguaje humano
            metodo_textos = {
                "colaborativo": f"docentes con perfil similar al tuyo lo completaron con éxito",
                "historial":    f"docentes de tu nivel y región lo culminaron y valoraron positivamente",
                "popularidad":  f"es uno de los cursos más completados y mejor calificados de la plataforma",
                "novedad":      f"es parte de la oferta formativa más reciente de DIFODS",
            }
            razones = [metodo_textos[m] for m in metodos if m in metodo_textos]
            razones_txt = "; ".join(razones) if razones else "es relevante para tu perfil docente"

            # Datos del docente más ricos
            if perfil_row.empty:
                docente_contexto = "Docente de la plataforma SIFODS"
                nivel = ""
                dre   = ""
            else:
                p     = perfil_row.iloc[0]
                nivel = str(p.get("NIVELNEXUS", "")).strip()
                dre   = str(p.get("NOMBRE_DRE", "")).strip()
                n_cul = int(p.get("CURSOS_CULMINADOS", 0))
                n_tot = int(p.get("TOTAL_CURSOS", 0))
                tasa_doc = float(p.get("TASA_COMPLETITUD", 0)) * 100
                edad  = p.get("EDAD_APROX", None)

                docente_contexto = (
                    f"Nivel educativo: {nivel}. "
                    f"DRE: {dre}. "
                    f"Ha completado {n_cul} de {n_tot} cursos en SIFODS "
                    f"(tasa de completitud personal: {tasa_doc:.0f}%). "
                    + (f"Edad aproximada: {int(edad)} años. " if pd.notna(edad) else "")
                )

            cal_txt = f"{cal:.1f}/5.0" if cal > 0 else "sin calificación registrada aún"

            system_prompt = (
                f"Eres un orientador formativo del Ministerio de Educación del Perú, "
                f"experto en desarrollo profesional docente. "
                f"Tu tono es cercano, directo y motivador — como un colega que conoce "
                f"la realidad del aula peruana y quiere ayudar al docente a crecer. "
                f"Nunca suenas corporativo ni genérico. "
                f"Siempre conectas los datos concretos del curso con la situación real "
                f"del docente. Escribes en español peruano natural, sin tecnicismos."
                f"\n\nEJEMPLO DE JUSTIFICACIÓN CORRECTA:\n"
                f"Docentes con tu perfil en Lima lo seleccionaron como uno de sus"
                f"cursos más útiles para el trabajo en aula. Con 20 horas y 91% de aprobación,"
                f" es una de las formaciones más eficientes disponibles ahora mismo."
            )

            user_prompt = (
                f"PERFIL DEL DOCENTE:\n"
                f"{docente_contexto}\n\n"
                f"CURSO RECOMENDADO:\n"
                f"- Nombre: {rec.get('CURSO', '')}\n"
                f"- Duración: {horas} horas\n"
                f"- Calificación promedio: {cal_txt}\n"
                f"- Público objetivo: {rec.get('PUBLICO_OBJETIVO', 'Todos los niveles')}\n"
                f"- Propósito del curso: {rec.get('PROPOSITO', '')[:250]}\n"
                f"- Tasa de culminación: {tasa_cul*100:.0f}% de docentes lo completó\n"
                f"- Tasa de aprobación: {tasa_apr*100:.0f}%\n\n"
                f"POR QUÉ SE RECOMIENDA A ESTE DOCENTE:\n"
                f"{razones_txt}.\n\n"
                f"TAREA:\n"
                f"Escribe 2 oraciones cortas (máximo 40 palabras en total) que expliquen "
                f"de forma personalizada y motivadora por qué este curso es valioso "
                f"para ESTE docente en particular. "
                f"Usa datos concretos del curso (calificación, tasa de culminación, horas) "
                f"y conecta con el perfil del docente (nivel, región). "
                f"La primera oración explica el valor del curso. "
                f"La segunda motiva a tomarlo con un dato concreto o beneficio específico. "
                f"No empieces ninguna oración con 'Este curso'. "
                f"No uses comillas. No uses listas. Solo el texto directo."
            )

            return self._llamar_llm_justificacion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                curso_id=rec.get("ID_OFERTA_FORMATIVA", "?")
            )

        # Ejecutar en paralelo con ThreadPoolExecutor
        max_workers = min(RecommenderConfig.MAX_LLM_WORKERS, len(recomendaciones))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_generar_una, rec): i for i, rec in enumerate(recomendaciones)}
            for future, i in futures.items():
                try:
                    recomendaciones[i]["justificacion"] = future.result(timeout=15)
                except Exception as e:
                    logger.warning(f"⚠️  Justificación {i} falló: {e}")
                    recomendaciones[i]["justificacion"] = "Curso relevante para tu nivel y región educativa."

        return recomendaciones

    def _llamar_llm_justificacion(
        self,
        system_prompt: str,
        user_prompt: str,
        curso_id: str = "?"
    ) -> str:

        # Intento 1: OpenAI — con roles separados (system + user)
        if self.openai_client:
            try:
                resp = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens=120,      
                    temperature=0.75,    
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"⚠️  GPT falló (curso {curso_id}): {e} → Gemini...")

        # Intento 2: Gemini
        if self.gemini_client:
            try:
                resp = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"{system_prompt}\n\n{user_prompt}",
                    config={"temperature": 0.75, "max_output_tokens": 120}
                )
                return resp.text.strip()
            except Exception as e2:
                logger.warning(f"⚠️  Gemini falló (curso {curso_id}): {e2}")

        return "Curso relevante para tu nivel educativo y región. Docentes con tu perfil lo completaron con alta valoración."

    # ══════════════════════════════════════════════════════════════════
    # 📊 UTILIDADES PÚBLICAS
    # ══════════════════════════════════════════════════════════════════

    def cursos_del_docente(self, user_id: str) -> pd.DataFrame:
        return self.df_raw[self.df_raw["USUARIO_DOCUMENTO"] == user_id].copy()

    def info_curso(self, id_oferta: str) -> Optional[Dict]:
        f = self.df_cursos[self.df_cursos["ID_OFERTA_FORMATIVA"] == str(id_oferta)]
        return None if f.empty else f.iloc[0].to_dict()

    def refrescar_datos(self, df_nuevo: pd.DataFrame) -> "HybridRecommender":
        """
        Crea una NUEVA instancia con los datos actualizados.
        NO llama self.__init__() para evitar estado corrupto si falla.
        Retorna la nueva instancia; el caller debe hacer el swap.
        """
        logger.info("🔄 Creando nueva instancia del recomendador...")
        return HybridRecommender(
            df=df_nuevo,
            openai_api_key=self._openai_key,
            gemini_api_key=self._gemini_key,
        )


# ══════════════════════════════════════════════════════════════════════
# 🏭 FUNCIONES DE FÁBRICA
# ══════════════════════════════════════════════════════════════════════

def crear_recomendador(df: pd.DataFrame) -> HybridRecommender:
    api_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    return HybridRecommender(
        df=df,
        openai_api_key=api_key,
        gemini_api_key=gemini_key
    )


def obtener_recomendaciones(
    user_id: str,
    df: pd.DataFrame,
    top_k: int = 5,
    incluir_justificacion: bool = True,
) -> List[Dict]:
    """Conveniencia: instancia y ejecuta en una sola llamada."""
    return crear_recomendador(df).recomendar_hibrido(
        user_id=user_id,
        top_k=top_k,
        incluir_justificacion=incluir_justificacion
    )