"""
🎯 SISTEMA DE RECOMENDACIÓN DE CURSOS - FILTRO COLABORATIVO HÍBRIDO
====================================================================
DIFODS - Agente Tecnológico v2.0

Adaptado a una sola tabla con los campos disponibles:

    AÑO, TIPO_CONSTANCIA, CURSO, NOMBRE_DRE, NOMBRE_UGEL,
    USUARIO_DOCUMENTO, NOMBRE_COMPLETO, ID_MOODLE, NIVELNEXUS,
    APROBACION, ID_OFERTA_FORMATIVA, FECHA_NACIMIENTO,
    HORAS_PROGRAMA, PROPOSITO, ACTIVO,
    PORCENTAJE_AVANCE (0-1), NOTA, RATING_USUARIO (1-5)

Componentes del modelo híbrido:
    1. Colaborativo  (40%) → usuarios con mismo NIVELNEXUS/DRE que aprobaron
    2. Popularidad   (30%) → tasa aprobación × nota promedio × log(inscritos)
    3. Historial     (20%) → co-ocurrencia: cursos aprobados por docentes similares
    4. Novedad       (10%) → cursos del año más reciente

NOTA: No se manejan prerrequisitos ni programas. Cada curso es independiente.
"""

import os
import math
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

logger = logging.getLogger(__name__)

# ==============================
# ⚙️ CONFIGURACIÓN
# ==============================

class RecommenderConfig:
    """Pesos y parámetros del sistema híbrido."""

    # Pesos del modelo (deben sumar 1.0)
    PESOS = {
        "colaborativo": 0.40,
        "popularidad":  0.30,
        "historial":    0.20,
        "novedad":      0.10,
    }

    TOP_K_SIMILAR_USERS    = 20   # Vecinos más cercanos en colaborativo
    TOP_K_RECOMENDACIONES  = 5    # Cursos a devolver por defecto
    MIN_SCORE              = 0.05 # Score mínimo para considerar un curso

    # Peso del rating explícito vs comportamiento implícito
    # Rating explícito no es 100% confiable → peso reducido
    PESO_RATING_EXPLICITO  = 0.30
    PESO_COMPORTAMIENTO    = 0.70


# ==============================
# 🔧 CLASE PRINCIPAL
# ==============================

class HybridRecommender:
    """
    Sistema de recomendación híbrido para cursos SIFODS.

    Trabaja sobre un único DataFrame cargado desde PostgreSQL
    (o cualquier fuente). Todos los cálculos son en Python/pandas.
    """

    def __init__(self, df: pd.DataFrame, openai_api_key: str):
        """
        Args:
            df: DataFrame con la tabla completa de inscripciones/cursos.
                Columnas esperadas (case-insensitive, se normalizan internamente):
                AÑO, TIPO_CONSTANCIA, CURSO, NOMBRE_DRE, NOMBRE_UGEL,
                USUARIO_DOCUMENTO, NOMBRE_COMPLETO, ID_MOODLE, NIVELNEXUS,
                APROBACION, ID_OFERTA_FORMATIVA, FECHA_NACIMIENTO,
                HORAS_PROGRAMA, PROPOSITO, ACTIVO,
                PORCENTAJE_AVANCE, NOTA, RATING_USUARIO
            openai_api_key: API key de OpenAI para justificaciones.
        """
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Normalizar nombres de columnas a mayúsculas
        df.columns = [c.strip().upper() for c in df.columns]
        self.df_raw = df.copy()

        # Verificar columnas mínimas requeridas
        required = {
            "USUARIO_DOCUMENTO", "ID_OFERTA_FORMATIVA", "CURSO",
            "APROBACION", "ACTIVO", "NIVELNEXUS", "AÑO"
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"❌ Faltan columnas requeridas: {missing}")

        # Agregar columnas opcionales con valor por defecto si no existen
        for col, default in [
            ("PORCENTAJE_AVANCE", 0.0),
            ("NOTA", np.nan),
            ("RATING_USUARIO", np.nan),
            ("NOMBRE_DRE", ""),
            ("NOMBRE_UGEL", ""),
            ("HORAS_PROGRAMA", 0),
            ("PROPOSITO", ""),
            ("NOMBRE_COMPLETO", ""),
        ]:
            if col not in df.columns:
                logger.warning(f"⚠️  Columna '{col}' no encontrada. Se usará valor por defecto: {default}")
                self.df_raw[col] = default

        # ── Pre-procesar ──────────────────────────────────────────────
        self._preprocesar()

        # ── Construir vistas agregadas ────────────────────────────────
        self.df_cursos      = self._construir_catalogo_cursos()
        self.df_perfil      = self._construir_perfil_docentes()

        # ── Matriz usuario-curso y similitud ──────────────────────────
        self.matriz_uc      = self._construir_matriz_usuario_curso()
        self.sim_usuarios   = self._calcular_similitud_usuarios()

        logger.info(
            f"✅ HybridRecommender listo | "
            f"Cursos activos: {len(self.df_cursos)} | "
            f"Docentes: {len(self.df_perfil)} | "
            f"Registros: {len(self.df_raw)}"
        )

    # ══════════════════════════════════════════════════════════════════
    # 📐 PRE-PROCESAMIENTO
    # ══════════════════════════════════════════════════════════════════

    def _preprocesar(self):
        """Limpia y estandariza el DataFrame principal."""
        df = self.df_raw

        # Tipos básicos
        df["APROBACION"]        = pd.to_numeric(df["APROBACION"], errors="coerce").fillna(0).astype(int)
        df["ACTIVO"]            = pd.to_numeric(df["ACTIVO"], errors="coerce").fillna(0).astype(int)
        df["PORCENTAJE_AVANCE"] = pd.to_numeric(df["PORCENTAJE_AVANCE"], errors="coerce").fillna(0.0).clip(0, 1)
        df["NOTA"]              = pd.to_numeric(df["NOTA"], errors="coerce")
        df["RATING_USUARIO"]    = pd.to_numeric(df["RATING_USUARIO"], errors="coerce").clip(1, 5)
        df["HORAS_PROGRAMA"]    = pd.to_numeric(df["HORAS_PROGRAMA"], errors="coerce").fillna(0)
        df["AÑO"]               = pd.to_numeric(df["AÑO"], errors="coerce").fillna(0).astype(int)

        # IDs como string para evitar problemas de tipo
        df["USUARIO_DOCUMENTO"]    = df["USUARIO_DOCUMENTO"].astype(str).str.strip()
        df["ID_OFERTA_FORMATIVA"]  = df["ID_OFERTA_FORMATIVA"].astype(str).str.strip()

        # Calcular rating compuesto (columna derivada, usada en colaborativo)
        df["RATING_COMPUESTO"] = df.apply(self._calcular_rating_compuesto, axis=1)

        self.df_raw = df

    def _calcular_rating_compuesto(self, row: pd.Series) -> float:
        """
        Combina comportamiento real (70%) + rating explícito (30%).

        Escala final: 0.0 – 5.0

        Comportamiento real (hasta 3.5 puntos):
            +2.0  → aprobó el curso        (APROBACION == 1)
            +1.0  → avance alto            (PORCENTAJE_AVANCE >= 0.8)
            +0.5  → avance medio           (0.5 <= PORCENTAJE_AVANCE < 0.8)
            +nota_normalizada * 1.0        (NOTA / 20, escala vigesimal peruana)

        Rating explícito (hasta 1.5 puntos):
            rating_usuario / 5 * 1.5
        """
        score = 0.0

        # ── Comportamiento implícito ───────────────────────────────────
        if row["APROBACION"] == 1:
            score += 2.0

        avance = row["PORCENTAJE_AVANCE"]
        if avance >= 0.8:
            score += 1.0
        elif avance >= 0.5:
            score += 0.5

        nota = row["NOTA"]
        if pd.notna(nota) and nota > 0:
            nota_norm = min(nota / 20.0, 1.0)   # escala vigesimal → 0-1
            score += nota_norm * 1.0

        # Peso comportamiento implícito
        score_comportamiento = score * RecommenderConfig.PESO_COMPORTAMIENTO

        # ── Rating explícito ──────────────────────────────────────────
        score_rating = 0.0
        if pd.notna(row["RATING_USUARIO"]):
            score_rating = (row["RATING_USUARIO"] / 5.0) * 1.5 * RecommenderConfig.PESO_RATING_EXPLICITO

        return min(round(score_comportamiento + score_rating, 4), 5.0)

    # ══════════════════════════════════════════════════════════════════
    # 📚 CONSTRUCCIÓN DE VISTAS
    # ══════════════════════════════════════════════════════════════════

    def _construir_catalogo_cursos(self) -> pd.DataFrame:
        """
        Agrega métricas por curso calculadas desde la tabla principal.

        Devuelve un DataFrame con una fila por curso, con:
            ID_OFERTA_FORMATIVA, CURSO, HORAS_PROGRAMA, PROPOSITO,
            AÑO_MAX, ACTIVO, TOTAL_INSCRITOS, TOTAL_APROBADOS,
            TASA_APROBACION, NOTA_PROMEDIO, SCORE_POPULARIDAD
        """
        df = self.df_raw

        agg = df.groupby("ID_OFERTA_FORMATIVA").agg(
            CURSO           = ("CURSO",            "first"),
            HORAS_PROGRAMA  = ("HORAS_PROGRAMA",   "first"),
            PROPOSITO       = ("PROPOSITO",         "first"),
            AÑO_MAX         = ("AÑO",               "max"),
            ACTIVO          = ("ACTIVO",            "max"),   # 1 si al menos un registro está activo
            TOTAL_INSCRITOS = ("USUARIO_DOCUMENTO", "count"),
            TOTAL_APROBADOS = ("APROBACION",        "sum"),
            NOTA_PROMEDIO   = ("NOTA",              "mean"),
        ).reset_index()

        # Tasa de aprobación
        agg["TASA_APROBACION"] = (
            agg["TOTAL_APROBADOS"] / agg["TOTAL_INSCRITOS"].replace(0, np.nan)
        ).fillna(0).clip(0, 1)

        # Nota promedio normalizada (vigesimal → 0-1)
        agg["NOTA_NORM"] = (agg["NOTA_PROMEDIO"].fillna(0) / 20.0).clip(0, 1)

        # ── Score de popularidad ──────────────────────────────────────
        # score = tasa_aprobacion × nota_norm × log(total_inscritos + 1)
        agg["SCORE_POPULARIDAD"] = (
            agg["TASA_APROBACION"] *
            agg["NOTA_NORM"] *
            np.log1p(agg["TOTAL_INSCRITOS"])
        ).round(4)

        # ── Score de novedad ─────────────────────────────────────────
        # Solo se conoce el AÑO (no fecha exacta).
        # Año más reciente en los datos = máxima novedad.
        año_max_global = agg["AÑO_MAX"].max()
        agg["SCORE_NOVEDAD"] = agg["AÑO_MAX"].apply(
            lambda y: self._score_novedad_por_año(y, año_max_global)
        )

        return agg

    def _score_novedad_por_año(self, año_curso: int, año_max: int) -> float:
        """Convierte diferencia de años en score de novedad (0-1)."""
        diff = año_max - año_curso
        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.6
        elif diff == 2:
            return 0.3
        else:
            return 0.1

    def _construir_perfil_docentes(self) -> pd.DataFrame:
        """
        Construye una fila por docente con sus atributos y estadísticas.

            USUARIO_DOCUMENTO, NOMBRE_COMPLETO, NIVELNEXUS,
            NOMBRE_DRE, NOMBRE_UGEL,
            TOTAL_CURSOS, CURSOS_APROBADOS, TASA_COMPLETITUD
        """
        df = self.df_raw

        perfil = df.groupby("USUARIO_DOCUMENTO").agg(
            NOMBRE_COMPLETO  = ("NOMBRE_COMPLETO", "first"),
            NIVELNEXUS       = ("NIVELNEXUS",      "first"),
            NOMBRE_DRE       = ("NOMBRE_DRE",      "first"),
            NOMBRE_UGEL      = ("NOMBRE_UGEL",      "first"),
            TOTAL_CURSOS     = ("ID_OFERTA_FORMATIVA", "count"),
            CURSOS_APROBADOS = ("APROBACION",         "sum"),
        ).reset_index()

        perfil["TASA_COMPLETITUD"] = (
            perfil["CURSOS_APROBADOS"] / perfil["TOTAL_CURSOS"].replace(0, np.nan)
        ).fillna(0).round(4)

        return perfil

    # ══════════════════════════════════════════════════════════════════
    # 📊 MATRIZ USUARIO-CURSO Y SIMILITUD
    # ══════════════════════════════════════════════════════════════════

    def _construir_matriz_usuario_curso(self) -> pd.DataFrame:
        """
        Pivot table: filas = docentes, columnas = cursos, valores = RATING_COMPUESTO.
        Ceros donde el docente no tomó el curso.
        """
        matriz = self.df_raw.pivot_table(
            index="USUARIO_DOCUMENTO",
            columns="ID_OFERTA_FORMATIVA",
            values="RATING_COMPUESTO",
            aggfunc="max",    # Si hay duplicados, tomar el mejor rating
            fill_value=0.0
        )
        logger.info(f"📊 Matriz usuario-curso: {matriz.shape}")
        return matriz

    def _calcular_similitud_usuarios(self) -> np.ndarray:
        """Similitud coseno entre todos los docentes basada en la matriz UC."""
        sim = cosine_similarity(self.matriz_uc.values)
        logger.info(f"📐 Similitud calculada: {sim.shape}")
        return sim

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 1: FILTRO COLABORATIVO (40%)
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_colaborativo(
        self, user_id: str, cursos_excluir: List[str], top_k: int
    ) -> Dict[str, float]:
        """
        Predicción de rating para cursos no tomados, ponderada por
        similitud con los TOP_K_SIMILAR_USERS vecinos más cercanos.

        Returns:
            {ID_OFERTA_FORMATIVA: score_colaborativo}
        """
        if user_id not in self.matriz_uc.index:
            logger.warning(f"⚠️  Usuario {user_id} sin historial → colaborativo omitido")
            return {}

        user_idx  = self.matriz_uc.index.get_loc(user_id)
        similitudes = self.sim_usuarios[user_idx]

        # Vecinos más cercanos (excluir al propio usuario)
        vecinos_idx = np.argsort(similitudes)[::-1]
        vecinos_idx = [i for i in vecinos_idx if i != user_idx]
        vecinos_idx = vecinos_idx[:RecommenderConfig.TOP_K_SIMILAR_USERS]

        scores = {}
        for curso_id in self.matriz_uc.columns:
            if curso_id in cursos_excluir:
                continue

            score_num   = 0.0
            score_denom = 0.0

            for v_idx in vecinos_idx:
                rating = self.matriz_uc.iloc[v_idx][curso_id]
                if rating > 0:
                    sim = similitudes[v_idx]
                    score_num   += rating * sim
                    score_denom += sim

            if score_denom > 0:
                scores[curso_id] = round(score_num / score_denom, 4)

        return scores

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 2: POPULARIDAD (30%)
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_popularidad(
        self, cursos_excluir: List[str]
    ) -> Dict[str, float]:
        """
        Score basado en tasa de aprobación, nota promedio y volumen.

        Returns:
            {ID_OFERTA_FORMATIVA: score_popularidad_normalizado}
        """
        df = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ].copy()

        if df.empty:
            return {}

        max_score = df["SCORE_POPULARIDAD"].max()
        if max_score == 0:
            return {}

        df["SCORE_NORM"] = df["SCORE_POPULARIDAD"] / max_score

        return dict(zip(df["ID_OFERTA_FORMATIVA"], df["SCORE_NORM"].round(4)))

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 3: HISTORIAL POR SIMILITUD (20%)
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_historial(
        self, user_id: str, cursos_excluir: List[str], top_k: int
    ) -> Dict[str, float]:
        """
        Reemplaza el filtro por contenido cuando no se tienen metadatos
        de cursos (tags, área curricular, etc.).

        Estrategia:
            1. Encontrar docentes con el mismo NIVELNEXUS y NOMBRE_DRE.
            2. De esos docentes, obtener cursos que aprobaron.
            3. Contar frecuencia de aparición → normalizar → score.

        Returns:
            {ID_OFERTA_FORMATIVA: score_historial}
        """
        perfil = self.df_perfil[self.df_perfil["USUARIO_DOCUMENTO"] == user_id]
        if perfil.empty:
            logger.warning(f"⚠️  Usuario {user_id} sin perfil → historial omitido")
            return {}

        perfil    = perfil.iloc[0]
        nivel     = perfil["NIVELNEXUS"]
        dre       = perfil["NOMBRE_DRE"]

        # Docentes similares: mismo nivel Y misma DRE
        similares_ids = self.df_perfil[
            (self.df_perfil["NIVELNEXUS"] == nivel) &
            (self.df_perfil["NOMBRE_DRE"] == dre) &
            (self.df_perfil["USUARIO_DOCUMENTO"] != user_id)
        ]["USUARIO_DOCUMENTO"].tolist()

        if not similares_ids:
            # Fallback: solo mismo nivel (sin restricción de DRE)
            similares_ids = self.df_perfil[
                (self.df_perfil["NIVELNEXUS"] == nivel) &
                (self.df_perfil["USUARIO_DOCUMENTO"] != user_id)
            ]["USUARIO_DOCUMENTO"].tolist()

        if not similares_ids:
            return {}

        # Cursos aprobados por los similares, que el docente no ha tomado
        df_similares = self.df_raw[
            (self.df_raw["USUARIO_DOCUMENTO"].isin(similares_ids)) &
            (self.df_raw["APROBACION"] == 1) &
            (~self.df_raw["ID_OFERTA_FORMATIVA"].isin(cursos_excluir)) &
            (self.df_raw["ACTIVO"] == 1)
        ]

        if df_similares.empty:
            return {}

        # Frecuencia de aprobación entre similares como proxy de relevancia
        frecuencia = df_similares.groupby("ID_OFERTA_FORMATIVA")["APROBACION"].count()
        max_freq   = frecuencia.max()
        normalizado = (frecuencia / max_freq).round(4)

        return normalizado.to_dict()

    # ══════════════════════════════════════════════════════════════════
    # 🎯 COMPONENTE 4: NOVEDAD (10%)
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_novedad(
        self, cursos_excluir: List[str]
    ) -> Dict[str, float]:
        """
        Score basado en qué tan reciente es el año del curso.

        Returns:
            {ID_OFERTA_FORMATIVA: score_novedad}
        """
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
        """
        Punto de entrada principal. Combina los 4 componentes.

        Args:
            user_id:                DNI o ID del docente (USUARIO_DOCUMENTO).
            top_k:                  Número de cursos a devolver.
            incluir_justificacion:  Si True, llama al LLM para generar la justificación.

        Returns:
            Lista de dicts con información del curso recomendado.
        """
        logger.info(f"\n🎯 Generando recomendaciones para docente {user_id}...")

        # ── Cursos que el docente ya tomó (excluir) ───────────────────
        cursos_tomados = self.df_raw[
            self.df_raw["USUARIO_DOCUMENTO"] == user_id
        ]["ID_OFERTA_FORMATIVA"].unique().tolist()

        # ── Obtener scores de cada componente ─────────────────────────
        expand = top_k * 4  # candidatos amplios antes de filtrar

        s_colab  = self._recomendar_colaborativo(user_id, cursos_tomados, expand)
        s_popul  = self._recomendar_popularidad(cursos_tomados)
        s_hist   = self._recomendar_historial(user_id, cursos_tomados, expand)
        s_novel  = self._recomendar_novedad(cursos_tomados)

        # ── Unión de todos los candidatos ─────────────────────────────
        todos_cursos = set(s_colab) | set(s_popul) | set(s_hist) | set(s_novel)

        if not todos_cursos:
            logger.warning(f"⚠️  Sin candidatos para {user_id}. Devolviendo cursos populares como fallback.")
            return self._fallback_populares(cursos_tomados, top_k)

        # ── Normalizar scores colaborativo a 0-1 ─────────────────────
        if s_colab:
            max_c = max(s_colab.values())
            if max_c > 0:
                s_colab = {k: v / max_c for k, v in s_colab.items()}

        # ── Score final ponderado ─────────────────────────────────────
        resultados = []
        pesos = RecommenderConfig.PESOS

        for curso_id in todos_cursos:
            sc = s_colab.get(curso_id, 0.0) * pesos["colaborativo"]
            sp = s_popul.get(curso_id, 0.0) * pesos["popularidad"]
            sh = s_hist.get(curso_id, 0.0)  * pesos["historial"]
            sn = s_novel.get(curso_id, 0.0) * pesos["novedad"]

            score_final = sc + sp + sh + sn

            if score_final < RecommenderConfig.MIN_SCORE:
                continue

            metodos = []
            if s_colab.get(curso_id, 0) > 0: metodos.append("colaborativo")
            if s_popul.get(curso_id, 0) > 0: metodos.append("popularidad")
            if s_hist.get(curso_id, 0)  > 0: metodos.append("historial")
            if s_novel.get(curso_id, 0) > 0: metodos.append("novedad")

            resultados.append({
                "ID_OFERTA_FORMATIVA": curso_id,
                "score_final": round(score_final, 4),
                "scores_detalle": {
                    "colaborativo": round(sc, 4),
                    "popularidad":  round(sp, 4),
                    "historial":    round(sh, 4),
                    "novedad":      round(sn, 4),
                },
                "metodos_usados": metodos,
            })

        if not resultados:
            return self._fallback_populares(cursos_tomados, top_k)

        # ── Ordenar y tomar top K ─────────────────────────────────────
        resultados.sort(key=lambda x: x["score_final"], reverse=True)
        top_resultados = resultados[:top_k]

        # ── Enriquecer con info del catálogo ──────────────────────────
        top_resultados = self._enriquecer_con_catalogo(top_resultados)

        # ── Justificación con LLM ─────────────────────────────────────
        if incluir_justificacion:
            top_resultados = self._generar_justificaciones(top_resultados, user_id)

        logger.info(f"✅ {len(top_resultados)} cursos recomendados para {user_id}")
        return top_resultados

    # ══════════════════════════════════════════════════════════════════
    # 🛟 FALLBACK
    # ══════════════════════════════════════════════════════════════════

    def _fallback_populares(
        self, cursos_excluir: List[str], top_k: int
    ) -> List[Dict]:
        """
        Devuelve los cursos más populares cuando no hay datos suficientes
        para el modelo híbrido.
        """
        logger.info("🔄 Usando fallback: cursos más populares")

        df = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ].sort_values("SCORE_POPULARIDAD", ascending=False).head(top_k)

        resultados = []
        for _, row in df.iterrows():
            resultados.append({
                "ID_OFERTA_FORMATIVA": row["ID_OFERTA_FORMATIVA"],
                "CURSO":              row["CURSO"],
                "HORAS_PROGRAMA":     int(row["HORAS_PROGRAMA"]),
                "PROPOSITO":          row["PROPOSITO"],
                "AÑO":                int(row["AÑO_MAX"]),
                "TOTAL_INSCRITOS":    int(row["TOTAL_INSCRITOS"]),
                "TASA_APROBACION":    round(float(row["TASA_APROBACION"]), 2),
                "score_final":        round(float(row["SCORE_POPULARIDAD"]), 4),
                "scores_detalle":     {},
                "metodos_usados":     ["popularidad_fallback"],
                "justificacion":      "Curso muy valorado por docentes de la plataforma",
            })

        return resultados

    # ══════════════════════════════════════════════════════════════════
    # 🗂️ ENRIQUECIMIENTO CON CATÁLOGO
    # ══════════════════════════════════════════════════════════════════

    def _enriquecer_con_catalogo(self, resultados: List[Dict]) -> List[Dict]:
        """Agrega datos del catálogo de cursos a cada resultado."""
        catalogo_idx = self.df_cursos.set_index("ID_OFERTA_FORMATIVA")

        for rec in resultados:
            cid = rec["ID_OFERTA_FORMATIVA"]
            if cid in catalogo_idx.index:
                fila = catalogo_idx.loc[cid]
                rec["CURSO"]           = fila["CURSO"]
                rec["HORAS_PROGRAMA"]  = int(fila["HORAS_PROGRAMA"])
                rec["PROPOSITO"]       = str(fila["PROPOSITO"])[:300]  # truncar si es muy largo
                rec["AÑO"]             = int(fila["AÑO_MAX"])
                rec["TOTAL_INSCRITOS"] = int(fila["TOTAL_INSCRITOS"])
                rec["TASA_APROBACION"] = round(float(fila["TASA_APROBACION"]), 2)
            else:
                rec["CURSO"]           = "Curso no encontrado"
                rec["HORAS_PROGRAMA"]  = 0
                rec["PROPOSITO"]       = ""
                rec["AÑO"]             = 0
                rec["TOTAL_INSCRITOS"] = 0
                rec["TASA_APROBACION"] = 0.0

        return resultados

    # ══════════════════════════════════════════════════════════════════
    # 🤖 JUSTIFICACIÓN CON LLM
    # ══════════════════════════════════════════════════════════════════

    def _generar_justificaciones(
        self, recomendaciones: List[Dict], user_id: str
    ) -> List[Dict]:
        """
        Genera una frase corta (≤15 palabras) explicando por qué cada
        curso es relevante para el docente, usando GPT-4o-mini.
        """
        # Perfil del docente para contextualizar el prompt
        perfil = self.df_perfil[self.df_perfil["USUARIO_DOCUMENTO"] == user_id]
        if perfil.empty:
            perfil_texto = "Docente de la plataforma SIFODS"
        else:
            p = perfil.iloc[0]
            perfil_texto = (
                f"Docente de nivel {p['NIVELNEXUS']}, "
                f"DRE {p['NOMBRE_DRE']}, UGEL {p['NOMBRE_UGEL']}. "
                f"Ha completado {int(p['CURSOS_APROBADOS'])} cursos "
                f"con una tasa de aprobación de {p['TASA_COMPLETITUD']*100:.0f}%."
            )

        for rec in recomendaciones:
            prompt = f"""Eres un asistente de recomendación de cursos para docentes del Ministerio de Educación del Perú.

Perfil del docente: {perfil_texto}

Curso recomendado:
- Nombre: {rec.get('CURSO', '')}
- Duración: {rec.get('HORAS_PROGRAMA', 0)} horas
- Propósito: {rec.get('PROPOSITO', '')[:200]}
- Cómo se seleccionó: {', '.join(rec.get('metodos_usados', []))}

Genera UNA sola oración (máximo 15 palabras) explicando POR QUÉ este curso es relevante para este docente.
Sé específico y motivador. No empieces con "Este curso".
Responde SOLO con la oración, sin comillas ni puntos extra."""

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                    temperature=0.7,
                )
                rec["justificacion"] = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"⚠️  LLM falló para curso {rec.get('ID_OFERTA_FORMATIVA')}: {e}")
                rec["justificacion"] = "Curso relevante para tu nivel y región educativa."

        return recomendaciones

    # ══════════════════════════════════════════════════════════════════
    # 📊 UTILIDADES PÚBLICAS
    # ══════════════════════════════════════════════════════════════════

    def cursos_del_docente(self, user_id: str) -> pd.DataFrame:
        """Devuelve el historial de cursos tomados por el docente."""
        return self.df_raw[self.df_raw["USUARIO_DOCUMENTO"] == user_id].copy()

    def info_curso(self, id_oferta: str) -> Optional[Dict]:
        """Devuelve la información agregada de un curso por ID."""
        fila = self.df_cursos[self.df_cursos["ID_OFERTA_FORMATIVA"] == str(id_oferta)]
        if fila.empty:
            return None
        return fila.iloc[0].to_dict()

    def refrescar_datos(self, df_nuevo: pd.DataFrame):
        """
        Permite actualizar el DataFrame base sin reiniciar el servidor.
        Útil cuando llegan nuevos datos desde PostgreSQL.
        """
        logger.info("🔄 Refrescando datos del recomendador...")
        self.__init__(df_nuevo, openai_api_key=self.openai_client.api_key)


# ══════════════════════════════════════════════════════════════════════
# 🏭 FUNCIÓN DE FÁBRICA (usada desde app.py)
# ══════════════════════════════════════════════════════════════════════

def crear_recomendador(df: pd.DataFrame) -> HybridRecommender:
    """
    Crea una instancia del recomendador con la API key del entorno.

    Args:
        df: DataFrame con la tabla de inscripciones/cursos.

    Returns:
        Instancia de HybridRecommender lista para usar.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ OPENAI_API_KEY no definida en el entorno.")
    return HybridRecommender(df=df, openai_api_key=api_key)


def obtener_recomendaciones(
    user_id: str,
    df: pd.DataFrame,
    top_k: int = 5,
    incluir_justificacion: bool = True,
) -> List[Dict]:
    """
    Función de conveniencia para obtener recomendaciones en una sola llamada.
    Instancia el recomendador, ejecuta y devuelve los resultados.

    Args:
        user_id:                DNI o ID del docente.
        df:                     DataFrame con los datos (tabla única).
        top_k:                  Número de recomendaciones a devolver.
        incluir_justificacion:  Si se deben generar justificaciones con LLM.

    Returns:
        Lista de diccionarios con los cursos recomendados.

    Ejemplo de uso desde app.py:
        recomendaciones = obtener_recomendaciones(
            user_id="45737373",
            df=df_global,
            top_k=5,
            incluir_justificacion=True
        )
    """
    recomendador = crear_recomendador(df)
    return recomendador.recomendar_hibrido(
        user_id=user_id,
        top_k=top_k,
        incluir_justificacion=incluir_justificacion,
    )


# ══════════════════════════════════════════════════════════════════════
# 🧪 PRUEBA RÁPIDA (ejecutar directamente: python sistema_recomendacion.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()

    # Datos de ejemplo con la estructura real
    datos_ejemplo = [
        {
            "AÑO": 2025, "TIPO_CONSTANCIA": "Curso",
            "CURSO": "Inteligencia artificial en la práctica docente",
            "NOMBRE_DRE": "DRE SAN MARTIN", "NOMBRE_UGEL": "UGEL HUALLAGA",
            "USUARIO_DOCUMENTO": "45737373", "NOMBRE_COMPLETO": "ELIZABETH RUT SANDOVAL SALINAS",
            "ID_MOODLE": 505148, "NIVELNEXUS": "PRIMARIA",
            "APROBACION": 0, "ID_OFERTA_FORMATIVA": "512",
            "FECHA_NACIMIENTO": "1989-05-16", "HORAS_PROGRAMA": 64,
            "PROPOSITO": "Fortalecer competencias en IA para la práctica pedagógica.",
            "ACTIVO": 1, "PORCENTAJE_AVANCE": 0.45, "NOTA": None, "RATING_USUARIO": None
        },
        {
            "AÑO": 2024, "TIPO_CONSTANCIA": "Curso",
            "CURSO": "Evaluación formativa en el aula",
            "NOMBRE_DRE": "DRE SAN MARTIN", "NOMBRE_UGEL": "UGEL HUALLAGA",
            "USUARIO_DOCUMENTO": "45737373", "NOMBRE_COMPLETO": "ELIZABETH RUT SANDOVAL SALINAS",
            "ID_MOODLE": 505148, "NIVELNEXUS": "PRIMARIA",
            "APROBACION": 1, "ID_OFERTA_FORMATIVA": "300",
            "FECHA_NACIMIENTO": "1989-05-16", "HORAS_PROGRAMA": 40,
            "PROPOSITO": "Desarrollar estrategias de evaluación formativa.",
            "ACTIVO": 1, "PORCENTAJE_AVANCE": 1.0, "NOTA": 16.0, "RATING_USUARIO": 4
        },
        {
            "AÑO": 2025, "TIPO_CONSTANCIA": "Curso",
            "CURSO": "Didáctica de la matemática en primaria",
            "NOMBRE_DRE": "DRE SAN MARTIN", "NOMBRE_UGEL": "UGEL HUALLAGA",
            "USUARIO_DOCUMENTO": "99887766", "NOMBRE_COMPLETO": "CARLOS MENDOZA RIOS",
            "ID_MOODLE": 600001, "NIVELNEXUS": "PRIMARIA",
            "APROBACION": 1, "ID_OFERTA_FORMATIVA": "610",
            "FECHA_NACIMIENTO": "1980-03-10", "HORAS_PROGRAMA": 48,
            "PROPOSITO": "Estrategias didácticas para enseñanza de matemáticas.",
            "ACTIVO": 1, "PORCENTAJE_AVANCE": 1.0, "NOTA": 18.0, "RATING_USUARIO": 5
        },
        {
            "AÑO": 2025, "TIPO_CONSTANCIA": "Curso",
            "CURSO": "Inteligencia artificial en la práctica docente",
            "NOMBRE_DRE": "DRE SAN MARTIN", "NOMBRE_UGEL": "UGEL HUALLAGA",
            "USUARIO_DOCUMENTO": "99887766", "NOMBRE_COMPLETO": "CARLOS MENDOZA RIOS",
            "ID_MOODLE": 600001, "NIVELNEXUS": "PRIMARIA",
            "APROBACION": 1, "ID_OFERTA_FORMATIVA": "512",
            "FECHA_NACIMIENTO": "1980-03-10", "HORAS_PROGRAMA": 64,
            "PROPOSITO": "Fortalecer competencias en IA para la práctica pedagógica.",
            "ACTIVO": 1, "PORCENTAJE_AVANCE": 1.0, "NOTA": 15.0, "RATING_USUARIO": 4
        },
    ]

    df_prueba = pd.DataFrame(datos_ejemplo)

    print("=" * 60)
    print("🧪 PRUEBA DEL SISTEMA DE RECOMENDACIÓN")
    print("=" * 60)

    recomendaciones = obtener_recomendaciones(
        user_id="45737373",
        df=df_prueba,
        top_k=3,
        incluir_justificacion=True,
    )

    for i, rec in enumerate(recomendaciones, 1):
        print(f"\n{'─'*50}")
        print(f"#{i} {rec.get('CURSO', 'Sin nombre')}")
        print(f"   ID: {rec['ID_OFERTA_FORMATIVA']} | "
              f"{rec.get('HORAS_PROGRAMA', 0)}h | "
              f"Año {rec.get('AÑO', '?')}")
        print(f"   Score: {rec['score_final']} | "
              f"Métodos: {', '.join(rec['metodos_usados'])}")
        print(f"   💬 {rec.get('justificacion', '')}")
