"""
SISTEMA DE RECOMENDACIÓN DE CURSOS - FILTRO COLABORATIVO HÍBRIDO
=========================================================
Tabla única en PostgreSQL (vista o tabla plana) con los campos reales:

    AÑO, TIPO_CONSTANCIA, CURSO, NOMBRE_DRE, NOMBRE_UGEL,
    USUARIO_DOCUMENTO, NOMBRE_COMPLETO, NIVELNEXUS,
    APROBACION, ID_OFERTA_FORMATIVA, ID_GRUPO,
    FECHA_NACIMIENTO, ES_FOCALIZADO, HORAS_PROGRAMA,
    CALIFICACIONES,   ← rating del CURSO (0-5), NO del docente
    PROPOSITO, ACTIVO, PUBLICO_OBJETIVO, CURSO_CULMINADO, EDAD

Componentes del modelo híbrido (pesos configurables en .env):
    1. Colaborativo  (40%) → producto matricial numpy (NO doble loop)
    2. Popularidad   (30%) → tasa_culminacion × calificacion_norm × log(inscritos)
    3. Historial     (20%) → co-ocurrencia con peso PUBLICO_OBJETIVO
    4. Novedad       (10%) → score por AÑO más reciente
"""

import logging
import concurrent.futures

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from openai import OpenAI
from google import genai

from config.settings import settings
from config.prompts import (
    PROMPT_JUSTIFICACION_SYSTEM,
    PROMPT_JUSTIFICACION_USER,
    METODO_TEXTOS,
)

logger = logging.getLogger(__name__)

_cfg = settings.recomendacion


# ══════════════════════════════════════════════════════════════════════
# COLUMNAS ESPERADAS
# ══════════════════════════════════════════════════════════════════════

COLUMNAS_REQUERIDAS = {
    "USUARIO_DOCUMENTO", "ID_OFERTA_FORMATIVA", "CURSO",
    "APROBACION", "ACTIVO", "NIVELNEXUS", "AÑO",
}

COLUMNAS_OPCIONALES: Dict = {
    "PORCENTAJE_AVANCE":  0.0,
    "CALIFICACIONES":     np.nan,
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
# CLASE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

class HybridRecommender:
    """
    Sistema de recomendación híbrido para cursos SIFODS.
    Optimizado para gran escala: operaciones vectorizadas, sparse matrices,
    LLM concurrente.
    Los parámetros de configuración se leen desde config.settings (→ .env).
    """

    def __init__(self, df: pd.DataFrame):
        self.openai_client: Optional[OpenAI] = None
        if settings.llm.tiene_openai:
            self.openai_client = OpenAI(api_key=settings.llm.openai_api_key)

        self.gemini_client = None
        if settings.llm.tiene_gemini:
            try:
                self.gemini_client = genai.Client(api_key=settings.llm.gemini_api_key)
                logger.info("✅ Gemini configurado como fallback LLM")
            except Exception as e:
                logger.warning(f"⚠️  Gemini no disponible: {e}")

        df = df.copy()
        df.columns = [c.strip().upper() for c in df.columns]

        missing = COLUMNAS_REQUERIDAS - set(df.columns)
        if missing:
            raise ValueError(f"❌ Faltan columnas requeridas: {missing}")

        for col, default in COLUMNAS_OPCIONALES.items():
            if col not in df.columns:
                logger.warning(f"⚠️  '{col}' no encontrada → default: {default!r}")
                df[col] = default

        self.df_raw = df

        self._preprocesar()
        self.df_cursos    = self._construir_catalogo_cursos()
        self.df_perfil    = self._construir_perfil_docentes()
        self.matriz_uc, self.user_index, self.curso_index = self._construir_matriz_usuario_curso()
        self.sim_matrix   = self._calcular_similitud_usuarios()

        logger.info(
            f"✅ HybridRecommender| "
            f"Registros: {len(self.df_raw):,} | "
            f"Cursos activos: {(self.df_cursos['ACTIVO'] == 1).sum()} | "
            f"Docentes: {len(self.df_perfil):,} | "
            f"Top_k: {_cfg.top_k}"
        )

    # ══════════════════════════════════════════════════════════════════
    # PRE-PROCESAMIENTO (VECTORIZADO)
    # ══════════════════════════════════════════════════════════════════

    def _preprocesar(self):
        df = self.df_raw

        df["APROBACION"]        = pd.to_numeric(df["APROBACION"],        errors="coerce").fillna(0).astype(int)
        df["ACTIVO"]            = pd.to_numeric(df["ACTIVO"],            errors="coerce").fillna(0).astype(int)
        df["PORCENTAJE_AVANCE"] = pd.to_numeric(df["PORCENTAJE_AVANCE"], errors="coerce").fillna(0.0).clip(0, 1)
        df["HORAS_PROGRAMA"]    = pd.to_numeric(df["HORAS_PROGRAMA"],    errors="coerce").fillna(0)
        df["AÑO"]               = pd.to_numeric(df["AÑO"],               errors="coerce").fillna(0).astype(int)
        df["EDAD"]              = pd.to_numeric(df["EDAD"],               errors="coerce")
        df["CALIFICACIONES"]    = pd.to_numeric(df["CALIFICACIONES"],    errors="coerce").clip(0, 5)
        df["CURSO_CULMINADO"]   = self._normalizar_binario_series(df["CURSO_CULMINADO"])
        df["USUARIO_DOCUMENTO"]   = df["USUARIO_DOCUMENTO"].astype(str).str.strip()
        df["ID_OFERTA_FORMATIVA"] = df["ID_OFERTA_FORMATIVA"].astype(str).str.strip()

        # RATING_COMPUESTO vectorizado
        #  +2.5  si CURSO_CULMINADO == 1  (señal más fuerte)
        #  +1.5  si APROBACION == 1 Y CURSO_CULMINADO == 0
        #  +1.0  si CURSO_CULMINADO==0 Y PORCENTAJE_AVANCE >= 0.8
        #  +0.5  si CURSO_CULMINADO==0 Y 0.5 <= PORCENTAJE_AVANCE < 0.8


        culmino = df["CURSO_CULMINADO"] == 1
        aprobo  = df["APROBACION"]       == 1
        sin_cul = ~culmino
        avance  = df["PORCENTAJE_AVANCE"]

        df["RATING_COMPUESTO"] = (
            culmino.astype(float) * 2.5
            + (sin_cul & aprobo).astype(float) * 1.5
            + (sin_cul & (avance >= 0.8)).astype(float) * 1.0
            + (sin_cul & (avance >= 0.5) & (avance < 0.8)).astype(float) * 0.5
        ).clip(0, 5).round(4)

        self.df_raw = df

    @staticmethod
    def _normalizar_binario_series(serie: pd.Series) -> pd.Series:
        numerico = pd.to_numeric(serie, errors="coerce")
        resultado = (numerico > 0).astype(int)
        es_string = numerico.isna() & serie.notna()
        if es_string.any():
            positivos = serie[es_string].astype(str).str.strip().str.upper().isin(
                {"SI", "SÍ", "S", "TRUE", "YES", "Y", "1"}
            )
            resultado[es_string] = positivos.astype(int)
        return resultado.fillna(0).astype(int)

    # ══════════════════════════════════════════════════════════════════
    # CATÁLOGO DE CURSOS
    # ══════════════════════════════════════════════════════════════════

    def _construir_catalogo_cursos(self) -> pd.DataFrame:
        df = self.df_raw
        agg = df.groupby("ID_OFERTA_FORMATIVA").agg(
            CURSO             = ("CURSO",            "first"),
            HORAS_PROGRAMA    = ("HORAS_PROGRAMA",   "first"),
            PROPOSITO         = ("PROPOSITO",         "first"),
            PUBLICO_OBJETIVO  = ("PUBLICO_OBJETIVO",  "first"),
            AÑO_MAX           = ("AÑO",               "max"),
            ACTIVO            = ("ACTIVO",            "max"),
            TOTAL_INSCRITOS   = ("USUARIO_DOCUMENTO", "count"),
            TOTAL_CULMINADOS  = ("CURSO_CULMINADO",   "sum"),
            TOTAL_APROBADOS   = ("APROBACION",        "sum"),
            CALIFICACION_PROM = ("CALIFICACIONES",    "mean"),
        ).reset_index()

        n = agg["TOTAL_INSCRITOS"].replace(0, np.nan)
        agg["TASA_CULMINACION"]  = (agg["TOTAL_CULMINADOS"] / n).fillna(0).clip(0, 1)
        agg["TASA_APROBACION"]   = (agg["TOTAL_APROBADOS"]  / n).fillna(0).clip(0, 1)
        agg["CALIFICACION_NORM"] = (agg["CALIFICACION_PROM"].fillna(0) / 5.0).clip(0, 1)
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
    # PERFIL DE DOCENTES
    # ══════════════════════════════════════════════════════════════════

    def _construir_perfil_docentes(self) -> pd.DataFrame:
        df = self.df_raw
        perfil = df.groupby("USUARIO_DOCUMENTO").agg(
            NOMBRE_COMPLETO   = ("NOMBRE_COMPLETO",    "first"),
            NIVELNEXUS        = ("NIVELNEXUS",          "first"),
            NOMBRE_DRE        = ("NOMBRE_DRE",          "first"),
            NOMBRE_UGEL       = ("NOMBRE_UGEL",         "first"),
            EDAD_APROX        = ("EDAD",                "first"),
            TOTAL_CURSOS      = ("ID_OFERTA_FORMATIVA", "count"),
            CURSOS_CULMINADOS = ("CURSO_CULMINADO",     "sum"),
            CURSOS_APROBADOS  = ("APROBACION",          "sum"),
        ).reset_index()

        perfil["TASA_COMPLETITUD"] = (
            perfil["CURSOS_CULMINADOS"] / perfil["TOTAL_CURSOS"].replace(0, np.nan)
        ).fillna(0).round(4)
        return perfil

    # ══════════════════════════════════════════════════════════════════
    # MATRIZ USUARIO-CURSO (SPARSE) Y SIMILITUD
    # ══════════════════════════════════════════════════════════════════

    def _construir_matriz_usuario_curso(self) -> Tuple[csr_matrix, pd.Index, pd.Index]:
        pivot = self.df_raw.pivot_table(
            index="USUARIO_DOCUMENTO",
            columns="ID_OFERTA_FORMATIVA",
            values="RATING_COMPUESTO",
            aggfunc="max",
            fill_value=0.0,
        )
        user_index  = pivot.index
        curso_index = pivot.columns
        sparse = csr_matrix(pivot.values.astype(np.float32))
        logger.info(
            f"📊 Matriz usuario-curso: {sparse.shape} | "
            f"Densidad: {sparse.nnz / (sparse.shape[0] * sparse.shape[1]):.2%} | "
            f"RAM estimada: {sparse.data.nbytes / 1e6:.1f} MB"
        )
        return sparse, user_index, curso_index

    def _calcular_similitud_usuarios(self) -> np.ndarray:
        n_usuarios = self.matriz_uc.shape[0]
        max_u      = _cfg.max_usuarios_matriz

        if n_usuarios > max_u:
            logger.warning(
                f"⚠️  {n_usuarios:,} docentes superan el límite ({max_u:,}). "
                f"Usando los {max_u:,} más activos para la similitud."
            )
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
    # COMPONENTE 1: COLABORATIVO
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_colaborativo(
        self, user_id: str, cursos_excluir: List[str], top_k: int
    ) -> Dict[str, float]:
        sim_idx_series = pd.Index(self._sim_user_index)
        if user_id not in sim_idx_series:
            logger.warning(f"⚠️  {user_id} no está en submatriz similitud → colaborativo omitido")
            return {}

        user_pos = sim_idx_series.get_loc(user_id)
        sim_vec  = self.sim_matrix[user_pos].copy()
        sim_vec[user_pos] = 0.0  

        top_n  = min(_cfg.top_k_similar_users, len(sim_vec) - 1)
        top_ix = np.argsort(sim_vec)[::-1][:top_n]

        if not top_ix.size:
            return {}

        sim_weights = sim_vec[top_ix]
        sim_sum     = sim_weights.sum()
        if sim_sum == 0:
            return {}

        all_users_in_sim = self._sim_user_index
        vecinos_idx = [
            np.where(self.user_index == all_users_in_sim[i])[0][0]
            for i in top_ix
            if all_users_in_sim[i] in self.user_index
        ]
        if not vecinos_idx:
            return {}

        sub_matriz = self.matriz_uc[vecinos_idx].toarray()  # shape (V, C)
        scores_vec = (sim_weights[:len(vecinos_idx)] @ sub_matriz) / sim_sum

        cursos_excluir_set = set(cursos_excluir)
        resultados = {}
        for i, curso_id in enumerate(self.curso_index):
            if curso_id in cursos_excluir_set:
                continue
            if scores_vec[i] > 0:
                resultados[str(curso_id)] = float(scores_vec[i])

        if resultados:
            max_s = max(resultados.values())
            if max_s > 0:
                resultados = {k: v / max_s for k, v in resultados.items()}

        return dict(sorted(resultados.items(), key=lambda x: x[1], reverse=True)[:top_k])

    # ══════════════════════════════════════════════════════════════════
    # COMPONENTE 2: POPULARIDAD
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_popularidad(self, cursos_excluir: List[str]) -> Dict[str, float]:
        df = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ]
        max_s = df["SCORE_POPULARIDAD"].max()
        if max_s == 0:
            return {}
        scores = (df["SCORE_POPULARIDAD"] / max_s).clip(0, 1)
        return dict(zip(df["ID_OFERTA_FORMATIVA"].astype(str), scores))

    # ══════════════════════════════════════════════════════════════════
    # COMPONENTE 3: HISTORIAL
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_historial(
        self, user_id: str, cursos_excluir: List[str], top_k: int
    ) -> Dict[str, float]:
        user_data = self.df_raw[self.df_raw["USUARIO_DOCUMENTO"] == user_id]
        if user_data.empty:
            return {}

        nivel = user_data["NIVELNEXUS"].iloc[0]
        dre   = user_data["NOMBRE_DRE"].iloc[0]

        candidatos = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ].copy()

        if candidatos.empty:
            return {}

        # Score base: popularidad
        max_pop = candidatos["SCORE_POPULARIDAD"].max() or 1
        candidatos["score"] = (candidatos["SCORE_POPULARIDAD"] / max_pop).clip(0, 1)

        # Bonus por PUBLICO_OBJETIVO coincidente con nivel
        if nivel:
            match_nivel = candidatos["PUBLICO_OBJETIVO"].str.upper().str.contains(
                str(nivel).upper(), na=False
            ) | (candidatos["PUBLICO_OBJETIVO"].str.upper() == "TODOS")
            candidatos.loc[match_nivel, "score"] *= 1.3

        candidatos["score"] = candidatos["score"].clip(0, 1)

        # Docentes similares de la misma DRE
        if dre:
            misma_dre = self.df_raw[
                (self.df_raw["NOMBRE_DRE"] == dre) &
                (self.df_raw["USUARIO_DOCUMENTO"] != user_id) &
                (self.df_raw["CURSO_CULMINADO"] == 1)
            ]
            if not misma_dre.empty:
                cursos_dre = misma_dre["ID_OFERTA_FORMATIVA"].value_counts(normalize=True)
                for curso_id, freq in cursos_dre.items():
                    mask = candidatos["ID_OFERTA_FORMATIVA"] == curso_id
                    candidatos.loc[mask, "score"] = (
                        candidatos.loc[mask, "score"] + freq * 0.5
                    ).clip(0, 1)

        top = candidatos.nlargest(top_k, "score")
        return dict(zip(top["ID_OFERTA_FORMATIVA"].astype(str), top["score"]))

    # ══════════════════════════════════════════════════════════════════
    # COMPONENTE 4: NOVEDAD
    # ══════════════════════════════════════════════════════════════════

    def _recomendar_novedad(self, cursos_excluir: List[str]) -> Dict[str, float]:
        df = self.df_cursos[
            (self.df_cursos["ACTIVO"] == 1) &
            (~self.df_cursos["ID_OFERTA_FORMATIVA"].isin(cursos_excluir))
        ]
        return dict(zip(df["ID_OFERTA_FORMATIVA"].astype(str), df["SCORE_NOVEDAD"]))

    # ══════════════════════════════════════════════════════════════════
    # RECOMENDACIÓN HÍBRIDA FINAL
    # ══════════════════════════════════════════════════════════════════

    def recomendar_hibrido(
        self,
        user_id: str,
        top_k: Optional[int] = None,
        incluir_justificacion: bool = True,
    ) -> List[Dict]:
        """
        Genera top_k recomendaciones para el docente user_id.
        """
        if top_k is None or top_k <= 0:
            top_k = _cfg.top_k

        logger.info(f"Recomendando para: {user_id} | top_k={top_k}")

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

        pesos = _cfg.pesos
        resultados = []

        for curso_id in todos:
            sc = s_colab.get(curso_id, 0.0) * pesos["colaborativo"]
            sp = s_popul.get(curso_id, 0.0) * pesos["popularidad"]
            sh = s_hist.get(curso_id, 0.0)  * pesos["historial"]
            sn = s_novel.get(curso_id, 0.0) * pesos["novedad"]
            score_final = sc + sp + sh + sn

            if score_final < _cfg.min_score:
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
    # FALLBACK
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
    # ENRIQUECIMIENTO CON CATÁLOGO
    # ══════════════════════════════════════════════════════════════════

    def _enriquecer_con_catalogo(self, resultados: List[Dict]) -> List[Dict]:
        idx = self.df_cursos.set_index("ID_OFERTA_FORMATIVA")
        for rec in resultados:
            cid = rec["ID_OFERTA_FORMATIVA"]
            if cid in idx.index:
                f = idx.loc[cid]
                rec.update({
                    "CURSO":            str(f["CURSO"]),
                    "HORAS_PROGRAMA":   int(f["HORAS_PROGRAMA"]),
                    "PROPOSITO":        str(f["PROPOSITO"])[:300],
                    "PUBLICO_OBJETIVO": str(f["PUBLICO_OBJETIVO"]),
                    "AÑO":              int(f["AÑO_MAX"]),
                    "TOTAL_INSCRITOS":  int(f["TOTAL_INSCRITOS"]),
                    "TASA_CULMINACION": round(float(f["TASA_CULMINACION"]), 2),
                    "TASA_APROBACION":  round(float(f["TASA_APROBACION"]), 2),
                    "CALIFICACION_PROM":round(float(f["CALIFICACION_PROM"] or 0), 2),
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
    # JUSTIFICACIONES CONCURRENTES (GPT → Gemini fallback)
    # ══════════════════════════════════════════════════════════════════

    def _generar_justificaciones_concurrente(
        self, recomendaciones: List[Dict], user_id: str
    ) -> List[Dict]:
        perfil_row = self.df_perfil[self.df_perfil["USUARIO_DOCUMENTO"] == user_id]

        def _generar_una(rec: Dict) -> str:
            cal      = rec.get("CALIFICACION_PROM", 0)
            tasa_cul = rec.get("TASA_CULMINACION", 0)
            tasa_apr = rec.get("TASA_APROBACION", 0)
            horas    = rec.get("HORAS_PROGRAMA", 0)
            metodos  = rec.get("metodos_usados", [])

            razones_txt = "; ".join(
                METODO_TEXTOS[m] for m in metodos if m in METODO_TEXTOS
            ) or "es relevante para tu perfil docente"

            if perfil_row.empty:
                docente_contexto = "Docente de la plataforma SIFODS"
            else:
                p = perfil_row.iloc[0]
                nivel  = str(p.get("NIVELNEXUS", "")).strip()
                dre    = str(p.get("NOMBRE_DRE", "")).strip()
                n_cul  = int(p.get("CURSOS_CULMINADOS", 0))
                n_tot  = int(p.get("TOTAL_CURSOS", 0))
                tasa_doc = float(p.get("TASA_COMPLETITUD", 0)) * 100
                edad   = p.get("EDAD_APROX", None)
                docente_contexto = (
                    f"Nivel educativo: {nivel}. DRE: {dre}. "
                    f"Ha completado {n_cul} de {n_tot} cursos "
                    f"(tasa personal: {tasa_doc:.0f}%). "
                    + (f"Edad aprox: {int(edad)} años. " if pd.notna(edad) else "")
                )

            cal_txt = f"{cal:.1f}/5.0" if cal > 0 else "sin calificación registrada"

            user_prompt = PROMPT_JUSTIFICACION_USER.format(
                docente_contexto  = docente_contexto,
                curso             = rec.get("CURSO", ""),
                horas             = horas,
                calificacion      = cal_txt,
                publico_objetivo  = rec.get("PUBLICO_OBJETIVO", "Todos los niveles"),
                proposito         = str(rec.get("PROPOSITO", ""))[:250],
                tasa_culminacion  = f"{tasa_cul * 100:.0f}",
                tasa_aprobacion   = f"{tasa_apr * 100:.0f}",
                razones           = razones_txt,
            )

            return self._llamar_llm_justificacion(
                system_prompt = PROMPT_JUSTIFICACION_SYSTEM,
                user_prompt   = user_prompt,
                curso_id      = rec.get("ID_OFERTA_FORMATIVA", "?"),
            )

        max_workers = min(_cfg.max_llm_workers, len(recomendaciones))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_generar_una, rec): i for i, rec in enumerate(recomendaciones)}
            for future, i in futures.items():
                try:
                    recomendaciones[i]["justificacion"] = future.result(timeout=15)
                except Exception as e:
                    logger.warning(f"⚠️  Justificación {i} falló: {e}")
                    recomendaciones[i]["justificacion"] = (
                        "Curso relevante para tu nivel y región educativa."
                    )

        return recomendaciones

    def _llamar_llm_justificacion(
        self,
        system_prompt: str,
        user_prompt: str,
        curso_id: str = "?",
    ) -> str:
        llm_cfg = settings.justificacion

        if self.openai_client:
            try:
                resp = self.openai_client.chat.completions.create(
                    model=settings.llm.modelo_principal,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    max_tokens  = llm_cfg.max_tokens,
                    temperature = llm_cfg.temperature,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"⚠️  GPT falló (curso {curso_id}): {e} → Gemini...")

        if self.gemini_client:
            try:
                resp = self.gemini_client.models.generate_content(
                    model    = settings.llm.modelo_fallback,
                    contents = f"{system_prompt}\n\n{user_prompt}",
                    config   = {
                        "temperature":       llm_cfg.temperature,
                        "max_output_tokens": llm_cfg.max_tokens,
                    },
                )
                return resp.text.strip()
            except Exception as e2:
                logger.warning(f"⚠️  Gemini falló (curso {curso_id}): {e2}")

        return "Curso relevante para tu nivel educativo y región. Docentes con tu perfil lo completaron con alta valoración."
    
    @staticmethod
    def _asegurar_completo(texto: str) -> str:
        """
        Garantiza que el texto devuelto termine en oración completa.
        Si el LLM cortó por límite de tokens, trunca en el último punto.
        """
        if not texto:
            return texto
        if texto[-1] in ".!?":
            return texto
        ultimo_punto = max(texto.rfind("."), texto.rfind("!"), texto.rfind("?"))
        if ultimo_punto > len(texto) // 2:
            return texto[:ultimo_punto + 1].strip()
        return texto

    # ══════════════════════════════════════════════════════════════════
    # UTILIDADES PÚBLICAS
    # ══════════════════════════════════════════════════════════════════

    def cursos_del_docente(self, user_id: str) -> pd.DataFrame:
        return self.df_raw[self.df_raw["USUARIO_DOCUMENTO"] == user_id].copy()

    def info_curso(self, id_oferta: str) -> Optional[Dict]:
        f = self.df_cursos[self.df_cursos["ID_OFERTA_FORMATIVA"] == str(id_oferta)]
        return None if f.empty else f.iloc[0].to_dict()

    def refrescar_datos(self, df_nuevo: pd.DataFrame) -> "HybridRecommender":
        """Crea nueva instancia con datos actualizados (swap seguro)."""
        logger.info("🔄 Creando nueva instancia del recomendador...")
        return HybridRecommender(df=df_nuevo)


# ══════════════════════════════════════════════════════════════════════
# FÁBRICA
# ══════════════════════════════════════════════════════════════════════

def crear_recomendador(df: pd.DataFrame) -> HybridRecommender:
    """Crea y retorna una instancia lista del recomendador."""
    return HybridRecommender(df=df)


def obtener_recomendaciones(
    user_id: str,
    df: pd.DataFrame,
    top_k: Optional[int] = None,
    incluir_justificacion: bool = True,
) -> List[Dict]:
    """
    Conveniencia: instancia el recomendador y ejecuta en una sola llamada.
    top_k=None usa el valor de settings.recomendacion.top_k (desde .env).
    """
    return crear_recomendador(df).recomendar_hibrido(
        user_id=user_id,
        top_k=top_k,
        incluir_justificacion=incluir_justificacion,
    )
