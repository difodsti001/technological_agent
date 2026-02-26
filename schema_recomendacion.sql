-- =====================================================================
-- SCHEMA OPERACIONAL - AGENTE TECNOLÓGICO DIFODS
-- =====================================================================
-- Este schema crea ÚNICAMENTE las tablas propias del agente.
-- Los datos de negocio (inscripciones, cursos, docentes) viven en
-- tu BD origen y el agente los lee como fuente externa (no los posee).
--
-- FUENTE DE DATOS PARA RECOMENDACIÓN (configurar en .env):
--   PostgreSQL → DB_TABLE_RECOMENDACION=nombre_de_tu_tabla_o_vista
--   Excel      → EXCEL_FALLBACK_PATH=data/inscripciones.xlsx
--   Los cálculos del modelo híbrido los hace Python, no SQL.
--
-- TABLAS QUE CREA ESTE SCHEMA:
--   1. conversaciones_agente   — log de todas las interacciones
--   2. recomendaciones_detalle — detalle por curso recomendado (scores + justificación LLM)
--
-- VISTAS DE MONITOREO:
--   v_metricas_diarias          — consultas por día y módulo
--   v_historial_recomendaciones — recomendaciones con justificaciones LLM
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; 

-- =====================================================================
-- TABLA 1: conversaciones_agente
-- Log unificado de todas las interacciones del agente.
--
-- tarea = 'sifods'        → consulta RAG sobre la plataforma (/api/sifods)
-- tarea = 'recomendacion' → recomendación de cursos (/api/recomendar)
--
-- Para recomendaciones, el detalle por curso se guarda en
-- recomendaciones_detalle (FK → conversaciones_agente.id).
-- =====================================================================

CREATE TABLE IF NOT EXISTS conversaciones_agente (
    id              SERIAL PRIMARY KEY,
    usuario         VARCHAR(100)  NOT NULL,
    nombre_usuario  VARCHAR(200),
    mensaje         TEXT          NOT NULL,
    respuesta       TEXT          NOT NULL,
    tarea           VARCHAR(20)   NOT NULL
                    CHECK (tarea IN ('sifods', 'recomendacion')),
    fuente_datos    VARCHAR(50),
    tokens_entrada  INTEGER,
    tokens_salida   INTEGER,
    latencia_ms     INTEGER,
    timestamp       TIMESTAMPTZ   DEFAULT NOW()
);

COMMENT ON TABLE conversaciones_agente IS
    'Log unificado de interacciones. '
    'tarea=sifods → RAG. tarea=recomendacion → modelo híbrido. '
    'Las recomendaciones tienen detalle en recomendaciones_detalle.';

CREATE INDEX IF NOT EXISTS idx_conv_usuario   ON conversaciones_agente(usuario);
CREATE INDEX IF NOT EXISTS idx_conv_tarea     ON conversaciones_agente(tarea);
CREATE INDEX IF NOT EXISTS idx_conv_timestamp ON conversaciones_agente(timestamp DESC);


-- =====================================================================
-- TABLA 2: recomendaciones_detalle
-- Una fila por curso recomendado dentro de cada interacción.
-- Permite auditar qué recomendó el modelo, con qué scores
-- y qué justificación generó el LLM para cada docente.
--
-- Relación: conversaciones_agente (1) ──< recomendaciones_detalle (N)
-- =====================================================================

CREATE TABLE IF NOT EXISTS recomendaciones_detalle (
    id                  SERIAL PRIMARY KEY,
    conversacion_id     INTEGER      NOT NULL
                        REFERENCES conversaciones_agente(id) ON DELETE CASCADE,
    usuario             VARCHAR(100) NOT NULL,

    -- Datos del curso (snapshot en el momento de la recomendación)
    posicion            SMALLINT     NOT NULL,
    id_oferta_formativa VARCHAR(50),
    curso               VARCHAR(255),
    horas_programa      INTEGER,
    publico_objetivo    VARCHAR(200),
    tasa_culminacion    NUMERIC(4,2),
    tasa_aprobacion     NUMERIC(4,2),
    calificacion_prom   NUMERIC(3,1),

    -- Scores del modelo híbrido (calculados en Python)
    score_final         NUMERIC(6,4),
    score_colaborativo  NUMERIC(6,4),
    score_popularidad   NUMERIC(6,4),
    score_historial     NUMERIC(6,4),
    score_novedad       NUMERIC(6,4),
    algoritmos_usados   TEXT[],

    -- Justificación generada por el LLM
    justificacion       TEXT,

    timestamp           TIMESTAMPTZ  DEFAULT NOW()
);

COMMENT ON TABLE recomendaciones_detalle IS
    'Detalle por curso de cada recomendación. '
    'Una fila = un curso recomendado dentro de una interacción. '
    'Los datos del curso se copian al momento para tener snapshot histórico.';

COMMENT ON COLUMN recomendaciones_detalle.posicion IS
    'Ranking del curso en la respuesta (1 = mejor recomendado).';

COMMENT ON COLUMN recomendaciones_detalle.algoritmos_usados IS
    'Algoritmos que contribuyeron: colaborativo, popularidad, historial, novedad.';

COMMENT ON COLUMN recomendaciones_detalle.justificacion IS
    'Texto generado por el LLM explicando por qué este curso es relevante para el docente.';

CREATE INDEX IF NOT EXISTS idx_rec_det_conversacion ON recomendaciones_detalle(conversacion_id);
CREATE INDEX IF NOT EXISTS idx_rec_det_usuario      ON recomendaciones_detalle(usuario);
CREATE INDEX IF NOT EXISTS idx_rec_det_oferta       ON recomendaciones_detalle(id_oferta_formativa);
CREATE INDEX IF NOT EXISTS idx_rec_det_timestamp    ON recomendaciones_detalle(timestamp DESC);


-- =====================================================================
-- VISTAS DE MONITOREO
-- =====================================================================

-- DROP previo necesario si las vistas ya existen con estructura distinta
-- (CREATE OR REPLACE VIEW no puede cambiar nombres/orden de columnas)
DROP VIEW IF EXISTS v_historial_recomendaciones;
DROP VIEW IF EXISTS v_metricas_diarias;

CREATE OR REPLACE VIEW v_metricas_diarias AS
SELECT
    DATE(timestamp)                                                    AS fecha,
    tarea,
    COUNT(*)                                                           AS consultas,
    ROUND(AVG(latencia_ms))                                            AS latencia_promedio_ms,
    SUM(COALESCE(tokens_entrada, 0) + COALESCE(tokens_salida, 0))      AS tokens_totales
FROM conversaciones_agente
GROUP BY DATE(timestamp), tarea
ORDER BY fecha DESC, tarea;

COMMENT ON VIEW v_metricas_diarias IS
    'Métricas agregadas por día y módulo (sifods / recomendacion).';


CREATE OR REPLACE VIEW v_historial_recomendaciones AS
SELECT
    ca.timestamp,
    ca.usuario,
    ca.nombre_usuario,
    rd.posicion,
    rd.id_oferta_formativa,
    rd.curso,
    rd.horas_programa,
    rd.publico_objetivo,
    rd.score_final,
    rd.algoritmos_usados,
    rd.tasa_culminacion,
    rd.calificacion_prom,
    rd.justificacion,
    ca.latencia_ms
FROM conversaciones_agente   ca
JOIN recomendaciones_detalle rd ON rd.conversacion_id = ca.id
WHERE ca.tarea = 'recomendacion'
ORDER BY ca.timestamp DESC, rd.posicion ASC;

COMMENT ON VIEW v_historial_recomendaciones IS
    'Historial completo de recomendaciones con justificaciones LLM. '
    'Una fila por curso recomendado, ordenado por fecha desc y posición asc.';


-- =====================================================================
-- GRANTS (descomentar y ajustar según roles en producción)
-- =====================================================================

-- CREATE ROLE agente_tecnologico_role;
-- GRANT SELECT, INSERT        ON conversaciones_agente       TO agente_tecnologico_role;
-- GRANT SELECT, INSERT        ON recomendaciones_detalle     TO agente_tecnologico_role;
-- GRANT SELECT                ON v_metricas_diarias          TO agente_tecnologico_role;
-- GRANT SELECT                ON v_historial_recomendaciones TO agente_tecnologico_role;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public      TO agente_tecnologico_role;