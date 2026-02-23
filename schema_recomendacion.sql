-- =====================================================================
-- 📊 SCHEMA - AGENTE TECNOLÓGICO DIFODS v2.2
-- =====================================================================
-- IMPORTANTE: El sistema de recomendación Python trabaja sobre UNA
-- sola vista/tabla plana (v_inscripciones_recomendacion). Este schema
-- define las tablas fuente y la vista que las une en el formato que
-- espera sistema_recomendacion.py.
--
-- Campos que espera Python (case-insensitive, se normalizan a mayúsculas):
--   AÑO, TIPO_CONSTANCIA, CURSO, NOMBRE_DRE, NOMBRE_UGEL,
--   USUARIO_DOCUMENTO, NOMBRE_COMPLETO, NIVELNEXUS, APROBACION,
--   ID_OFERTA_FORMATIVA, ID_GRUPO, FECHA_NACIMIENTO, ES_FOCALIZADO,
--   HORAS_PROGRAMA, CALIFICACIONES (0-5), PROPOSITO, ACTIVO,
--   PUBLICO_OBJETIVO, CURSO_CULMINADO, EDAD
-- =====================================================================

-- Habilitar extensiones
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- búsqueda por similitud de texto

-- =====================================================================
-- TABLA 1: ofertas_formativas
-- Catálogo de cursos disponibles en SIFODS
-- Nota: CALIFICACIONES aquí es el rating del CURSO (0-5),
--       no la nota del docente.
-- =====================================================================

CREATE TABLE IF NOT EXISTS ofertas_formativas (
    id_oferta_formativa     SERIAL PRIMARY KEY,
    codigo_curso            VARCHAR(30) UNIQUE,
    curso                   VARCHAR(255) NOT NULL,
    tipo_constancia         VARCHAR(50),            -- diploma, certificado, constancia
    horas_programa          INTEGER NOT NULL DEFAULT 0,
    proposito               TEXT,
    publico_objetivo        VARCHAR(200),            -- ej: "PRIMARIA", "SECUNDARIA", "TODOS"
    activo                  SMALLINT NOT NULL DEFAULT 1 CHECK (activo IN (0,1)),
    año                     SMALLINT,
    id_grupo                VARCHAR(50),

    -- Calificación del curso (0.0 - 5.0) — promedio de ratings de docentes
    -- DISTINTO de la nota del docente (que es aprobacion/curso_culminado)
    calificaciones          NUMERIC(3,1) DEFAULT NULL CHECK (calificaciones BETWEEN 0 AND 5),

    -- Auditoría
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

COMMENT ON COLUMN ofertas_formativas.calificaciones IS
    'Rating promedio del CURSO asignado por los docentes (escala 0-5). '
    'NO es la nota del docente. Usado en el score de popularidad del recomendador.';

CREATE INDEX idx_of_activo  ON ofertas_formativas(activo);
CREATE INDEX idx_of_año     ON ofertas_formativas(año DESC);
CREATE INDEX idx_of_publico ON ofertas_formativas(publico_objetivo);


-- =====================================================================
-- TABLA 2: docentes
-- Perfil demográfico del docente
-- =====================================================================

CREATE TABLE IF NOT EXISTS docentes (
    usuario_documento   VARCHAR(20) PRIMARY KEY,    -- DNI
    nombre_completo     VARCHAR(200),
    nivelnexus          VARCHAR(50),                -- nivel educativo en SIFODS
    nombre_dre          VARCHAR(100),
    nombre_ugel         VARCHAR(100),
    fecha_nacimiento    DATE,
    edad                SMALLINT,
    es_focalizado       SMALLINT DEFAULT 0 CHECK (es_focalizado IN (0,1)),

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_docentes_nivel ON docentes(nivelnexus);
CREATE INDEX idx_docentes_dre   ON docentes(nombre_dre);


-- =====================================================================
-- TABLA 3: inscripciones_cursos
-- Historial de inscripciones — tabla principal de hechos
-- Esta es la tabla que se configura en DB_TABLE_RECOMENDACION
-- (o se puede apuntar a la vista v_inscripciones_recomendacion)
-- =====================================================================

CREATE TABLE IF NOT EXISTS inscripciones_cursos (
    inscripcion_id      BIGSERIAL PRIMARY KEY,
    usuario_documento   VARCHAR(20) NOT NULL REFERENCES docentes(usuario_documento),
    id_oferta_formativa INTEGER     NOT NULL REFERENCES ofertas_formativas(id_oferta_formativa),

    -- Estado del docente en el curso
    aprobacion          SMALLINT    DEFAULT 0 CHECK (aprobacion IN (0,1)),
    curso_culminado     SMALLINT    DEFAULT 0 CHECK (curso_culminado IN (0,1)),
    porcentaje_avance   NUMERIC(5,2) DEFAULT 0 CHECK (porcentaje_avance BETWEEN 0 AND 100),

    -- Fecha del registro/año académico
    año                 SMALLINT,

    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),

    -- Un docente no puede tener dos registros idénticos para la misma oferta en el mismo año
    UNIQUE (usuario_documento, id_oferta_formativa, año)
);

COMMENT ON COLUMN inscripciones_cursos.aprobacion IS
    'Indica si el docente aprobó el curso (1=Sí, 0=No). '
    'Señal de engagement: +1.5 en RATING_COMPUESTO.';

COMMENT ON COLUMN inscripciones_cursos.curso_culminado IS
    'Indica si el docente culminó el curso (1=Sí, 0=No). '
    'Señal más fuerte de engagement: +2.5 en RATING_COMPUESTO. '
    'Puede venir como 1/0, SI/NO, TRUE/FALSE — el Python lo normaliza.';

COMMENT ON COLUMN inscripciones_cursos.porcentaje_avance IS
    'Avance del docente (0-100%). Se normaliza a 0-1 en Python. '
    'Aporta al RATING_COMPUESTO solo si curso_culminado = 0.';

CREATE INDEX idx_ic_usuario  ON inscripciones_cursos(usuario_documento);
CREATE INDEX idx_ic_oferta   ON inscripciones_cursos(id_oferta_formativa);
CREATE INDEX idx_ic_año      ON inscripciones_cursos(año);
CREATE INDEX idx_ic_aprobado ON inscripciones_cursos(aprobacion);
CREATE INDEX idx_ic_culminado ON inscripciones_cursos(curso_culminado);


-- =====================================================================
-- VISTA PRINCIPAL PARA EL RECOMENDADOR
-- Une las 3 tablas en el formato plano que espera sistema_recomendacion.py
-- Configurar en .env: DB_TABLE_RECOMENDACION=v_inscripciones_recomendacion
-- =====================================================================

CREATE OR REPLACE VIEW v_inscripciones_recomendacion AS
SELECT
    -- De inscripciones
    ic.año,
    ic.aprobacion,
    ic.curso_culminado,
    ic.porcentaje_avance,

    -- De ofertas_formativas
    of.tipo_constancia,
    of.curso,
    of.horas_programa,
    of.proposito,
    of.activo,
    of.publico_objetivo,
    of.id_grupo,
    of.calificaciones,          -- rating del curso (0-5)
    of.id_oferta_formativa::VARCHAR AS id_oferta_formativa,

    -- De docentes
    d.usuario_documento,
    d.nombre_completo,
    d.nivelnexus,
    d.nombre_dre,
    d.nombre_ugel,
    d.fecha_nacimiento,
    d.es_focalizado,
    d.edad

FROM inscripciones_cursos ic
JOIN ofertas_formativas    of ON ic.id_oferta_formativa = of.id_oferta_formativa
JOIN docentes              d  ON ic.usuario_documento   = d.usuario_documento
WHERE of.activo = 1;

COMMENT ON VIEW v_inscripciones_recomendacion IS
    'Vista plana para el sistema de recomendación. '
    'Usar DB_TABLE_RECOMENDACION=v_inscripciones_recomendacion en .env. '
    'Columna CALIFICACIONES (0-5) = rating del CURSO, no del docente.';


-- =====================================================================
-- TABLA 4: conversaciones_agente
-- Log de conversaciones del agente (creado automáticamente por app.py)
-- =====================================================================

CREATE TABLE IF NOT EXISTS conversaciones_agente (
    id              SERIAL PRIMARY KEY,
    usuario         VARCHAR(100) NOT NULL,
    nombre_usuario  VARCHAR(200),
    mensaje         TEXT NOT NULL,
    respuesta       TEXT NOT NULL,
    tarea           VARCHAR(20) NOT NULL,   -- 'sifods' | 'recomendacion'
    fuente_datos    VARCHAR(50),
    tokens_entrada  INTEGER,
    tokens_salida   INTEGER,
    latencia_ms     INTEGER,
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_conv_usuario   ON conversaciones_agente(usuario);
CREATE INDEX idx_conv_tarea     ON conversaciones_agente(tarea);
CREATE INDEX idx_conv_timestamp ON conversaciones_agente(timestamp DESC);


-- =====================================================================
-- TABLA 5: metricas
-- Métricas diarias del agente (creado automáticamente por app.py)
-- =====================================================================

CREATE TABLE IF NOT EXISTS metricas (
    id                   SERIAL PRIMARY KEY,
    fecha                DATE NOT NULL,
    tarea                VARCHAR(20) NOT NULL,
    total_consultas      INTEGER DEFAULT 0,
    latencia_promedio_ms INTEGER,
    tokens_totales       INTEGER,
    fecha_actualizacion  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(fecha, tarea)
);


-- =====================================================================
-- TRIGGER: actualizar updated_at automáticamente
-- =====================================================================

CREATE OR REPLACE FUNCTION fn_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_of_updated_at
    BEFORE UPDATE ON ofertas_formativas
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

CREATE TRIGGER trg_ic_updated_at
    BEFORE UPDATE ON inscripciones_cursos
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();

CREATE TRIGGER trg_docentes_updated_at
    BEFORE UPDATE ON docentes
    FOR EACH ROW EXECUTE FUNCTION fn_set_updated_at();


-- =====================================================================
-- VISTAS DE MONITOREO
-- =====================================================================

-- Top cursos por popularidad (refleja la lógica del recomendador)
CREATE OR REPLACE VIEW v_cursos_populares AS
SELECT
    of.id_oferta_formativa,
    of.curso,
    of.horas_programa,
    of.calificaciones          AS calificacion_curso,
    of.publico_objetivo,
    COUNT(ic.inscripcion_id)   AS total_inscritos,
    SUM(ic.curso_culminado)    AS total_culminados,
    SUM(ic.aprobacion)         AS total_aprobados,
    ROUND(
        SUM(ic.curso_culminado)::NUMERIC / NULLIF(COUNT(*), 0), 2
    )                          AS tasa_culminacion,
    ROUND(
        -- Score de popularidad: mismo cálculo que Python
        (SUM(ic.curso_culminado)::NUMERIC / NULLIF(COUNT(*), 0))
        * (COALESCE(of.calificaciones, 0) / 5.0)
        * LN(COUNT(*) + 1),
    4)                         AS score_popularidad
FROM ofertas_formativas    of
JOIN inscripciones_cursos  ic ON ic.id_oferta_formativa = of.id_oferta_formativa
WHERE of.activo = 1
GROUP BY of.id_oferta_formativa, of.curso, of.horas_programa,
         of.calificaciones, of.publico_objetivo
ORDER BY score_popularidad DESC;


-- Docentes más activos
CREATE OR REPLACE VIEW v_docentes_activos AS
SELECT
    d.usuario_documento,
    d.nombre_completo,
    d.nivelnexus,
    d.nombre_dre,
    COUNT(ic.inscripcion_id)  AS total_cursos,
    SUM(ic.curso_culminado)   AS cursos_culminados,
    ROUND(SUM(ic.curso_culminado)::NUMERIC / NULLIF(COUNT(*), 0), 2) AS tasa_completitud
FROM docentes              d
JOIN inscripciones_cursos  ic ON ic.usuario_documento = d.usuario_documento
GROUP BY d.usuario_documento, d.nombre_completo, d.nivelnexus, d.nombre_dre
ORDER BY cursos_culminados DESC;


-- Métricas del agente por día
CREATE OR REPLACE VIEW v_metricas_diarias AS
SELECT
    DATE(timestamp)               AS fecha,
    tarea,
    COUNT(*)                      AS consultas,
    ROUND(AVG(latencia_ms))       AS latencia_promedio_ms,
    SUM(tokens_entrada + COALESCE(tokens_salida, 0)) AS tokens_totales
FROM conversaciones_agente
GROUP BY DATE(timestamp), tarea
ORDER BY fecha DESC, tarea;


-- =====================================================================
-- DATOS DE EJEMPLO (borrar en producción)
-- =====================================================================

-- Docentes de ejemplo
INSERT INTO docentes (usuario_documento, nombre_completo, nivelnexus, nombre_dre, nombre_ugel, edad)
VALUES
  ('12345678', 'María García Flores',   'PRIMARIA',   'DRE LIMA',     'UGEL 01',  38),
  ('87654321', 'Juan Pérez Torres',     'SECUNDARIA', 'DRE LIMA',     'UGEL 02',  45),
  ('11223344', 'Rosa Mamani Quispe',    'PRIMARIA',   'DRE AREQUIPA', 'UGEL AQP', 32),
  ('44332211', 'Carlos Huanca López',   'INICIAL',    'DRE CUSCO',    'UGEL CUS', 29)
ON CONFLICT DO NOTHING;

-- Ofertas formativas de ejemplo
INSERT INTO ofertas_formativas (curso, tipo_constancia, horas_programa, proposito, publico_objetivo, activo, año, calificaciones)
VALUES
  ('Estrategias Didácticas en Matemática', 'Certificado', 40, 'Mejorar enseñanza de matemáticas', 'PRIMARIA',   1, 2024, 4.2),
  ('Evaluación Formativa',                 'Diploma',     30, 'Implementar evaluación formativa',  'TODOS',      1, 2024, 4.5),
  ('TIC en el Aula',                       'Certificado', 35, 'Integrar TIC en la enseñanza',      'SECUNDARIA', 1, 2024, 3.8),
  ('Comprensión Lectora',                  'Certificado', 45, 'Mejorar comprensión lectora',       'PRIMARIA',   1, 2023, 4.0),
  ('Enseñanza de Ciencias Naturales',      'Diploma',     50, 'Metodologías activas en ciencias',  'SECUNDARIA', 1, 2023, 4.3)
ON CONFLICT DO NOTHING;

-- Inscripciones de ejemplo
INSERT INTO inscripciones_cursos (usuario_documento, id_oferta_formativa, aprobacion, curso_culminado, porcentaje_avance, año)
SELECT '12345678', id_oferta_formativa, 1, 1, 100, 2024
FROM ofertas_formativas WHERE curso = 'Evaluación Formativa'
ON CONFLICT DO NOTHING;

INSERT INTO inscripciones_cursos (usuario_documento, id_oferta_formativa, aprobacion, curso_culminado, porcentaje_avance, año)
SELECT '87654321', id_oferta_formativa, 1, 1, 100, 2024
FROM ofertas_formativas WHERE curso = 'TIC en el Aula'
ON CONFLICT DO NOTHING;


-- =====================================================================
-- GRANTS (descomentar y ajustar según roles en producción)
-- =====================================================================

-- CREATE ROLE agente_tecnologico_role;
-- GRANT SELECT ON v_inscripciones_recomendacion TO agente_tecnologico_role;
-- GRANT SELECT, INSERT ON conversaciones_agente  TO agente_tecnologico_role;
-- GRANT SELECT, INSERT ON metricas               TO agente_tecnologico_role;
-- GRANT SELECT ON v_cursos_populares             TO agente_tecnologico_role;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agente_tecnologico_role;