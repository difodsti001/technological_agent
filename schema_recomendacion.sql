-- =====================================================
-- 📊 SCHEMA COMPLETO PARA SISTEMA DE RECOMENDACIÓN
-- =====================================================
-- Base de Datos: agente_tecnologico
-- Propósito: Almacenar datos para filtro colaborativo híbrido
-- =====================================================

-- Habilitar extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =====================================================
-- TABLA 1: cursos
-- Catálogo completo de cursos disponibles en SIFODS
-- =====================================================

CREATE TABLE IF NOT EXISTS cursos (
    -- Identificación
    curso_id SERIAL PRIMARY KEY,
    codigo_curso VARCHAR(20) UNIQUE NOT NULL,  -- Ej: "MAT-2024-001"
    nombre VARCHAR(255) NOT NULL,
    descripcion TEXT,
    
    -- Clasificación
    categoria VARCHAR(50) NOT NULL,  -- matemáticas, evaluación, didáctica, tic, etc.
    subcategoria VARCHAR(50),
    nivel_dificultad VARCHAR(20) NOT NULL,  -- basico, intermedio, avanzado
    
    -- Contenido
    duracion_horas INTEGER NOT NULL,
    total_modulos INTEGER DEFAULT 0,
    total_actividades INTEGER DEFAULT 0,
    
    -- Alcance
    area_curricular VARCHAR(50),  -- matemáticas, comunicación, ciencias, sociales, etc.
    nivel_educativo VARCHAR[] DEFAULT '{}',  -- {inicial, primaria, secundaria}
    region_enfoque VARCHAR[] DEFAULT '{todas}',  -- {costa, sierra, selva, todas}
    modalidad VARCHAR(20) DEFAULT 'virtual',  -- virtual, presencial, hibrido
    
    -- Metadata
    fecha_creacion DATE NOT NULL DEFAULT CURRENT_DATE,
    fecha_actualizacion DATE,
    fecha_inicio DATE,
    fecha_fin DATE,
    
    -- Requisitos
    cursos_prerequisito INTEGER[] DEFAULT '{}',  -- IDs de cursos que debe tomar antes
    competencias_previas VARCHAR[] DEFAULT '{}',
    
    -- Estadísticas (se actualizan con triggers)
    total_inscritos INTEGER DEFAULT 0,
    total_completados INTEGER DEFAULT 0,
    calificacion_promedio REAL DEFAULT 0.0,
    tasa_completitud REAL DEFAULT 0.0,  -- total_completados / total_inscritos
    
    -- Clasificación semántica
    tags VARCHAR[] DEFAULT '{}',  -- {evaluación, competencias, retroalimentación, etc}
    palabras_clave TEXT[] DEFAULT '{}',
    
    -- Control
    estado VARCHAR(20) DEFAULT 'activo',  -- activo, inactivo, archivado, borrador
    cupo_maximo INTEGER,
    requiere_aprobacion BOOLEAN DEFAULT FALSE,
    
    -- Auditoría
    creado_por UUID,
    actualizado_por UUID,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para cursos
CREATE INDEX idx_cursos_categoria ON cursos(categoria);
CREATE INDEX idx_cursos_nivel ON cursos(nivel_dificultad);
CREATE INDEX idx_cursos_area ON cursos(area_curricular);
CREATE INDEX idx_cursos_estado ON cursos(estado);
CREATE INDEX idx_cursos_fecha_creacion ON cursos(fecha_creacion DESC);
CREATE INDEX idx_cursos_tags ON cursos USING GIN(tags);
CREATE INDEX idx_cursos_nivel_educativo ON cursos USING GIN(nivel_educativo);

-- =====================================================
-- TABLA 2: inscripciones
-- Historial de inscripciones y avances de usuarios
-- =====================================================

CREATE TABLE IF NOT EXISTS inscripciones (
    -- Identificación
    inscripcion_id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,  -- FK a usuarios
    curso_id INTEGER NOT NULL REFERENCES cursos(curso_id) ON DELETE CASCADE,
    
    -- Fechas
    fecha_inscripcion TIMESTAMPTZ DEFAULT NOW(),
    fecha_inicio TIMESTAMPTZ,  -- Cuando comenzó a interactuar
    fecha_completado TIMESTAMPTZ,
    fecha_vencimiento DATE,
    
    -- Progreso
    porcentaje_avance REAL DEFAULT 0.0,  -- 0.0 a 100.0
    modulos_completados INTEGER DEFAULT 0,
    actividades_completadas INTEGER DEFAULT 0,
    tiempo_total_minutos INTEGER DEFAULT 0,
    ultimo_acceso TIMESTAMPTZ,
    
    -- Calificaciones
    calificacion_final REAL,  -- 0 a 20
    calificacion_promedio_actividades REAL,
    total_intentos INTEGER DEFAULT 0,
    aprobado BOOLEAN,
    
    -- Rating del usuario
    rating_usuario INTEGER CHECK (rating_usuario >= 1 AND rating_usuario <= 5),
    comentario_usuario TEXT,
    fecha_rating TIMESTAMPTZ,
    
    -- Engagement
    numero_accesos INTEGER DEFAULT 0,
    dias_activo INTEGER DEFAULT 0,
    sesiones_estudio INTEGER DEFAULT 0,
    
    -- Estado
    estado VARCHAR(20) DEFAULT 'en_curso',  -- inscrito, en_curso, completado, abandonado, reprobado
    razon_abandono VARCHAR(100),
    
    -- Certificación
    certificado_generado BOOLEAN DEFAULT FALSE,
    fecha_certificado DATE,
    
    -- Metadata
    fuente_inscripcion VARCHAR(50),  -- recomendacion_agente, busqueda, obligatorio, etc
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    UNIQUE(user_id, curso_id)
);

-- Índices para inscripciones
CREATE INDEX idx_inscripciones_user ON inscripciones(user_id);
CREATE INDEX idx_inscripciones_curso ON inscripciones(curso_id);
CREATE INDEX idx_inscripciones_estado ON inscripciones(estado);
CREATE INDEX idx_inscripciones_completado ON inscripciones(fecha_completado) WHERE fecha_completado IS NOT NULL;
CREATE INDEX idx_inscripciones_rating ON inscripciones(rating_usuario) WHERE rating_usuario IS NOT NULL;

-- =====================================================
-- TABLA 3: usuarios_perfil
-- Extensión de la tabla usuarios con datos para recomendación
-- =====================================================

CREATE TABLE IF NOT EXISTS usuarios_perfil (
    user_id UUID PRIMARY KEY,
    
    -- Información demográfica
    nivel_educativo VARCHAR(20),  -- inicial, primaria, secundaria
    area_especialidad VARCHAR(50),  -- matemáticas, comunicación, etc.
    region VARCHAR(50),  -- costa, sierra, selva
    departamento VARCHAR(50),
    tipo_institucion VARCHAR(30),  -- publica, privada
    zona VARCHAR(20),  -- urbana, rural
    
    -- Experiencia
    anos_experiencia INTEGER,
    anos_en_sifods INTEGER DEFAULT 0,
    total_cursos_completados INTEGER DEFAULT 0,
    total_cursos_en_progreso INTEGER DEFAULT 0,
    
    -- Preferencias explícitas
    preferencias JSONB DEFAULT '{}',
    /*
    Ejemplo de preferencias:
    {
        "areas_interes": ["evaluación", "tic", "didáctica"],
        "nivel_preferido": "intermedio",
        "duracion_preferida": "corta",  // corta (<20h), media (20-40h), larga (>40h)
        "modalidad_preferida": "virtual",
        "temas_evitar": ["estadística"],
        "horario_estudio": "noche",
        "ritmo_aprendizaje": "rapido"  // lento, normal, rapido
    }
    */
    
    -- Historial agregado
    calificacion_promedio_historica REAL,
    tasa_completitud REAL DEFAULT 0.0,
    tiempo_promedio_por_curso INTEGER,  -- minutos
    
    -- Objetivos
    objetivo_formacion TEXT[] DEFAULT '{}',  -- {ascenso, mejora_practica, certificacion}
    metas_corto_plazo VARCHAR[] DEFAULT '{}',
    
    -- Contexto
    grados_atiende INTEGER[] DEFAULT '{}',  -- {1, 2, 3, 4, 5, 6}
    numero_estudiantes INTEGER,
    recursos_disponibles VARCHAR[] DEFAULT '{}',  -- {internet, computadora, proyector, etc}
    
    -- Engagement
    dias_desde_ultimo_curso INTEGER,
    frecuencia_uso VARCHAR(20),  -- diaria, semanal, mensual, esporadica
    
    -- Control
    perfil_completado BOOLEAN DEFAULT FALSE,
    fecha_ultima_actualizacion TIMESTAMPTZ DEFAULT NOW(),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para usuarios_perfil
CREATE INDEX idx_perfil_nivel ON usuarios_perfil(nivel_educativo);
CREATE INDEX idx_perfil_area ON usuarios_perfil(area_especialidad);
CREATE INDEX idx_perfil_region ON usuarios_perfil(region);
CREATE INDEX idx_perfil_experiencia ON usuarios_perfil(anos_experiencia);

-- =====================================================
-- TABLA 4: interacciones_cursos
-- Eventos de interacción para análisis fino
-- =====================================================

CREATE TABLE IF NOT EXISTS interacciones_cursos (
    interaccion_id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    curso_id INTEGER NOT NULL REFERENCES cursos(curso_id) ON DELETE CASCADE,
    
    -- Tipo de interacción
    tipo_evento VARCHAR(30) NOT NULL,
    /*
    Tipos de evento:
    - vista: Vio la página del curso
    - inscripcion: Se inscribió
    - inicio_modulo: Comenzó un módulo
    - completado_actividad: Completó una actividad
    - calificacion: Recibió calificación
    - acceso_recurso: Descargó/accedió a un recurso
    - participacion_foro: Participó en foro
    - abandono: Dejó de acceder por >30 días
    */
    
    -- Detalles del evento
    modulo_id INTEGER,
    actividad_id INTEGER,
    recurso_id INTEGER,
    
    -- Metadata del evento
    duracion_segundos INTEGER,
    resultado VARCHAR(20),  -- aprobado, desaprobado, en_progreso
    calificacion REAL,
    
    -- Contexto
    dispositivo VARCHAR(20),  -- desktop, mobile, tablet
    navegador VARCHAR(20),
    
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para interacciones
CREATE INDEX idx_interacciones_user ON interacciones_cursos(user_id);
CREATE INDEX idx_interacciones_curso ON interacciones_cursos(curso_id);
CREATE INDEX idx_interacciones_tipo ON interacciones_cursos(tipo_evento);
CREATE INDEX idx_interacciones_timestamp ON interacciones_cursos(timestamp DESC);

-- Particionamiento por fecha (opcional, para grandes volúmenes)
-- CREATE TABLE interacciones_cursos_2024_01 PARTITION OF interacciones_cursos
-- FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- =====================================================
-- TABLA 5: similitud_cursos_precalculada
-- Matriz de similitud entre cursos (para performance)
-- =====================================================

CREATE TABLE IF NOT EXISTS similitud_cursos_precalculada (
    curso_a_id INTEGER NOT NULL REFERENCES cursos(curso_id) ON DELETE CASCADE,
    curso_b_id INTEGER NOT NULL REFERENCES cursos(curso_id) ON DELETE CASCADE,
    
    -- Scores de similitud
    similitud_contenido REAL NOT NULL,  -- Basado en features del curso
    similitud_colaborativa REAL,  -- Basado en usuarios que tomaron ambos
    
    -- Metadata
    fecha_calculo TIMESTAMPTZ DEFAULT NOW(),
    
    PRIMARY KEY (curso_a_id, curso_b_id),
    CHECK (curso_a_id < curso_b_id)  -- Evitar duplicados (a,b) y (b,a)
);

CREATE INDEX idx_similitud_curso_a ON similitud_cursos_precalculada(curso_a_id);
CREATE INDEX idx_similitud_curso_b ON similitud_cursos_precalculada(curso_b_id);

-- =====================================================
-- TABLA 6: recomendaciones_generadas
-- Cache de recomendaciones ya calculadas
-- =====================================================

CREATE TABLE IF NOT EXISTS recomendaciones_generadas (
    recomendacion_id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    
    -- Recomendaciones (JSON array)
    cursos_recomendados JSONB NOT NULL,
    /*
    Formato:
    [
        {
            "curso_id": 123,
            "score": 0.87,
            "metodos": ["colaborativo", "contenido"],
            "justificacion": "..."
        }
    ]
    */
    
    -- Metadata
    fecha_generacion TIMESTAMPTZ DEFAULT NOW(),
    fecha_expiracion TIMESTAMPTZ,  -- Se regenera después de X días
    version_algoritmo VARCHAR(10),
    
    -- Engagement con recomendaciones
    cursos_vistos INTEGER[] DEFAULT '{}',
    cursos_inscritos INTEGER[] DEFAULT '{}',
    tasa_aceptacion REAL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_recomendaciones_user ON recomendaciones_generadas(user_id);
CREATE INDEX idx_recomendaciones_vigentes ON recomendaciones_generadas(fecha_expiracion) 
    WHERE fecha_expiracion > NOW();

-- =====================================================
-- TABLA 7: feedback_recomendaciones
-- Feedback explícito del usuario sobre recomendaciones
-- =====================================================

CREATE TABLE IF NOT EXISTS feedback_recomendaciones (
    feedback_id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    curso_id INTEGER NOT NULL REFERENCES cursos(curso_id) ON DELETE CASCADE,
    recomendacion_id INTEGER REFERENCES recomendaciones_generadas(recomendacion_id),
    
    -- Tipo de feedback
    tipo_feedback VARCHAR(20) NOT NULL,
    /*
    Tipos:
    - util: Marcó como útil
    - no_util: Marcó como no útil
    - no_relevante: No es relevante para él
    - ya_tomado: Ya tomó este curso
    - interesado: Le interesa
    */
    
    -- Comentario opcional
    comentario TEXT,
    
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_feedback_user ON feedback_recomendaciones(user_id);
CREATE INDEX idx_feedback_curso ON feedback_recomendaciones(curso_id);
CREATE INDEX idx_feedback_tipo ON feedback_recomendaciones(tipo_feedback);

-- =====================================================
-- TRIGGERS Y FUNCIONES
-- =====================================================

-- Función para actualizar estadísticas de curso
CREATE OR REPLACE FUNCTION actualizar_estadisticas_curso()
RETURNS TRIGGER AS $$
BEGIN
    -- Actualizar total de inscritos
    UPDATE cursos
    SET 
        total_inscritos = (
            SELECT COUNT(*) 
            FROM inscripciones 
            WHERE curso_id = NEW.curso_id
        ),
        total_completados = (
            SELECT COUNT(*) 
            FROM inscripciones 
            WHERE curso_id = NEW.curso_id 
              AND fecha_completado IS NOT NULL
        ),
        calificacion_promedio = (
            SELECT COALESCE(AVG(calificacion_final), 0)
            FROM inscripciones
            WHERE curso_id = NEW.curso_id
              AND calificacion_final IS NOT NULL
        ),
        tasa_completitud = (
            SELECT 
                CASE 
                    WHEN COUNT(*) > 0 THEN 
                        (COUNT(*) FILTER (WHERE fecha_completado IS NOT NULL)::FLOAT / COUNT(*))
                    ELSE 0
                END
            FROM inscripciones
            WHERE curso_id = NEW.curso_id
        ),
        updated_at = NOW()
    WHERE curso_id = NEW.curso_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger en inscripciones
CREATE TRIGGER trigger_actualizar_stats_curso
AFTER INSERT OR UPDATE ON inscripciones
FOR EACH ROW
EXECUTE FUNCTION actualizar_estadisticas_curso();

-- Función para actualizar perfil de usuario
CREATE OR REPLACE FUNCTION actualizar_perfil_usuario()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO usuarios_perfil (user_id)
    VALUES (NEW.user_id)
    ON CONFLICT (user_id) DO UPDATE
    SET
        total_cursos_completados = (
            SELECT COUNT(*)
            FROM inscripciones
            WHERE user_id = NEW.user_id
              AND fecha_completado IS NOT NULL
        ),
        total_cursos_en_progreso = (
            SELECT COUNT(*)
            FROM inscripciones
            WHERE user_id = NEW.user_id
              AND estado = 'en_curso'
        ),
        calificacion_promedio_historica = (
            SELECT AVG(calificacion_final)
            FROM inscripciones
            WHERE user_id = NEW.user_id
              AND calificacion_final IS NOT NULL
        ),
        tasa_completitud = (
            SELECT 
                CASE 
                    WHEN COUNT(*) > 0 THEN
                        (COUNT(*) FILTER (WHERE fecha_completado IS NOT NULL)::FLOAT / COUNT(*))
                    ELSE 0
                END
            FROM inscripciones
            WHERE user_id = NEW.user_id
        ),
        dias_desde_ultimo_curso = (
            SELECT 
                CASE 
                    WHEN MAX(fecha_completado) IS NOT NULL THEN
                        EXTRACT(DAY FROM NOW() - MAX(fecha_completado))::INTEGER
                    ELSE NULL
                END
            FROM inscripciones
            WHERE user_id = NEW.user_id
        ),
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger en inscripciones para perfil
CREATE TRIGGER trigger_actualizar_perfil
AFTER INSERT OR UPDATE ON inscripciones
FOR EACH ROW
EXECUTE FUNCTION actualizar_perfil_usuario();

-- =====================================================
-- VISTAS ÚTILES
-- =====================================================

-- Vista: Cursos populares
CREATE OR REPLACE VIEW v_cursos_populares AS
SELECT 
    c.*,
    ROUND(
        (
            (c.total_completados::FLOAT / NULLIF(c.total_inscritos, 0)) * 
            c.calificacion_promedio * 
            LN(c.total_inscritos + 1)
        )::numeric,
        2
    ) as score_popularidad
FROM cursos c
WHERE c.estado = 'activo'
  AND c.total_inscritos > 5
ORDER BY score_popularidad DESC;

-- Vista: Cursos nuevos trending
CREATE OR REPLACE VIEW v_cursos_trending AS
SELECT 
    c.*,
    EXTRACT(DAY FROM NOW() - c.fecha_creacion) as dias_desde_creacion,
    ROUND(
        (
            c.calificacion_promedio * 
            (1 - (EXTRACT(DAY FROM NOW() - c.fecha_creacion) / 90.0))
        )::numeric,
        2
    ) as score_novedad
FROM cursos c
WHERE c.estado = 'activo'
  AND c.fecha_creacion >= NOW() - INTERVAL '90 days'
ORDER BY score_novedad DESC;

-- Vista: Matriz usuario-curso simplificada
CREATE OR REPLACE VIEW v_matriz_usuario_curso AS
SELECT 
    i.user_id,
    i.curso_id,
    COALESCE(
        i.rating_usuario,
        CASE 
            WHEN i.fecha_completado IS NOT NULL THEN 4.0
            WHEN i.porcentaje_avance > 50 THEN 3.0
            ELSE 2.0
        END
    ) as rating_implicito
FROM inscripciones i;

-- =====================================================
-- DATOS DE EJEMPLO (PARA TESTING)
-- =====================================================

-- Insertar cursos de ejemplo
INSERT INTO cursos (codigo_curso, nombre, descripcion, categoria, nivel_dificultad, duracion_horas, area_curricular, nivel_educativo, region_enfoque, tags) VALUES
('MAT-001', 'Estrategias Didácticas en Matemática', 'Aprende estrategias innovadoras para enseñar matemáticas', 'didáctica', 'intermedio', 40, 'matemáticas', '{primaria, secundaria}', '{todas}', '{didáctica, matemáticas, estrategias}'),
('EVA-001', 'Evaluación Formativa', 'Herramientas para implementar evaluación formativa efectiva', 'evaluación', 'básico', 30, 'general', '{inicial, primaria, secundaria}', '{todas}', '{evaluación, retroalimentación, formativa}'),
('TIC-001', 'TIC en el Aula', 'Integración de tecnologías en la enseñanza', 'tecnología', 'intermedio', 35, 'general', '{primaria, secundaria}', '{todas}', '{tic, tecnología, digital}'),
('COM-001', 'Comprensión Lectora', 'Estrategias para mejorar la comprensión lectora', 'didáctica', 'intermedio', 45, 'comunicación', '{primaria}', '{todas}', '{lectura, comprensión, comunicación}'),
('CIE-001', 'Enseñanza de Ciencias Naturales', 'Metodologías activas para ciencias', 'didáctica', 'avanzado', 50, 'ciencias', '{secundaria}', '{todas}', '{ciencias, metodología, indagación}');

-- =====================================================
-- COMENTARIOS Y DOCUMENTACIÓN
-- =====================================================

COMMENT ON TABLE cursos IS 'Catálogo completo de cursos disponibles en SIFODS';
COMMENT ON TABLE inscripciones IS 'Historial de inscripciones y progreso de usuarios en cursos';
COMMENT ON TABLE usuarios_perfil IS 'Perfil extendido de usuarios para sistema de recomendación';
COMMENT ON TABLE interacciones_cursos IS 'Log detallado de interacciones para análisis fino';
COMMENT ON TABLE similitud_cursos_precalculada IS 'Matriz de similitud entre cursos (precalculada para performance)';
COMMENT ON TABLE recomendaciones_generadas IS 'Cache de recomendaciones ya calculadas por usuario';
COMMENT ON TABLE feedback_recomendaciones IS 'Feedback explícito del usuario sobre recomendaciones';

-- =====================================================
-- GRANTS (ajustar según roles)
-- =====================================================

-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO agente_tecnologico_role;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agente_tecnologico_role;
