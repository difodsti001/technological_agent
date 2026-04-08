# 🤖 Agente Tecnológico DIFODS

Asistente especializado en navegación de la plataforma SIFODS y recomendación de cursos para docentes del Ministerio de Educación del Perú.

## 🚀 Características

- **Navegación SIFODS**: Sistema RAG (Retrieval-Augmented Generation) para consultas sobre la plataforma educativa
- **Recomendación de Cursos**: Motor híbrido de recomendación basado en filtrado colaborativo, popularidad, historial y novedad
- **APIs REST**: Endpoints FastAPI para integración con otros sistemas
- **Configuración Flexible**: Parámetros ajustables vía variables de entorno
- **Fallback Inteligente**: Soporte para múltiples LLMs (OpenAI GPT, Google Gemini)
- **Base de Datos**: PostgreSQL con fallback a Excel

## 📋 Requisitos del Sistema

- **Python**: 3.12+
- **PostgreSQL**: 12+ (opcional, usa Excel como fallback)
- **Qdrant**: Vector database (localhost o cloud)
- **APIs**: OpenAI API key y/o Google Gemini API key

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd agente_tecnologico
```

### 2. Crear entorno virtual
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno
```bash
cp .env.example .env
# Editar .env con tus credenciales reales
```

## ⚙️ Configuración Completa

### Variables de Entorno (.env)

Todas las variables de entorno soportadas por el sistema:

#### 🤖 APIs de LLM
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `OPENAI_API_KEY` | *(requerido)* | API Key de OpenAI para GPT models |
| `GEMINI_API_KEY` | *(requerido)* | API Key de Google Gemini |
| `LLM_PRINCIPAL` | `gpt-4o-mini` | Modelo principal de OpenAI |
| `LLM_FALLBACK` | `gemini-2.5-flash` | Modelo fallback de Gemini |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Modelo de embeddings para RAG |

#### 🗄️ Base de Datos Qdrant
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | URL del servidor Qdrant |
| `QDRANT_API_KEY` | *(opcional)* | API Key para Qdrant cloud |
| `QDRANT_COLLECTION` | `Curso_0` | Nombre de la colección en Qdrant |
| `QDRANT_TOP_K` | `10` | Número máximo de documentos a recuperar |

#### 🎓 Moodle API
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `MOODLE_API_URL` | `https://campusvirtual-sifods.minedu.gob.pe/webservice/rest/server.php` | URL base de la API Moodle |
| `MOODLE_API_TOKEN` | *(requerido)* | Token de autenticación Moodle |

#### 🐘 Base de Datos PostgreSQL
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `DB_HOST` | `localhost` | Host del servidor PostgreSQL |
| `DB_PORT` | `5432` | Puerto del servidor PostgreSQL |
| `DB_NAME` | `agente_tecnologico` | Nombre de la base de datos |
| `DB_USER` | `postgres` | Usuario de PostgreSQL |
| `DB_PASSWORD` | *(requerido)* | Contraseña de PostgreSQL |
| `DB_SCHEMA` | `public` | Schema de PostgreSQL |
| `DB_POOL_MIN` | `2` | Conexiones mínimas en el pool |
| `DB_POOL_MAX` | `10` | Conexiones máximas en el pool |
| `DB_TABLE_RECOMENDACION` | `data_recomendacion` | Tabla con datos de recomendación |

#### 📊 Fallback Excel
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `EXCEL_FALLBACK_PATH` | `data/DATA_RECOM.xlsx` | Ruta al archivo Excel de respaldo |
| `EXCEL_SHEET_NAME` | `Sheet1` | Nombre de la hoja en el Excel |

#### 🎯 Sistema de Recomendación
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `REC_TOP_K` | `3` | Número de cursos a recomendar por defecto |
| `REC_MIN_SCORE` | `0.05` | Score mínimo para incluir un curso |
| `REC_TOP_K_SIMILAR_USERS` | `30` | Usuarios similares en filtro colaborativo |
| `REC_MAX_USUARIOS_MATRIZ` | `15000` | Límite de usuarios en matriz (protección RAM) |
| `REC_MAX_LLM_WORKERS` | `4` | Threads paralelos para justificaciones LLM |
| `REC_PESO_COLABORATIVO` | `0.40` | Peso del filtro colaborativo (0.0-1.0) |
| `REC_PESO_POPULARIDAD` | `0.30` | Peso de la popularidad (0.0-1.0) |
| `REC_PESO_HISTORIAL` | `0.20` | Peso del historial del usuario (0.0-1.0) |
| `REC_PESO_NOVEDAD` | `0.10` | Peso de la novedad de cursos (0.0-1.0) |

#### 💬 Módulo SIFODS (RAG)
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `SIFODS_MAX_TOKENS` | `1500` | Máximo tokens en respuestas |
| `SIFODS_TEMPERATURE` | `0.50` | Creatividad del modelo (0.0-1.0) |
| `SIFODS_TOP_P` | `0.9` | Nucleus sampling (0.0-1.0) |

#### 🤖 LLM Justificaciones
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `JUSTIF_MAX_TOKENS` | `250` | Máximo tokens en justificaciones |
| `JUSTIF_TEMPERATURE` | `0.75` | Creatividad para justificaciones (0.0-1.0) |

#### 🌐 Servidor
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `PORT` | `7002` | Puerto del servidor FastAPI |
| `LOG_LEVEL` | `INFO` | Nivel de logging (DEBUG, INFO, WARNING, ERROR) |

#### 💾 Persistencia
| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `GUARDAR_CONVERSACIONES` | `true` | Guardar historial de conversaciones |
| `GUARDAR_METRICAS` | `true` | Guardar métricas de uso |
| `CACHE_TTL_SEGUNDOS` | `21600` | TTL del cache en segundos (6 horas) |

### Archivo .env de Ejemplo

```bash
# Copiar desde .env.example y configurar
cp .env.example .env

# Variables críticas (requeridas):
OPENAI_API_KEY=sk-proj-tu-api-key-aqui
GEMINI_API_KEY=AIzaSy-tu-api-key-aqui
DB_PASSWORD=tu_password_postgres
MOODLE_API_TOKEN=tu_token_moodle

# El resto mantiene valores por defecto razonables
```

### Base de Datos

Ejecutar el script SQL incluido:
```bash
psql -U postgres -d agente_tecnologico -f schema_recomendacion.sql
```

## 🚀 Uso

### Ejecutar la aplicación
```bash
# Opción 1: Directamente
python app.py

# Opción 2: Con uvicorn
uvicorn app:app --reload --port 7002
```

### Acceder a la interfaz
- **Frontend**: http://localhost:7002
- **API Docs**: http://localhost:7002/docs
- **Health Check**: http://localhost:7002/health

### Ejemplos de API

#### Consulta SIFODS
```bash
curl -X POST "http://localhost:7002/api/sifods" \
  -H "Content-Type: application/json" \
  -d '{
    "mensaje": "¿Cómo acceder al Centro de Recursos?",
    "usuario": "usuario123",
    "nombre_usuario": "Juan Pérez"
  }'
```

#### Recomendación de Cursos
```bash
curl -X POST "http://localhost:7002/api/recomendar" \
  -H "Content-Type: application/json" \
  -d '{
    "usuario": "usuario123",
    "nombre_usuario": "Juan Pérez",
    "top_k": 5
  }'
```

## 📁 Estructura del Proyecto

```
agente_tecnologico/
│
├── .env                          ← Variables de entorno (NO commitear)
├── .env.example                  ← Plantilla de .env (sí commitear)
│
├── app.py                        ← FastAPI: endpoints, orquestación
├── sistema_recomendacion.py      ← Motor de recomendación híbrido
│
├── config/
│   ├── __init__.py               ← Exporta `settings`
│   ├── settings.py               ← ⭐ ÚNICA fuente de verdad para parámetros
│   └── prompts.py                ← Todos los prompts del sistema
│
├── templates/
│   └── agente_tecnologico.html   ← Frontend HTML
│
├── data/                         ← (opcional) Excel de fallback
│   └── inscripciones.xlsx
│
└── schema_recomendacion.sql      ← Schema de PostgreSQL
```

## 🔧 Configuración Avanzada

### Flujo de Configuración

```
.env
 └─→ config/settings.py (Settings dataclasses)
       ├─→ app.py               (usa settings.*)
       ├─→ sistema_recomendacion.py (usa settings.recomendacion, settings.llm)
       └─→ config/prompts.py    (plantillas de texto, sin parámetros)
```

**Regla**: ningún archivo fuera de `config/settings.py` debe leer `os.getenv()` directamente.

### Parámetros de Recomendación

| Variable              | Default  | Descripción                               |
|-----------------------|----------|-------------------------------------------|
| `REC_TOP_K`           | `3`      | Cursos a devolver por recomendación       |
| `REC_MIN_SCORE`       | `0.05`   | Score mínimo para incluir un curso        |
| `REC_PESO_COLABORATIVO` | `0.40` | Peso del filtro colaborativo (suma = 1.0) |
| `REC_PESO_POPULARIDAD`  | `0.30` | Peso de popularidad                       |
| `REC_PESO_HISTORIAL`    | `0.20` | Peso del historial                        |
| `REC_PESO_NOVEDAD`      | `0.10` | Peso de novedad                           |

### Parámetros LLM

| Variable              | Default  | Descripción                               |
|-----------------------|----------|-------------------------------------------|
| `SIFODS_MAX_TOKENS`   | `1500`   | Max tokens en respuestas SIFODS           |
| `JUSTIF_MAX_TOKENS`   | `120`    | Max tokens en justificaciones LLM         |
| `LLM_PRINCIPAL`       | `gpt-4o-mini` | Modelo OpenAI                        |
| `LLM_FALLBACK`        | `gemini-2.5-flash` | Modelo fallback Gemini          |

## 🐛 Problemas Conocidos y Soluciones

### Bug Corregido: top_k Ignorado

**Problema original**: `obtener_recomendaciones()` en `sistema_recomendacion.py`
tenía `top_k=5` hardcodeado. `RecommenderConfig.TOP_K_RECOMENDACIONES = 3` existía
pero no se aplicaba como default real de `recomendar_hibrido()`.

**Solución**:
1. `recomendar_hibrido(top_k=None)` → si `None`, lee `settings.recomendacion.top_k`
2. `settings.recomendacion.top_k` lee `REC_TOP_K` del `.env` (default `3`)
3. El endpoint `/api/recomendar` acepta `top_k` opcional; si no viene, usa el settings
4. Eliminado el `top_k=5` hardcodeado de `obtener_recomendaciones()`

## 📡 API Endpoints

| Método | Path                                | Descripción                        |
|--------|-------------------------------------|------------------------------------|
| GET    | `/`                                 | Frontend HTML                      |
| GET    | `/health`                           | Estado del sistema                 |
| GET    | `/api/config`                       | Config activa (sin secrets)        |
| POST   | `/api/sifods`                       | Consultas sobre la plataforma SIFODS (RAG) |
| POST   | `/api/recomendar`                   | Recomendación de cursos            |
| POST   | `/api/admin/refrescar-recomendador` | Recarga datos sin restart          |

> No hay endpoint unificado ni clasificador — cada módulo tiene su endpoint dedicado.

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- Crear un issue en el repositorio
- Contactar al equipo de desarrollo DIFODS

