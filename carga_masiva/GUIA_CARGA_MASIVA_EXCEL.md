# 📊 GUÍA DE USO - CARGA MASIVA DESDE EXCEL

## 🎯 Descripción

Este módulo permite cargar datos de forma masiva desde archivos Excel a las tablas del sistema de recomendación del Agente Tecnológico.

---

## 📋 Instalación de Dependencias

```bash
pip install pandas openpyxl psycopg2-binary python-dotenv
```

---

## 🚀 Uso Rápido

### Opción 1: Menú Interactivo

```bash
python carga_masiva_excel.py
```

Te mostrará un menú con opciones:
1. Generar plantillas Excel vacías
2. Cargar cursos
3. Cargar perfiles de usuarios
4. Cargar inscripciones
5. Cargar TODO desde un directorio
6. Salir

### Opción 2: Uso Programático

```python
from carga_masiva_excel import (
    cargar_cursos_desde_excel,
    cargar_usuarios_perfil_desde_excel,
    cargar_inscripciones_desde_excel,
    cargar_todo_desde_directorio
)

# Cargar cursos
insertados, actualizados = cargar_cursos_desde_excel('datos/cursos.xlsx')

# Cargar todo desde un directorio
cargar_todo_desde_directorio('datos/')
```

---

## 📁 Estructura de Archivos Excel

### 1. **cursos.xlsx**

| Columna | Tipo | Requerido | Descripción | Ejemplo |
|---------|------|-----------|-------------|---------|
| codigo_curso | Texto | ✅ | Código único del curso | MAT-2024-001 |
| nombre | Texto | ✅ | Nombre del curso | Estrategias Didácticas en Matemática |
| descripcion | Texto | ⚪ | Descripción breve | Aprende estrategias innovadoras... |
| categoria | Texto | ✅ | Categoría del curso | didáctica, evaluación, tic |
| nivel_dificultad | Texto | ✅ | Nivel de dificultad | basico, intermedio, avanzado |
| duracion_horas | Número | ✅ | Duración en horas | 40 |
| area_curricular | Texto | ✅ | Área del curso | matemáticas, comunicación |
| nivel_educativo | Texto | ⚪ | Niveles separados por \| | primaria\|secundaria |
| region_enfoque | Texto | ⚪ | Regiones separadas por \| | costa\|sierra\|todas |
| fecha_creacion | Fecha | ⚪ | Fecha de creación | 2024-01-15 |
| tags | Texto | ⚪ | Tags separados por \| | didáctica\|matemáticas\|estrategias |
| estado | Texto | ⚪ | Estado del curso | activo, inactivo, archivado |

**Ejemplo de fila:**
```
MAT-2024-001 | Estrategias Didácticas en Matemática | Aprende estrategias... | didáctica | intermedio | 40 | matemáticas | primaria|secundaria | todas | 2024-01-15 | didáctica|matemáticas | activo
```

---

### 2. **usuarios_perfil.xlsx**

| Columna | Tipo | Requerido | Descripción | Ejemplo |
|---------|------|-----------|-------------|---------|
| user_id | Texto | ✅ | UUID del usuario | user-123-abc |
| nivel_educativo | Texto | ⚪ | Nivel que enseña | inicial, primaria, secundaria |
| area_especialidad | Texto | ⚪ | Área de especialidad | matemáticas, comunicación |
| region | Texto | ⚪ | Región del docente | costa, sierra, selva |
| departamento | Texto | ⚪ | Departamento | Lima, Cusco, Arequipa |
| anos_experiencia | Número | ⚪ | Años de experiencia | 5 |
| preferencias | JSON | ⚪ | Preferencias en JSON | {"areas_interes": ["didáctica"]} |

**Ejemplo de fila:**
```
user-123-abc | primaria | matemáticas | costa | Lima | 5 | {"areas_interes": ["didáctica", "evaluación"], "nivel_preferido": "intermedio"}
```

**Formato de preferencias (JSON):**
```json
{
  "areas_interes": ["didáctica", "evaluación", "tic"],
  "nivel_preferido": "intermedio",
  "duracion_preferida": "corta",
  "temas_evitar": ["estadística"]
}
```

---

### 3. **inscripciones.xlsx**

| Columna | Tipo | Requerido | Descripción | Ejemplo |
|---------|------|-----------|-------------|---------|
| user_id | Texto | ✅ | UUID del usuario | user-123-abc |
| curso_id | Texto/Número | ✅ | ID o código del curso | MAT-2024-001 o 1 |
| fecha_inscripcion | Fecha | ⚪ | Fecha de inscripción | 2024-01-20 |
| fecha_completado | Fecha | ⚪ | Fecha de finalización | 2024-03-15 |
| calificacion_final | Decimal | ⚪ | Nota final (0-20) | 18.5 |
| rating_usuario | Número | ⚪ | Rating del curso (1-5) | 5 |
| porcentaje_avance | Decimal | ⚪ | Avance % (0-100) | 100.0 |
| tiempo_total_minutos | Número | ⚪ | Tiempo invertido | 2400 |
| estado | Texto | ⚪ | Estado de la inscripción | inscrito, en_curso, completado, abandonado |

**Ejemplo de fila:**
```
user-123-abc | MAT-2024-001 | 2024-01-20 | 2024-03-15 | 18.5 | 5 | 100.0 | 2400 | completado
```

**Nota importante:** El campo `curso_id` puede ser:
- El código del curso (ej: MAT-2024-001) → se busca automáticamente el ID numérico
- El ID numérico directamente (ej: 1)

---

## 🔄 Flujo de Carga Recomendado

### Paso 1: Generar Plantillas

```bash
python carga_masiva_excel.py
# Seleccionar opción 1
# Especificar directorio: ./datos
```

Esto generará:
```
datos/
├── cursos.xlsx (con 1 fila de ejemplo)
├── usuarios_perfil.xlsx (con 1 fila de ejemplo)
└── inscripciones.xlsx (con 1 fila de ejemplo)
```

### Paso 2: Completar Plantillas

Abre cada archivo Excel y completa con tus datos reales.

**Tips:**
- No elimines la fila de encabezados (primera fila)
- Usa el formato correcto de fechas: YYYY-MM-DD
- Para arrays (nivel_educativo, tags), separa con `|`
- Deja celdas vacías para campos opcionales (no escribas "NULL")

### Paso 3: Cargar Datos

**Orden recomendado:**
1. Primero cursos (porque inscripciones depende de ellos)
2. Luego usuarios_perfil
3. Finalmente inscripciones

**Opción A - Cargar todo de una vez:**
```bash
python carga_masiva_excel.py
# Seleccionar opción 5
# Especificar directorio: ./datos
```

**Opción B - Cargar uno por uno:**
```bash
python carga_masiva_excel.py
# Opción 2 → cursos.xlsx
# Opción 3 → usuarios_perfil.xlsx
# Opción 4 → inscripciones.xlsx
```

---

## ✅ Validaciones Automáticas

El módulo realiza las siguientes validaciones:

1. **Cursos:**
   - ✅ Código único (no duplicados)
   - ✅ Campos requeridos presentes
   - ✅ Arrays convertidos correctamente

2. **Usuarios:**
   - ✅ user_id existe
   - ✅ Preferencias en formato JSON válido

3. **Inscripciones:**
   - ✅ curso_id existe en la tabla cursos
   - ✅ Fechas en formato correcto
   - ✅ Calificación entre 0-20
   - ✅ Rating entre 1-5
   - ✅ Aprobado calculado automáticamente (>= 14)

---

## 🔄 Actualización de Datos

El módulo usa `ON CONFLICT ... DO UPDATE`, lo que significa:

- Si el registro **ya existe** → Se **actualiza**
- Si el registro **no existe** → Se **inserta**

**Identificadores únicos:**
- Cursos: `codigo_curso`
- Usuarios: `user_id`
- Inscripciones: combinación de `(user_id, curso_id)`

**Ejemplo:**
Si cargas `cursos.xlsx` con un curso que ya existe:
```
Antes: MAT-2024-001 | Matemáticas Básica | 30 horas
Nuevo: MAT-2024-001 | Matemáticas Avanzada | 40 horas
Resultado: ✅ Actualizado a 40 horas
```

---

## 📊 Triggers Automáticos

Al insertar/actualizar inscripciones, se actualizan automáticamente:

**En tabla `cursos`:**
- `total_inscritos`
- `total_completados`
- `calificacion_promedio`
- `tasa_completitud`

**En tabla `usuarios_perfil`:**
- `total_cursos_completados`
- `total_cursos_en_progreso`
- `calificacion_promedio_historica`
- `tasa_completitud`

**Ejemplo:**
```
Cargas inscripción de Juan en Matemáticas (completado, nota 18)
   ↓
Trigger actualiza automáticamente:
- cursos.total_completados += 1
- cursos.calificacion_promedio (recalcula)
- usuarios_perfil.total_cursos_completados += 1
```

---

## ⚠️ Errores Comunes

### Error: "Falta la columna requerida: codigo_curso"
**Solución:** Verifica que la primera fila tenga EXACTAMENTE los nombres de columnas especificados.

### Error: "Curso no encontrado: MAT-2024-001"
**Solución:** Carga primero `cursos.xlsx` antes de `inscripciones.xlsx`.

### Error: "duplicate key value violates unique constraint"
**Solución:** Ya existe ese registro. El sistema lo actualizará automáticamente, no es un error crítico.

### Error: "invalid input syntax for type json"
**Solución:** En el campo `preferencias`, usa comillas dobles para JSON válido:
```
✅ Correcto: {"areas_interes": ["didáctica"]}
❌ Incorrecto: {'areas_interes': ['didáctica']}
```

---

## 🧪 Testing

### Test de Conexión

```python
from carga_masiva_excel import test_conexion_db

if test_conexion_db():
    print("Base de datos lista para cargar datos")
```

### Cargar Datos de Prueba

```bash
# 1. Generar plantillas
python carga_masiva_excel.py
# Opción 1 → ./datos_prueba

# 2. Las plantillas vienen con 1 fila de ejemplo
# 3. Cargar directamente
python carga_masiva_excel.py
# Opción 5 → ./datos_prueba
```

---

## 📈 Monitoreo Post-Carga

### Verificar Datos Cargados

```sql
-- Ver cursos cargados
SELECT codigo_curso, nombre, categoria, total_inscritos 
FROM cursos 
ORDER BY fecha_creacion DESC 
LIMIT 10;

-- Ver inscripciones
SELECT 
    i.user_id,
    c.nombre as curso,
    i.estado,
    i.calificacion_final
FROM inscripciones i
JOIN cursos c ON i.curso_id = c.curso_id
ORDER BY i.fecha_inscripcion DESC
LIMIT 10;

-- Estadísticas generales
SELECT 
    COUNT(*) as total_cursos,
    SUM(total_inscritos) as total_inscripciones,
    AVG(calificacion_promedio) as promedio_general
FROM cursos
WHERE estado = 'activo';
```

---

## 🎯 Ejemplos Completos

### Ejemplo 1: Cargar 10 Cursos

**cursos.xlsx:**
```
codigo_curso | nombre | categoria | nivel_dificultad | duracion_horas | area_curricular | estado
MAT-001 | Matemáticas 1 | didáctica | basico | 30 | matemáticas | activo
MAT-002 | Matemáticas 2 | didáctica | intermedio | 40 | matemáticas | activo
EVA-001 | Evaluación Formativa | evaluación | basico | 25 | general | activo
...
```

**Resultado:**
```
✅ Cursos procesados: 10 insertados, 0 actualizados
```

### Ejemplo 2: Actualizar Curso Existente

**cursos.xlsx (segunda carga):**
```
codigo_curso | nombre | categoria | nivel_dificultad | duracion_horas | area_curricular | estado
MAT-001 | Matemáticas Básica ACTUALIZADA | didáctica | basico | 35 | matemáticas | activo
```

**Resultado:**
```
✅ Cursos procesados: 0 insertados, 1 actualizados
```

### Ejemplo 3: Cargar 100 Inscripciones

```python
from carga_masiva_excel import cargar_inscripciones_desde_excel

insertados, actualizados = cargar_inscripciones_desde_excel('inscripciones_masivas.xlsx')
print(f"Procesadas {insertados + actualizados} inscripciones")
```

---

## 💡 Tips Pro

1. **Usa código de curso en vez de ID numérico:**
   ```
   ✅ Mejor: curso_id = "MAT-2024-001"
   ⚪ También funciona: curso_id = 1
   ```

2. **Deja fechas vacías si no aplican:**
   ```
   user-123 | MAT-001 | 2024-01-20 | [vacío] | [vacío] | ... | en_curso
   ```

3. **Aprovecha los triggers:**
   No necesitas calcular manualmente `total_inscritos` o `calificacion_promedio`, los triggers lo hacen.

4. **Valida datos antes de cargar masivamente:**
   Prueba con 5-10 filas primero, verifica que se carguen bien, luego carga todo.

---

¿Listo para cargar tus datos? 🚀
