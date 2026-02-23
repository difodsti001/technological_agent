"""
📊 MÓDULO DE CARGA MASIVA DESDE EXCEL - AGENTE TECNOLÓGICO
===========================================================

Permite cargar datos de cursos, inscripciones y perfiles de usuarios
desde archivos Excel a PostgreSQL.
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()

# ==============================
# 🔧 CONFIGURACIÓN
# ==============================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "proyectos_ia"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", 5432)
}


# ==============================
# 🗄️ CONEXIÓN A BASE DE DATOS
# ==============================

def get_db_connection():
    """Obtiene conexión a PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)


# ==============================
# 📚 FUNCIONES DE CARGA POR TABLA
# ==============================

def cargar_cursos_desde_excel(filepath: str) -> Tuple[int, int]:
    """
    Carga cursos desde un archivo Excel.
    
    Formato esperado del Excel (columnas):
    - codigo_curso: Código único del curso (ej: MAT-2024-001)
    - nombre: Nombre del curso
    - descripcion: Descripción breve
    - categoria: matemáticas, evaluación, didáctica, etc.
    - nivel_dificultad: basico, intermedio, avanzado
    - duracion_horas: Número entero
    - area_curricular: matemáticas, comunicación, ciencias, etc.
    - nivel_educativo: inicial|primaria|secundaria (separados por |)
    - region_enfoque: costa|sierra|selva|todas (separados por |)
    - fecha_creacion: YYYY-MM-DD
    - tags: tag1|tag2|tag3 (separados por |)
    - estado: activo, inactivo, archivado
    
    Returns:
        Tupla (insertados, actualizados)
    """
    print(f"📚 Cargando cursos desde {filepath}...")
    
    # Leer Excel
    df = pd.read_excel(filepath)
    
    # Validar columnas requeridas
    columnas_requeridas = [
        'codigo_curso', 'nombre', 'categoria', 'nivel_dificultad', 
        'duracion_horas', 'area_curricular'
    ]
    
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"❌ Falta la columna requerida: {col}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    insertados = 0
    actualizados = 0
    
    try:
        for idx, row in df.iterrows():
            # Preparar arrays de PostgreSQL
            nivel_educativo = row.get('nivel_educativo', 'todas')
            if isinstance(nivel_educativo, str):
                nivel_educativo = [x.strip() for x in nivel_educativo.split('|')]
            else:
                nivel_educativo = ['todas']
            
            region_enfoque = row.get('region_enfoque', 'todas')
            if isinstance(region_enfoque, str):
                region_enfoque = [x.strip() for x in region_enfoque.split('|')]
            else:
                region_enfoque = ['todas']
            
            tags = row.get('tags', '')
            if isinstance(tags, str) and tags:
                tags = [x.strip() for x in tags.split('|')]
            else:
                tags = []
            
            # Insertar o actualizar
            cursor.execute("""
                INSERT INTO cursos (
                    codigo_curso, nombre, descripcion, categoria, 
                    nivel_dificultad, duracion_horas, area_curricular,
                    nivel_educativo, region_enfoque, fecha_creacion,
                    tags, estado
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (codigo_curso) DO UPDATE SET
                    nombre = EXCLUDED.nombre,
                    descripcion = EXCLUDED.descripcion,
                    categoria = EXCLUDED.categoria,
                    nivel_dificultad = EXCLUDED.nivel_dificultad,
                    duracion_horas = EXCLUDED.duracion_horas,
                    area_curricular = EXCLUDED.area_curricular,
                    nivel_educativo = EXCLUDED.nivel_educativo,
                    region_enfoque = EXCLUDED.region_enfoque,
                    tags = EXCLUDED.tags,
                    estado = EXCLUDED.estado,
                    updated_at = NOW()
                RETURNING (xmax = 0) AS inserted
            """, (
                row['codigo_curso'],
                row['nombre'],
                row.get('descripcion', ''),
                row['categoria'],
                row['nivel_dificultad'],
                int(row['duracion_horas']),
                row['area_curricular'],
                nivel_educativo,
                region_enfoque,
                pd.to_datetime(row.get('fecha_creacion', datetime.now())).date(),
                tags,
                row.get('estado', 'activo')
            ))
            
            # Verificar si fue insert o update
            result = cursor.fetchone()
            if result[0]:
                insertados += 1
            else:
                actualizados += 1
        
        conn.commit()
        print(f"✅ Cursos procesados: {insertados} insertados, {actualizados} actualizados")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    
    return insertados, actualizados


def cargar_usuarios_perfil_desde_excel(filepath: str) -> Tuple[int, int]:
    """
    Carga perfiles de usuarios desde un archivo Excel.
    
    Formato esperado del Excel (columnas):
    - user_id: UUID del usuario
    - nivel_educativo: inicial, primaria, secundaria
    - area_especialidad: matemáticas, comunicación, etc.
    - region: costa, sierra, selva
    - departamento: Lima, Cusco, etc.
    - anos_experiencia: Número entero
    - preferencias: JSON string (opcional)
    
    Returns:
        Tupla (insertados, actualizados)
    """
    print(f"👥 Cargando perfiles de usuarios desde {filepath}...")
    
    df = pd.read_excel(filepath)
    
    # Validar columnas requeridas
    columnas_requeridas = ['user_id']
    
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"❌ Falta la columna requerida: {col}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    insertados = 0
    actualizados = 0
    
    try:
        for idx, row in df.iterrows():
            # Parsear preferencias si vienen como string JSON
            preferencias = row.get('preferencias', '{}')
            if isinstance(preferencias, str):
                import json
                try:
                    preferencias = json.loads(preferencias)
                except:
                    preferencias = {}
            
            cursor.execute("""
                INSERT INTO usuarios_perfil (
                    user_id, nivel_educativo, area_especialidad, 
                    region, departamento, anos_experiencia, preferencias
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    nivel_educativo = EXCLUDED.nivel_educativo,
                    area_especialidad = EXCLUDED.area_especialidad,
                    region = EXCLUDED.region,
                    departamento = EXCLUDED.departamento,
                    anos_experiencia = EXCLUDED.anos_experiencia,
                    preferencias = EXCLUDED.preferencias,
                    updated_at = NOW()
                RETURNING (xmax = 0) AS inserted
            """, (
                row['user_id'],
                row.get('nivel_educativo'),
                row.get('area_especialidad'),
                row.get('region'),
                row.get('departamento'),
                int(row.get('anos_experiencia', 0)),
                preferencias if isinstance(preferencias, str) else str(preferencias)
            ))
            
            result = cursor.fetchone()
            if result[0]:
                insertados += 1
            else:
                actualizados += 1
        
        conn.commit()
        print(f"✅ Perfiles procesados: {insertados} insertados, {actualizados} actualizados")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    
    return insertados, actualizados


def cargar_inscripciones_desde_excel(filepath: str) -> Tuple[int, int]:
    """
    Carga inscripciones desde un archivo Excel.
    
    Formato esperado del Excel (columnas):
    - user_id: UUID del usuario
    - curso_id: ID del curso (se busca por codigo_curso)
    - fecha_inscripcion: YYYY-MM-DD HH:MM:SS
    - fecha_completado: YYYY-MM-DD HH:MM:SS (opcional)
    - calificacion_final: Decimal 0-20
    - rating_usuario: Integer 1-5 (opcional)
    - porcentaje_avance: Decimal 0-100
    - tiempo_total_minutos: Integer
    - estado: inscrito, en_curso, completado, abandonado
    
    Returns:
        Tupla (insertados, actualizados)
    """
    print(f"📝 Cargando inscripciones desde {filepath}...")
    
    df = pd.read_excel(filepath)
    
    # Validar columnas requeridas
    columnas_requeridas = ['user_id', 'curso_id']
    
    for col in columnas_requeridas:
        if col not in df.columns:
            raise ValueError(f"❌ Falta la columna requerida: {col}")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    insertados = 0
    actualizados = 0
    errores = 0
    
    try:
        for idx, row in df.iterrows():
            # Obtener curso_id real desde codigo_curso
            curso_id = row['curso_id']
            
            # Si viene codigo_curso en vez de ID numérico, buscarlo
            if isinstance(curso_id, str):
                cursor.execute("SELECT curso_id FROM cursos WHERE codigo_curso = %s", (curso_id,))
                result = cursor.fetchone()
                if result:
                    curso_id = result[0]
                else:
                    print(f"⚠️  Curso no encontrado: {curso_id} (fila {idx+2})")
                    errores += 1
                    continue
            
            # Procesar fechas
            fecha_inscripcion = pd.to_datetime(row.get('fecha_inscripcion', datetime.now()))
            fecha_completado = None
            if pd.notna(row.get('fecha_completado')):
                fecha_completado = pd.to_datetime(row['fecha_completado'])
            
            # Calcular aprobado automáticamente
            calificacion = row.get('calificacion_final')
            aprobado = None
            if pd.notna(calificacion):
                aprobado = float(calificacion) >= 14.0
            
            cursor.execute("""
                INSERT INTO inscripciones (
                    user_id, curso_id, fecha_inscripcion, fecha_completado,
                    calificacion_final, rating_usuario, porcentaje_avance,
                    tiempo_total_minutos, estado, aprobado
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, curso_id) DO UPDATE SET
                    fecha_completado = EXCLUDED.fecha_completado,
                    calificacion_final = EXCLUDED.calificacion_final,
                    rating_usuario = EXCLUDED.rating_usuario,
                    porcentaje_avance = EXCLUDED.porcentaje_avance,
                    tiempo_total_minutos = EXCLUDED.tiempo_total_minutos,
                    estado = EXCLUDED.estado,
                    aprobado = EXCLUDED.aprobado,
                    updated_at = NOW()
                RETURNING (xmax = 0) AS inserted
            """, (
                row['user_id'],
                curso_id,
                fecha_inscripcion,
                fecha_completado,
                float(calificacion) if pd.notna(calificacion) else None,
                int(row['rating_usuario']) if pd.notna(row.get('rating_usuario')) else None,
                float(row.get('porcentaje_avance', 0.0)),
                int(row.get('tiempo_total_minutos', 0)),
                row.get('estado', 'en_curso'),
                aprobado
            ))
            
            result = cursor.fetchone()
            if result[0]:
                insertados += 1
            else:
                actualizados += 1
        
        conn.commit()
        print(f"✅ Inscripciones procesadas: {insertados} insertados, {actualizados} actualizados")
        if errores > 0:
            print(f"⚠️  {errores} registros con errores (cursos no encontrados)")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        cursor.close()
        conn.close()
    
    return insertados, actualizados


# ==============================
# 📋 FUNCIÓN PARA CARGAR TODO
# ==============================

def cargar_todo_desde_directorio(directorio: str):
    """
    Busca y carga todos los archivos Excel en un directorio.
    
    Nombres esperados de archivos:
    - cursos.xlsx
    - usuarios_perfil.xlsx
    - inscripciones.xlsx
    
    Args:
        directorio: Ruta al directorio con los archivos
    """
    import os.path
    
    print(f"\n{'='*60}")
    print(f"📂 Cargando datos desde: {directorio}")
    print(f"{'='*60}\n")
    
    resumen = {}
    
    # 1. Cargar cursos (primero, porque inscripciones depende de ellos)
    cursos_file = os.path.join(directorio, 'cursos.xlsx')
    if os.path.exists(cursos_file):
        try:
            ins, upd = cargar_cursos_desde_excel(cursos_file)
            resumen['cursos'] = {'insertados': ins, 'actualizados': upd}
        except Exception as e:
            print(f"❌ Error cargando cursos: {e}")
            resumen['cursos'] = {'error': str(e)}
    else:
        print(f"⚠️  No se encontró {cursos_file}")
    
    print()
    
    # 2. Cargar perfiles de usuarios
    usuarios_file = os.path.join(directorio, 'usuarios_perfil.xlsx')
    if os.path.exists(usuarios_file):
        try:
            ins, upd = cargar_usuarios_perfil_desde_excel(usuarios_file)
            resumen['usuarios_perfil'] = {'insertados': ins, 'actualizados': upd}
        except Exception as e:
            print(f"❌ Error cargando perfiles: {e}")
            resumen['usuarios_perfil'] = {'error': str(e)}
    else:
        print(f"⚠️  No se encontró {usuarios_file}")
    
    print()
    
    # 3. Cargar inscripciones (último, porque depende de cursos y usuarios)
    inscripciones_file = os.path.join(directorio, 'inscripciones.xlsx')
    if os.path.exists(inscripciones_file):
        try:
            ins, upd = cargar_inscripciones_desde_excel(inscripciones_file)
            resumen['inscripciones'] = {'insertados': ins, 'actualizados': upd}
        except Exception as e:
            print(f"❌ Error cargando inscripciones: {e}")
            resumen['inscripciones'] = {'error': str(e)}
    else:
        print(f"⚠️  No se encontró {inscripciones_file}")
    
    # Imprimir resumen
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE CARGA:")
    print(f"{'='*60}\n")
    
    for tabla, stats in resumen.items():
        print(f"📋 {tabla.upper()}:")
        if 'error' in stats:
            print(f"   ❌ Error: {stats['error']}")
        else:
            print(f"   ✅ Insertados: {stats.get('insertados', 0)}")
            print(f"   🔄 Actualizados: {stats.get('actualizados', 0)}")
        print()


# ==============================
# 📝 GENERADOR DE PLANTILLAS
# ==============================

def generar_plantillas_excel(directorio: str):
    """
    Genera archivos Excel de plantilla vacíos para facilitar la carga de datos.
    
    Args:
        directorio: Dónde guardar las plantillas
    """
    import os
    os.makedirs(directorio, exist_ok=True)
    
    # Plantilla de cursos
    df_cursos = pd.DataFrame(columns=[
        'codigo_curso', 'nombre', 'descripcion', 'categoria', 
        'nivel_dificultad', 'duracion_horas', 'area_curricular',
        'nivel_educativo', 'region_enfoque', 'fecha_creacion',
        'tags', 'estado'
    ])
    
    # Agregar fila de ejemplo
    df_cursos.loc[0] = [
        'MAT-2024-001',
        'Estrategias Didácticas en Matemática',
        'Aprende estrategias innovadoras para enseñar matemáticas',
        'didáctica',
        'intermedio',
        40,
        'matemáticas',
        'primaria|secundaria',
        'todas',
        '2024-01-15',
        'didáctica|matemáticas|estrategias',
        'activo'
    ]
    
    df_cursos.to_excel(os.path.join(directorio, 'cursos.xlsx'), index=False)
    print(f"✅ Plantilla generada: {directorio}/cursos.xlsx")
    
    # Plantilla de usuarios_perfil
    df_usuarios = pd.DataFrame(columns=[
        'user_id', 'nivel_educativo', 'area_especialidad', 
        'region', 'departamento', 'anos_experiencia', 'preferencias'
    ])
    
    df_usuarios.loc[0] = [
        'user-123-abc',
        'primaria',
        'matemáticas',
        'costa',
        'Lima',
        5,
        '{"areas_interes": ["didáctica", "evaluación"], "nivel_preferido": "intermedio"}'
    ]
    
    df_usuarios.to_excel(os.path.join(directorio, 'usuarios_perfil.xlsx'), index=False)
    print(f"✅ Plantilla generada: {directorio}/usuarios_perfil.xlsx")
    
    # Plantilla de inscripciones
    df_inscripciones = pd.DataFrame(columns=[
        'user_id', 'curso_id', 'fecha_inscripcion', 'fecha_completado',
        'calificacion_final', 'rating_usuario', 'porcentaje_avance',
        'tiempo_total_minutos', 'estado'
    ])
    
    df_inscripciones.loc[0] = [
        'user-123-abc',
        'MAT-2024-001',
        '2024-01-20',
        '2024-03-15',
        18.5,
        5,
        100.0,
        2400,
        'completado'
    ]
    
    df_inscripciones.to_excel(os.path.join(directorio, 'inscripciones.xlsx'), index=False)
    print(f"✅ Plantilla generada: {directorio}/inscripciones.xlsx")
    
    print(f"\n📁 Todas las plantillas generadas en: {directorio}")


# ==============================
# 🧪 FUNCIÓN DE TESTING
# ==============================

def test_conexion_db():
    """Prueba la conexión a la base de datos."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"✅ Conectado a PostgreSQL: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return False


# ==============================
# 🚀 MENÚ INTERACTIVO
# ==============================

def menu_principal():
    """Menú interactivo para cargar datos."""
    print("\n" + "="*60)
    print("📊 CARGADOR DE DATOS MASIVOS - AGENTE TECNOLÓGICO")
    print("="*60)
    
    # Verificar conexión
    if not test_conexion_db():
        print("\n❌ No se pudo conectar a la base de datos. Verifica tu .env")
        return
    
    while True:
        print("\n¿Qué deseas hacer?")
        print("1. Generar plantillas Excel vacías")
        print("2. Cargar cursos desde Excel")
        print("3. Cargar perfiles de usuarios desde Excel")
        print("4. Cargar inscripciones desde Excel")
        print("5. Cargar TODO desde un directorio")
        print("6. Salir")
        
        opcion = input("\nSelecciona una opción (1-6): ").strip()
        
        if opcion == '1':
            directorio = input("Directorio donde guardar plantillas (default: ./plantillas): ").strip()
            if not directorio:
                directorio = './plantillas'
            generar_plantillas_excel(directorio)
        
        elif opcion == '2':
            filepath = input("Ruta del archivo Excel de cursos: ").strip()
            try:
                cargar_cursos_desde_excel(filepath)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '3':
            filepath = input("Ruta del archivo Excel de usuarios: ").strip()
            try:
                cargar_usuarios_perfil_desde_excel(filepath)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '4':
            filepath = input("Ruta del archivo Excel de inscripciones: ").strip()
            try:
                cargar_inscripciones_desde_excel(filepath)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '5':
            directorio = input("Directorio con los archivos Excel: ").strip()
            if not directorio:
                print("❌ Debes especificar un directorio")
                continue
            try:
                cargar_todo_desde_directorio(directorio)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif opcion == '6':
            print("\n👋 ¡Hasta pronto!")
            break
        
        else:
            print("❌ Opción inválida")


# ==============================
# 🎯 PUNTO DE ENTRADA
# ==============================

if __name__ == "__main__":
    menu_principal()
