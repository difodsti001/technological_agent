import pandas as pd
from sistema_recomendacion import HybridRecommender, RecommenderConfig

print(f"TOP_K configurado: {RecommenderConfig.TOP_K_RECOMENDACIONES}")

# Cargar solo una muestra pequeña para probar rápido
df = pd.read_excel(r"G:\IA\technological_agent\data\DATA_RECOM.xlsx", nrows=5000)
print(f"Registros cargados: {len(df)}")

rec = HybridRecommender(df=df, openai_api_key="", gemini_api_key="")

# Probar con un DNI conocido
resultados = rec.recomendar_hibrido(
    user_id="75271429",
    top_k=3,
    incluir_justificacion=False   # sin LLM para ir más rápido
)

print(f"\n✅ Cursos recomendados: {len(resultados)}")
for i, r in enumerate(resultados, 1):
    print(f"  #{i} {r['CURSO']} | score: {r['score_final']} | métodos: {r['metodos_usados']}")