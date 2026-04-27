from flask import Flask, request, jsonify, render_template
import ollama 
import chromadb
import os
from dotenv import load_dotenv
import subprocess
import time
import urllib.request
from urllib.error import URLError



# Carga las variables de entorno desde el archivo .env
load_dotenv()

app = Flask(__name__)

# Lee la clave secreta. Si por alguna razón no encuentra el archivo .env, 
# usa una clave genérica de respaldo (útil para pruebas).
app.secret_key = os.getenv("FLASK_SECRET_KEY", "clave_por_defecto_desarrollo")


# --- 1. CONFIGURACIÓN DE LA BASE VECTORIAL ---
chroma_client = chromadb.Client()
# get_or_create_collection evita errores si la base ya existe
collection = chroma_client.get_or_create_collection(name="memoria_cecyt16")


def asegurar_ollama():
    try:
        # Intentamos hacer ping al puerto de Ollama para ver si ya está vivo
        urllib.request.urlopen("http://127.0.0.1:11434")
        print("✅ Servidor de Ollama detectado.")
    except URLError:
        print("⏳ El servidor de Ollama está apagado. Iniciándolo en segundo plano...")
        # Abrimos Ollama en segundo plano. DEVNULL oculta los logs de Ollama para no ensuciar tu terminal de Flask
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Le damos 3 segundos al motor para que arranque correctamente antes de seguir
        time.sleep(3)
        print("✅ Servidor de Ollama iniciado con éxito.")


def inicializar_base_vectorial():
    ruta = "conocimiento_cecyt16.txt"
    if not os.path.exists(ruta):
        print("⚠️ No se encontró el archivo de conocimiento. El bot no tendrá memoria externa.")
        return
    
    with open(ruta, "r", encoding="utf-8") as f:
        texto = f.read()
    
    # Separamos el texto por párrafos (asumiendo que en tu txt dejas una línea en blanco entre cada tema)
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    
    print("Cargando conocimiento a la base de datos vectorial...")
    for i, parrafo in enumerate(parrafos):
        # Generar vector para cada párrafo
        emb = ollama.embeddings(model="nomic-embed-text", prompt=parrafo)["embedding"]
        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[emb],
            documents=[parrafo]
        )
    print("✅ ¡Base de datos vectorial lista!")


# Ejecutamos esto antes de cargar ChromaDB
asegurar_ollama()
inicializar_base_vectorial()

historiales = {}

def get_historial(session_id):
    if session_id not in historiales:
        # El system prompt ahora es más general, porque los datos exactos se inyectan dinámicamente
        historiales[session_id] = [
            {"role": "system", "content": "Eres un asistente virtual oficial del CECyT 16 en Pachuca, Hidalgo. "
            "Tu objetivo es ayudar con trámites y dudas de la escuela. Usa el contexto proporcionado. "
            "REGLAS CRÍTICAS: 1. Si preguntan por ubicación, usa [MAPA_UBICACION] y NO menciones la composición/diagrama. "
            "2. Si preguntan por organización/composición/mapa interno, usa [DIAGRAMA_ESCUELA] y NO menciones la dirección física ni uses el mapa de ubicación. "
            "Da respuestas concisas."}
        ]
    return historiales[session_id]

# --- 2. RUTAS WEB ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    mensaje_usuario = data.get("mensaje")

    historial = get_historial(session_id)

    # --- MAGIA DEL RAG (Búsqueda Vectorial) ---

    # 1. Convertir la pregunta del usuario a vector (con memoria de contexto)
    texto_para_buscar = mensaje_usuario
    
    # Si ya hay historial previo (System + User + AI = mínimo 3 mensajes), 
    # le pegamos la pregunta anterior para darle contexto a ChromaDB.
    if len(historial) >= 3:
        pregunta_anterior = historial[-2]["content"] # Extrae lo que el usuario preguntó antes
        texto_para_buscar = f"{pregunta_anterior}. {mensaje_usuario}"
        
    vector_pregunta = ollama.embeddings(model="nomic-embed-text", prompt=texto_para_buscar)["embedding"]

    
    # 2. Buscar los 2 fragmentos de información más relevantes en ChromaDB
    resultados = collection.query(query_embeddings=[vector_pregunta], n_results=2)
    
    contexto_recuperado = ""
    if resultados['documents'] and resultados['documents'][0]:
        # Unimos los fragmentos encontrados
        contexto_recuperado = "\n".join(resultados['documents'][0])

    # 3. Armamos un mensaje "secreto" para la IA que incluye la info de la escuela y la pregunta
    mensaje_con_contexto = f"Contexto de la escuela encontrado:\n{contexto_recuperado}\n\nPregunta del usuario: {mensaje_usuario}"

    # Guardamos la pregunta simple en el historial de la sesión (para que la IA no se confunda recordando instrucciones viejas)
    historial.append({"role": "user", "content": mensaje_usuario})

    # Pero para esta petición específica, le mandamos a la IA la versión inyectada con la base de datos
    mensajes_a_enviar = [historial[0]] + historial[-19:-1] + [{"role": "user", "content": mensaje_con_contexto}]

    # --- CONSULTA AL MODELO ---
    respuesta = ollama.chat(
        model="llama3.2:3b",
        messages=mensajes_a_enviar
    )

    mensaje_ia = respuesta["message"]["content"]
    
    # PROCESAMIENTO EXTRA: Mostrar mapa si piden ubicación
    if "[MAPA_UBICACION]" in mensaje_ia:
        mapa_html = '<div style="position: relative; width: 100%; height: 200px; border-radius: 10px; overflow: hidden; margin-top: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"><iframe src="https://maps.google.com/maps?q=CECyT%2016%20Hidalgo,%20San%20Agust%C3%ADn%20Tlaxiaca&t=&z=15&ie=UTF8&iwloc=&output=embed" width="100%" height="100%" frameborder="0" style="border:0;" allowfullscreen></iframe><a href="https://www.google.com/maps/search/CECyT+16+Hidalgo" target="_blank" style="position: absolute; top:0; left:0; width:100%; height:100%; z-index:10; background:rgba(255,255,255,0.01);" title="Abrir en Google Maps"></a></div><p style="font-size: 0.8rem; text-align: center; margin-top: 5px; color: var(--text-gray);"><i>Da clic en el mapa para abrir en Google Maps</i></p>'
        mensaje_ia = mensaje_ia.replace("[MAPA_UBICACION]", mapa_html)

    # PROCESAMIENTO EXTRA: Mostrar diagrama (Placeholder)
    if "[DIAGRAMA_ESCUELA]" in mensaje_ia:
        diagrama_html = '<div style="margin-top: 10px; padding: 15px; border: 2px dashed #6B1C3A; border-radius: 10px; background: #fff5f8; text-align: center;"><p style="color: #6B1C3A; font-weight: bold;">[ PRÓXIMAMENTE: DIAGRAMA DE COMPOSICIÓN ]</p><p style="font-size: 0.85rem; color: #555;">Aquí se mostrará el diagrama detallado de la organización y distribución de los edificios del CECyT 16.</p></div>'
        mensaje_ia = mensaje_ia.replace("[DIAGRAMA_ESCUELA]", diagrama_html)

    # Guardamos la respuesta de la IA en el historial real
    historial.append({"role": "assistant", "content": mensaje_ia})

    return jsonify({"respuesta": mensaje_ia, "session_id": session_id})

if __name__ == "__main__":
    app.run(port=5002, debug=True)