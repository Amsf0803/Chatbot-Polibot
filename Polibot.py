from flask import Flask, request, jsonify, render_template
import ollama 
import chromadb
import os

app = Flask(__name__)
app.secret_key = "2fa71559dc6301ada2cdadddb3775fda9f9cb11eea491710265030a1e020ae96"

# --- 1. CONFIGURACIÓN DE LA BASE VECTORIAL ---
chroma_client = chromadb.Client()
# get_or_create_collection evita errores si la base ya existe
collection = chroma_client.get_or_create_collection(name="memoria_cecyt16")

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

# Ejecutamos la carga antes de arrancar las rutas
inicializar_base_vectorial()

historiales = {}

def get_historial(session_id):
    if session_id not in historiales:
        # El system prompt ahora es más general, porque los datos exactos se inyectan dinámicamente
        historiales[session_id] = [
            {"role": "system", "content": "Eres un asistente virtual oficial del CECyT 16 en Pachuca, Hidalgo. "
            "Tu objetivo es ayudar con trámites y dudas de la escuela. Usa la información de 'Contexto de la escuela' que se te proporcione para responder. "
            "Da respuestas concisas. Si no tienes la información exacta en el contexto, di: 'Perdón, no tengo esa información actualmente, te sugiero checar la página oficial'."}
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
    # 1. Convertir la pregunta del usuario a vector
    vector_pregunta = ollama.embeddings(model="nomic-embed-text", prompt=mensaje_usuario)["embedding"]
    
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
    
    # Guardamos la respuesta de la IA en el historial real
    historial.append({"role": "assistant", "content": mensaje_ia})

    return jsonify({"respuesta": mensaje_ia, "session_id": session_id})

if __name__ == "__main__":
    app.run(port=5002, debug=True)