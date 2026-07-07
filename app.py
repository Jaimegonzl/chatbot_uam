from flask import Flask, render_template, request, jsonify, session
from scripts.chatbot_service2 import ChatbotService
import os
from dotenv import load_dotenv

load_dotenv()          # Carga las variables de entorno desde el archivo .env

app = Flask(__name__)

# Si no la encuentra en el .env, usará la de respaldo por seguridad.
app.secret_key = os.getenv("FLASK_SECRET_KEY", "clave_alternativa_de_emergencia")

project_root = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(project_root, "db", "chatbot2.db")

print("="*50)
print(f"INICIANDO CHATBOT CON BASE DE DATOS:")
print(f"RUTA: {db_path}")
print("="*50)

bot = ChatbotService(db_path)

@app.route("/")
def index():
    # Limpiamos la sesión al cargar o recargar la página para empezar de cero
    session.clear()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])

def chat():
    data = request.json
    pregunta = data.get("mensaje")
    
    # 1. Si el usuario no tiene una estructura de "slots" en su memoria, se la creamos
    if "slots" not in session:
        session["slots"] = {
            "carrera": None,              # Aquí guardaremos 'computacion' o 'electrica'
            "esperando_carrera": False,   # Bandera para saber si el próximo mensaje es su carrera
            "intencion_pendiente": None   # Guarda la pregunta original que detonó el slot-filling
        }
    
    # 2. Extraemos los slots actuales de la sesión
    slots = session["slots"]
    
    # 3. Pasamos la pregunta y los slots al servicio del chatbot
    resultado = bot.buscar_respuesta(pregunta, slots)
    
    # 4. Guardamos los slots modificados de vuelta en la sesión de Flask
    session["slots"] = slots
    session.modified = True  # Le avisa a Flask que el diccionario interno cambió
    
    return jsonify(resultado)

if __name__ == "__main__":
    app.run(debug=True)