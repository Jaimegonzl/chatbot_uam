from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
import unicodedata
from sklearn.linear_model import LogisticRegression
from .bert_intent import cargar_embeddings_db, obtener_respuesta_db

SALONES_DEPTO = {
    # DEPTO ENERGÍA
    "O104": ("Laboratorio O104", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "O105": ("Laboratorio O105", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "O106": ("Laboratorio O106", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "SALAIRE": ("Laboratorio SALAIRE", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LEL1": ("Laboratorio LEL1", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LEL2": ("Laboratorio LEL2", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LEL3": ("Laboratorio LEL3", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LEL4": ("Laboratorio LEL4", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "CEDAC": ("Salón CEDAC", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "CEMAC": ("Salón CEMAC", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LABDIN": ("Laboratorio LABDIN", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LABMEC": ("Laboratorio LABMEC", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LABMET": ("Laboratorio LABMET", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "SM1": ("Salón SM1", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "SM2": ("Salón SM2", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "TMEC": ("Taller TMEC", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "APRO": ("Salón APRO", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LTER": ("Laboratorio LTER", "https://energia.azc.uam.mx/en/ubica-tu-salon"),
    "LTERA": ("Laboratorio LTER A", "https://energia.azc.uam.mx/en/ubica-tu-salon"),

    # DEPTO COMPUTACIÓN (CSC)
    "CHARLESBABBAGE": ("Charles Babbage", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "ADABYRON": ("Ada Byron", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "VINTONCERF": ("Vinton Cerf", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "DOUGLASENGELBART": ("Douglas Engelbart", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "ADOLFOGUZMANARENAS": ("Adolfo Guzman Arenas", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "HEDYLAMARR": ("Hedy Lamarr", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "CRISTINALOYO": ("Dra. María Cristina Loyo Varela", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "JUDEMILHON": ("Jude Milhon", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "GRACEMURRAYHOPPER": ("Grace Murray Hopper", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "ROBERTNOYCE": ("Robert Noyce", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "HANNAOKTABA": ("Dra. Hanna Oktaba", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
    "LEOPOLDOSOLIS": ("Leopoldo Solís", "https://csc.azc.uam.mx/?ref=11#texto-inicio"),
}

INFO_CARRERAS = {
    "computacion": {
        "nombre_oficial": "Ingeniería en Computación",
        "coordinador": "Dr. Carlos Ernesto Carrillo Arellano",
        "ubicacion": "Edificio H, Cubículo 283-B",
        "web": "https://cbi.azc.uam.mx/?page_id=28",
        "edificios_comunes": "Como alumno de Computación (CBI), tus salones y laboratorios principales suelen estar ubicados en los **Edificios E, G y T**."
    },
    "electrica": {
        "nombre_oficial": "Ingeniería Eléctrica",
        "coordinador": "Dr. David Antonio Aragón Verduzco",
        "ubicacion": "Edificio P primer piso, Cubículo 2, Área Eléctrica",
        "web": "https://cbi.azc.uam.mx/?page_id=29",
        "edificios_comunes": "Como alumno de Eléctrica (CBI), tus laboratorios especializados e instrumentación se encuentran principalmente en los **Edificios O y W**."
    }
}

def limpiar_texto(texto):
    texto = texto.lower()
    texto = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    # Quitar acentos alternos por si acaso
    texto = re.sub(r'[áàäâ]', 'a', texto)
    texto = re.sub(r'[éèëê]', 'e', texto)   
    texto = re.sub(r'[íìïî]', 'i', texto)
    texto = re.sub(r'[óòöô]', 'o', texto)
    texto = re.sub(r'[úùüû]', 'u', texto)

    reemplazos = {
        r'\bkomo\b': 'como', r'\bkando\b': 'cuando', r'\bkién\b|\bkien\b': 'quien', r'\bke\b|\bké\b': 'que',
        r'\brecu\b|\brrecu\b|\brrecooperacion\b|\bglobales\b|\bextra\b|\bestra\b': 'recuperación',
        r'\brreynscripcion\b|\bincripsion\b|\bincribirme\b': 'inscripción',
        r'\bvaja\b': 'baja', r'\bboy\b': 'voy', r'\bcrokis\b': 'mapa', r'\bobra\b|\bora\b': 'hora',
        r'\bauvisuales\b|\bauvisales\b': 'audiovisuales', r'\bservisios\b': 'servicios',
        r'\buvikacion\b': 'ubicacion', r'\bquimika\b': 'quimica', r'\banfeatro\b': 'anfiteatro',
        r'\bazca\b':'azcapotzalco', r'\bempiesa\b': 'empieza', r'\bedifisio\b': 'edificio',
        r'\buni\b': 'universidad', r'\btopa\b|\bse topa\b|\bqueda\b': 'esta', r'\bno se\b|\bnose\b|\bno sé\b': '',
        r'\bporfa\b|\bporfis\b|\bpor favor\b': '', r'\bpor \b': '', r'\blab\b': 'laboratorio', r'\bfisika\b': 'fisica'
    }
    for patron, reemplazo in reemplazos.items():
        texto = re.sub(patron, reemplazo, texto)
    texto = re.sub(r'(.)\1{2,}', r'\1', texto) 
    return texto

def buscar_salon_especial(pregunta):
    texto = pregunta.upper()
    texto_norm = ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')
    texto_norm = re.sub(r'[^A-Z0-9 ]', ' ', texto_norm)
    for clave, (nombre_real, link) in SALONES_DEPTO.items():
        if re.search(r'\b' + re.escape(clave) + r'\b', texto_norm):
            return nombre_real, link
    return None, None

def tipo_espacio(nombre):
    nombre_upper = nombre.upper()
    if nombre_upper.startswith("LABORATORIO"): return "laboratorio"
    if nombre_upper.startswith("SALÓN") or nombre_upper.startswith("SALON"): return "salon"
    if re.search(r'\d', nombre_upper): return "codigo"
    return "nombre"

def construir_respuesta_ubicacion(nombre, link):
    tipo = tipo_espacio(nombre)
    if tipo in ["laboratorio", "salon", "codigo"]:
        return f"Puedes consultar la guía para ubicar el {nombre} en el siguiente enlace: {link}"
    return f"El salón de cómputo \"{nombre}\" puede ubicarse utilizando la siguiente guía: {link}"


class ChatbotService:
    def __init__(self, ruta_db):
        print("Cargando modelo BERT...")
        self.modelo = SentenceTransformer('all-MiniLM-L6-v2')
        print("Cargando embeddings desde DB...")
        self.textos, self.categorias, self.embeddings = cargar_embeddings_db(ruta_db)
        self.ruta_db = ruta_db

        print("Entrenando clasificador de intenciones (Logistic Regression)...")
        X_train = np.array([emb.cpu().numpy() if hasattr(emb, 'cpu') else emb for emb in self.embeddings])
        y_train = np.array(self.categorias)
        
        self.clasificador = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        self.clasificador.fit(X_train, y_train)
        print("Chatbot listo.")

    def personalizar_respuesta(self, respuesta_base, categoria, carrera):
        info = INFO_CARRERAS.get(carrera)
        if not info:
            return respuesta_base

        if categoria in ["inscripcion", "recuperacion"]:
            complemento = (
                f" <br><br>📌 <b>Información específica para {info['nombre_oficial']}:</b><br>"
                f"Si tienes problemas con tu carga académica o necesitas firmas, acude con tu coordinator:<br>"
                f"• <b>Coordinación:</b> {info['coordinador']}<br>"
                f"• <b>Ubicación Física:</b> {info['ubicacion']}<br>"
                f"• <b>Sitio Web Oficial:</b> {info['web']}"
            )
            return respuesta_base + complemento

        elif categoria == "salones":
            complemento = (
                f" <br><br>📍 <b>Tip de orientación para {info['nombre_oficial']}:</b><br>"
                f"{info['edificios_comunes']}"
            )
            return respuesta_base + complemento

        return respuesta_base

    # --- NUEVO MÉTODO INTERNO: DETECTOR DE LABS GENERALES ---
    def buscar_laboratorio_general(self, pregunta_limpia):
        """ Detecta si se pide específicamente un laboratorio de física o química general """
        if "laboratorio" in pregunta_limpia:
            # Recuperamos dinámicamente la respuesta base de "salones" que contiene el link al mapa
            respuesta_base, _ = obtener_respuesta_db(self.ruta_db, "salones")
            
            #Cambiar la palabra salones por laboratorio en la respuesta base para que sea más coherente con la pregunta del usuario
            respuesta_base = re.sub(r'\bsalones\b', 'laboratorios', respuesta_base, flags=re.IGNORECASE)
            
            if "fisic" in pregunta_limpia:
                return (
                    f"Los laboratorios de <b>Física</b> se encuentran normalmente en el <b>Edificio G, primer piso</b>. "
                    f"Aquí está un mapa de la unidad para ubicarte:<br><br>{respuesta_base}"
                )
            elif "quimic" in pregunta_limpia:
                return (
                    f"Los laboratorios de <b>Química</b> se encuentran ubicados en el <b>Edificio G, planta baja</b>. "
                    f"Aquí está un mapa de la unidad para ubicarte:<br><br>{respuesta_base}"
                )
        return None

    def buscar_respuesta(self, pregunta_usuario, slots):
        pregunta_limpia = limpiar_texto(pregunta_usuario)

        # =========================================================
        # TRATAMIENTO DE SLOTS (GESTIÓN DE MEMORIA)
        # =========================================================
        if slots.get("esperando_carrera"):
            carrera_detectada = None
            if "computacion" in pregunta_limpia or "compu" in pregunta_limpia:
                carrera_detectada = "computacion"
            elif "electrica" in pregunta_limpia:
                carrera_detectada = "electrica"

            if carrera_detectada:
                slots["carrera"] = carrera_detectada
                slots["esperando_carrera"] = False
                
                categoria_pausada = slots.get("intencion_pendiente")
                slots["intencion_pendiente"] = None
                
                respuesta_base, fuente = obtener_respuesta_db(self.ruta_db, categoria_pausada)
                respuesta_final = self.personalizar_respuesta(respuesta_base, categoria_pausada, carrera_detectada)
                
                nombre_carrera = INFO_CARRERAS[carrera_detectada]["nombre_oficial"]
                return {
                    "respuesta": f"¡Entendido! Ya registré que eres de <b>{nombre_carrera}</b>. Aquí tienes la información:<br><br>{respuesta_final}",
                    "fuente": fuente,
                    "confianza": 1.0,
                    "categoria": categoria_pausada,
                    "opciones": []
                }
            else:
                return {
                    "respuesta": "Por favor, selecciona una de las opciones válidas para poder ayudarte:",
                    "fuente": "Sistema",
                    "confianza": 1.0,
                    "categoria": "solicitud_carrera_error",
                    "opciones": ["Computación", "Eléctrica"]
                }

        # LÓGICA DE SALONES ESPECIALES (Por código exacto, ej: O104)
        salon_detectado, link_salon = buscar_salon_especial(pregunta_limpia)
        if link_salon:
            respuesta = construir_respuesta_ubicacion(salon_detectado, link_salon)
            return {
                "respuesta": respuesta, "fuente": "Guía de ubicación", "confianza": 1.0, "categoria": "salones_especiales", "opciones": []
            }
        
        # --- LLAMADA CORREGIDA A LA LÓGICA INTERCEPTORA DE LABORATORIOS ---
        print("Pregunta limpia: ", pregunta_limpia)  # Debug: Ver la pregunta después de limpiar
        respuesta_lab = self.buscar_laboratorio_general(pregunta_limpia)
        if respuesta_lab:
            return {
                "respuesta": respuesta_lab,
                "fuente": "Ubicación de Laboratorios Generales",
                "confianza": 1.0,
                "categoria": "salones_laboratorios",
                "opciones": []
            }
        
        # Saludos y despedidas
        saludos = ["hola", "buenas", "buenos dias", "buenas tardes", "buenas noches","holaa","holaaa","holaaaa","holaaaaa"]
        despedidas = ["adios", "bye", "hasta luego", "nos vemos"]

        if any(s in pregunta_limpia for s in saludos):
            return {
                "respuesta": "¡Hola! Soy el asistente de información de la UAM. ¿En qué te puedo ayudar hoy?",
                "fuente": "Sistema", "confianza": 1.0, "categoria": "saludo", "opciones": []
            }

        if any(s in pregunta_limpia for s in despedidas):
            return {
                "respuesta": "¡Hasta luego! Si tienes más dudas sobre trámites o salones, aquí estaré.",
                "fuente": "Sistema", "confianza": 1.0, "categoria": "despedida", "opciones": []
            }
        

        # =========================================================
        # CLASIFICADOR DE INTENCIONES
        # =========================================================
        
        # 1. Generamos el embedding (convertido a tensor para usar util.cos_sim)
        emb_usuario = self.modelo.encode(pregunta_limpia, convert_to_tensor=True)
        
        # --- FILTRO 1: Umbral de Similitud de Coseno Máxima ---
        # Comparamos el texto del usuario contra TODOS los embeddings de la DB
        # self.embeddings debe ser un tensor o convertirse a uno
        similitudes = util.cos_sim(emb_usuario, self.embeddings)
        max_similitud_coseno = float(similitudes.max())
        
        UMBRAL_COSENO = 0.50  # Ajusta este valor (0.40 - 0.55 suele ser el punto dulce)
        print(f"DEBUG -> Similitud Coseno Máxima: {max_similitud_coseno:.3f}")
        
        if max_similitud_coseno < UMBRAL_COSENO:
            return {
                "respuesta": "Lo siento, no estoy seguro de entender tu pregunta. ¿Podrías reformularla?",
                "fuente": "Sistema (Filtro Coseno)", 
                "confianza": round(max_similitud_coseno, 3), 
                "categoria": "desconocido", 
                "opciones": []
            }

        # 2. Si pasa el filtro de coseno, procedemos a la Regresión Logística
        # Convertimos a numpy array para sklearn
        emb_numpy = emb_usuario.cpu().numpy().reshape(1, -1) if hasattr(emb_usuario, 'cpu') else emb_usuario.reshape(1, -1)
        probabilidades = self.clasificador.predict_proba(emb_numpy)[0]
        clases = self.clasificador.classes_
        
        # Ordenamos las probabilidades de mayor a menor
        indices_ordenados = np.argsort(probabilidades)[::-1]
        
        mejor_idx = indices_ordenados[0]
        segundo_idx = indices_ordenados[1] if len(indices_ordenados) > 1 else None
        
        categoria_final = clases[mejor_idx]
        confianza_final = float(probabilidades[mejor_idx])
        
        # --- FILTRO 2: Margen de confianza (Diferencia entre 1ro y 2do lugar) ---
        UMBRAL_REGRESION = 0.35  # Subimos un poco el umbral ya que pasó el filtro de coseno
        MARGEN_MINIMO = 0.02     # Exigimos que el ganador le gane al segundo
        
        if segundo_idx is not None:
            margen_real = confianza_final - float(probabilidades[segundo_idx])
        else:
            margen_real = confianza_final

        print(f"DEBUG -> Ganador: {categoria_final} ({confianza_final:.3f}), Margen: {margen_real:.3f}")

        if confianza_final < UMBRAL_REGRESION or margen_real < MARGEN_MINIMO:
            return {
                "respuesta": "Lo siento, tu pregunta se parece a varias cosas que sé, pero no estoy completamente seguro de cuál necesitas. ¿Podrías ser más específico?",
                "fuente": "Sistema (Incertidumbre)", 
                "confianza": round(confianza_final, 3), 
                "categoria": "desconocido", 
                "opciones": []
            }

        # =========================================================
        # INTERCEPCIÓN PARA SOLICITAR LA CARRERA (CON BOTONES)
        # =========================================================
        CATEGORIAS_CONTEXTUALES = ["inscripcion", "recuperacion", "salones"]

        if categoria_final in CATEGORIAS_CONTEXTUALES and not slots.get("carrera"):
            slots["esperando_carrera"] = True
            slots["intencion_pendiente"] = categoria_final
            
            return {
                "respuesta": "Para brindarte la información exacta y guiarte con las personas correctas, por favor <b>selecciona tu carrera</b>:",
                "fuente": "Sistema",
                "confianza": round(confianza_final, 3),
                "categoria": "solicitud_carrera",
                "opciones": ["Computación", "Eléctrica"]
            }

        respuesta_base, fuente = obtener_respuesta_db(self.ruta_db, categoria_final)
        
        if slots.get("carrera") and categoria_final in CATEGORIAS_CONTEXTUALES:
            respuesta_final = self.personalizar_respuesta(respuesta_base, categoria_final, slots["carrera"])
        else:
            respuesta_final = respuesta_base

        return {
            "respuesta": respuesta_final,
            "fuente": fuente,
            "confianza": round(confianza_final, 3),
            "categoria": categoria_final,
            "opciones": []
        }