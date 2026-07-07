from sentence_transformers import SentenceTransformer
import sqlite3
import numpy as np

# cargar modelo
modelo = SentenceTransformer('all-MiniLM-L6-v2')

# conectar base
conn = sqlite3.connect("../db/chatbot2.db")
cursor = conn.cursor()

# obtener preguntas
cursor.execute("SELECT rowid, texto FROM preguntas2")
datos = cursor.fetchall()

print(f"Generando embeddings para {len(datos)} preguntas2...")

for fila in datos:

    rowid = fila[0]
    texto = fila[1]

    # generar embedding
    embedding = modelo.encode(texto)

    # convertir a bytes (para SQLite)
    embedding_bytes = embedding.astype(np.float32).tobytes()

    # guardar embedding
    cursor.execute(
        "UPDATE preguntas2 SET embedding = ? WHERE rowid = ?",
        (embedding_bytes, rowid)
    )

conn.commit()
conn.close()

print("Embeddings generados correctamente.")