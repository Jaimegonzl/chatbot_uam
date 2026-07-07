from sentence_transformers import SentenceTransformer, util
import sqlite3
import numpy as np


def cargar_embeddings_db(ruta_db):

    conn = sqlite3.connect(ruta_db)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT texto, categoria, embedding
        FROM preguntas2
        WHERE embedding IS NOT NULL
    """)

    datos = cursor.fetchall()
    conn.close()

    textos = []
    categorias = []
    embeddings = []
    

    for texto, categoria, emb in datos:
        textos.append(texto)
        categorias.append(categoria)

        vector = np.frombuffer(emb, dtype=np.float32)
        embeddings.append(vector)

    return textos, categorias, np.array(embeddings)


def obtener_respuesta_db(ruta_db, categoria):

    conn = sqlite3.connect(ruta_db)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT respuesta, fuente
        FROM faq
        WHERE categoria = ?
    """, (categoria,))

    resultado = cursor.fetchone()
    conn.close()

    return resultado