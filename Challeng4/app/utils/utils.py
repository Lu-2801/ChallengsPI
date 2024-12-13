from data.historias import collection
import cohere

co = cohere.ClientV2()

def existe_documento(id_documento: str):
    # Recupera los IDs almacenados
    ids = collection.get(ids=None)["ids"]

    for id in ids:
        if id_documento == id:
            return True
    
    return False


def get_embeddings(textos):
    try:
        response = co.embed(
            texts=textos,
            model="embed-multilingual-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )
        
        return response.embeddings.float_
    except Exception as general_error:
        #manejo de error generico ya que tengo problemas para importar errores de la libreria de cohere.error
        
        raise RuntimeError("Fallo inesperado al procesar embeddings") from general_error
    
def get_documents(consulta):
    try:
        results = collection.query(
            query_embeddings=consulta,
            n_results=2
        )

        return results
    except Exception as general_error:
        #manejo de error generico 
        
        raise RuntimeError("Fallo inesperado al guardar en la base de datos") from general_error

def llm(pregunta, contexto):
    system_message = """Eres un asistente diseñado exclusivamente para responder preguntas basandote en el contexto"""
    
    
    prompt = f""" 
                ###
                Instrucciones: 
                - Responde la pregunta solo utilizando el contexto.
                - Si la pregunta no tiene relacion alguna con el contexto, la respuesta debe ser 'Lo siento no puedo ayudarte con eso'.
    
                ###
                Contexto:
                {contexto}
    
                ###
                Pregunta:
                {pregunta}
    
                """

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    
    try:
        "llamada al modelo para responder a la pregunta ingresada"
        response_pregunta = co.chat(
            model="command-r-plus-08-2024",
            messages=messages,
            temperature=0,
            seed=42,
        )
    
        respuesta = response_pregunta.message.content[0].text
        
        return respuesta
    except Exception as general_error:
        #manejo de error generico ya que tengo problemas para importar errores de la libreria de cohere.error
        
        raise RuntimeError("Fallo inesperado generar la respuesta") from general_error
