
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.utils import existe_documento, get_documents, get_embeddings, llm
from models.historia_model import Historia
from data.historias import collection



"""Funcion que se encarga de guardar en la base de datos vectorial las historias (sus embeddings)"""
def cargar_documento(historia: Historia):
    # Elijo esta tecnica para que se creen varios fragmentos con una historia, 
    # ya que el modelo de embbeding de cohere recomienda pasarle varios chunk de
    # pocos tokens (max 512), más que chunk extensos
    
    ##
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=2500, chunk_overlap=200)

    # Divido el texto en chucks (fragmentos)
    texts = text_splitter.split_text(historia.content)

    try:
        # Obtengo los ombbedings de los chucks generados
        cohere_embeddings =  get_embeddings(texts)

        # Genero id para las historias apartir de la cantidad de historias guardadas en base
        cant = collection.count()
        ids = []
        metadata_titulo = []
        for i, _ in enumerate(texts):
            ids.append(f"id{i + cant}")
            metadata_titulo.append({"titulo": historia.title})

        print(len(ids))
        print(len(texts))
        print(len(cohere_embeddings))
        print(len(metadata_titulo))

        # Cargo la data en la base
        collection.add(
            documents=texts,
            ids=ids,
            metadatas=metadata_titulo,
            embeddings=cohere_embeddings
        )

        print("todo ok2")

        return {
            "message": "documento guardado exitosamente",
            "document_ids": ids
        }
    
    # Si falla el servicio de cohere
    except RuntimeError:
        raise RuntimeError
    except Exception:
        raise Exception

def generar_embedding_documento(id_documento: str):
    # corroborar que el id existe
    if existe_documento(id_documento):
        resultado = collection.get(ids=[id_documento])
        try:
            get_embeddings(resultado["documents"])
            return {"message": f"Embedding generado correctamente para el documento {id_documento}"}
        except RuntimeError:
            raise RuntimeError
        except Exception:
            raise Exception
    else:
        raise KeyError
    
def obtener_documentos_mas_relevantes(consulta: str):
      
    try:
        frase_base = [consulta]
        # generar embedding de la consulta
        embbed_base = get_embeddings(frase_base)

        # buscar documentos coincidentes
        resultado = get_documents(embbed_base)

        # corrobora que traiga coincidencias
        if len(resultado["documents"][0]) == 0:
            raise ValueError
        
        print(resultado["documents"][0])

        documentos_relevantes = []

        # Extrae los datos del diccionario
        ids = resultado.get("ids", [[]])[0]
        metadatas = resultado.get("metadatas", [[]])[0]
        documents = resultado.get("documents", [[]])[0]
        distances = resultado.get("distances", [[]])[0]

        # Itera sobre los resultados y construye la respuesta
        for doc_id, metadata, document, distance in zip(ids, metadatas, documents, distances):
            documentos_relevantes.append({
                "document_id": doc_id,
                "title": metadata.get("titulo", "Sin título"),
                "content_snippet": document[:200] + "..." if len(document) > 200 else document,
                "similarity_score": distance
            })

        return {"results": documentos_relevantes}

    except RuntimeError:
        raise RuntimeError
    except Exception:
        raise Exception
    

def obtener_respuesta(pregunta):
    try:
        # OBTENGO EL EMBBEDING DE LA PREGUNTA (ya que utilizamos otro modelo en vez de utilizar el por defecto)
        frase_base = [pregunta]
        embbed_base = get_embeddings(frase_base)


        # en esta lista guardo todos los documentos que retorna la base como resultado y que sera utilizada como contexto
        contexto = []

        # CONSULTA A LA BASE VECTORIAL
        # pido que me retorne 7 resultados por la manera en que estan divididos los chucks, ya que tal vez 1 historia este dividida en 6 chucks aprox, 
        # por ende asi creo que traeria todos los chuck que coincidan y se cree un contexto mas preciso, o sea si necesita toda la historia para resolver la pregunta obtendriamos todo el contexto necesario
        results = get_documents(embbed_base)
        
        # extraigo solo los documentos de la respuesta
        contexto = results["documents"][0]

        # llamo a la funcion que me retornara la respuesta utilizando la pregunta y el contexto (los docs) 
        respuesta_usuario = llm(pregunta, contexto)
        
        return {"answer": respuesta_usuario}
    except RuntimeError:
        raise RuntimeError
    except Exception:
        raise Exception
