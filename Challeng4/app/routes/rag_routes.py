from fastapi import APIRouter, HTTPException
from models.historia_model import Historia
from services.historias import cargar_documento, generar_embedding_documento, obtener_documentos_mas_relevantes, obtener_respuesta

rag_routes = APIRouter()

@rag_routes.post("/upload")
async def add_document(historia: Historia):
    """Agrega una nueva historia"""
    if (historia.title != None and historia.content != None):
        try: 
            return cargar_documento(historia)
        except RuntimeError as e:
            print(e)
            raise HTTPException(status_code=503, detail="Servicio no disponible")
    else:
        raise HTTPException(status_code=400, detail="Solicitud inválida")

@rag_routes.post("/generate-embeddings")
async def generate_embedding(document_id: str):
    """ Genera embeddings para un documento específico o para todos los documentos almacenados."""
    try:
        return generar_embedding_documento(document_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="No se encuentró el documento con id")
    except Exception:
        return HTTPException(503, "Servicio no disponible")
    
@rag_routes.post("/search")
async def get_document(query: str):
    """ Busca los documentos más relvantes basados en una consulta."""
    if query != None:
        try:
            return obtener_documentos_mas_relevantes(query)
        except ValueError:
            raise HTTPException(status_code=404, detail="No se encontraron coincidencias")
        except RuntimeError as e:
            print(e)
            raise HTTPException(status_code=503, detail="Servicio no disponible")
        except Exception:
            raise HTTPException(status_code=400, detail="Ocurrio un error inesperado")
    else:
        raise HTTPException(status_code=400, detail="Solicitud inválida")
    
@rag_routes.post("/ask")
async def get_response(question: str):
    """ Genera una respuesta a una pregunta utilizando los documentos relevantes."""
    if question != None:
        try:
            return obtener_respuesta(question)
        except RuntimeError:
            return HTTPException(503, "Servicio no disponible")
        except Exception:
                raise HTTPException(status_code=400, detail="Ocurrio un error inesperado")
    else:
        raise HTTPException(status_code=400, detail="Solicitud inválida")
    