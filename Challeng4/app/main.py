from fastapi import FastAPI
from routes.rag_routes import rag_routes  # Importa el objeto correctamente

app = FastAPI()

# Montar las rutas
app.include_router(rag_routes)
