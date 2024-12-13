from pydantic import BaseModel


class Historia(BaseModel):
    title: str
    content: str
