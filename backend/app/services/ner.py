from fastapi import Request
from gliner2 import GLiNER2
from app.models.schemas import Entity
from app.core.config import settings


def load_ner_model() -> dict:
    return {"model": GLiNER2.from_pretrained(settings.NER_MODEL)}


async def extract_entities(text: str, request: Request) -> list[Entity]:
    model = request.app.state.ner_model["model"]
    raw = model.extract_entities(text, settings.NER_LABELS)

    entities = [
        Entity(text=entity_text, label=label)
        for label, entity_texts in raw["entities"].items()
        for entity_text in entity_texts
    ]
    return entities
