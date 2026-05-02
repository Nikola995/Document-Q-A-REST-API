from fastapi import Request
from gliner2 import GLiNER2
from app.models.schemas import Entity
from loguru import logger
import time
from app.core.config import settings


def load_ner_model() -> dict:
    logger.info("Loading NER model", model=settings.NER_MODEL)
    return {"model": GLiNER2.from_pretrained(settings.NER_MODEL)}


async def extract_entities(
    text: str, session_id: str, request: Request
) -> list[Entity]:
    ner_time_start = time.perf_counter()
    model = request.app.state.ner_model["model"]
    raw = model.extract_entities(text, settings.NER_LABELS)

    entities = [
        Entity(text=entity_text, label=label)
        for label, entity_texts in raw["entities"].items()
        for entity_text in entity_texts
    ]
    ner_time_total = time.perf_counter() - ner_time_start
    logger.info(
        "ner_success",
        extra={
            "session_id": session_id,
            "answer": text,
            "num_entities": len(entities),
            "execution_time_s": ner_time_total,
        },
    )
    return entities
