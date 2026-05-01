import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from fastapi import Request
from app.core.config import settings


def load_qa_model() -> dict:
    tokenizer = DistilBertTokenizer.from_pretrained(settings.QA_MODEL)
    model = DistilBertForQuestionAnswering.from_pretrained(settings.QA_MODEL)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


async def generate_answer(question: str, context: str, request: Request) -> str:
    """Calls the configured LLM and returns a string answer.

    Args:
        question (str): The user's query
        context (str): The retrieved context

    Returns:
        str: The answer
    """
    model = request.app.state.qa_model["model"]
    tokenizer = request.app.state.qa_model["tokenizer"]

    # DistilBERT has a 512 token limit — truncation is a temporary measure.
    # TODO: Chunking + retrieval (next iteration) will handle this properly by only
    # passing the most relevant chunks as context instead of the full document.
    inputs = tokenizer(
        question, context, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()
    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    result = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
    return result
