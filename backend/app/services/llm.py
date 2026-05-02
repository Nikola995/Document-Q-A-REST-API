import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from fastapi import Request
from loguru import logger
import time
from app.core.config import settings


def load_qa_model() -> dict:
    logger.info("Loading QA tokenizer", model=settings.QA_MODEL)
    tokenizer = DistilBertTokenizer.from_pretrained(settings.QA_MODEL)
    logger.info("Loading QA model", model=settings.QA_MODEL)
    model = DistilBertForQuestionAnswering.from_pretrained(settings.QA_MODEL)
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


async def generate_answer(
    question: str, context: str, session_id: str, request: Request
) -> str:
    """Calls the configured LLM and returns a string answer.

    Args:
        question (str): The user's query
        context (str): The retrieved context

    Returns:
        str: The answer
    """
    model = request.app.state.qa_model["model"]
    tokenizer = request.app.state.qa_model["tokenizer"]

    llm_time_start = time.perf_counter()
    # DistilBERT has a 512 token limit — truncation is a temporary measure.
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
    llm_time_total = time.perf_counter() - llm_time_start
    logger.info(
        "llm_success",
        extra={
            "session_id": session_id,
            "question": question,
            "answer": result,
            "context_length_tokens": len(inputs["input_ids"][0]),
            "answer_length_tokens": int(answer_end_index - answer_start_index) + 1,
            "truncated": len(inputs["input_ids"][0]) == 512,
            "execution_time_s": llm_time_total,
        },
    )
    return result
