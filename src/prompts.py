from __future__ import annotations

from typing import Any

from .logging_utils import token_ids_sha256


def make_messages(
    question: str, system_message: str | None = None
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": question})
    return messages


def build_prompt(
    tokenizer: Any,
    question: str,
    enable_thinking: bool,
    system_message: str | None = None,
) -> dict[str, Any]:
    messages = make_messages(question, system_message)
    template_kwargs = {
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, **template_kwargs)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    token_ids = [int(item) for item in input_ids]
    return {
        "prompt": prompt,
        "input_token_ids": token_ids,
        "input_token_count": len(token_ids),
        "input_token_ids_sha256": token_ids_sha256(token_ids),
    }


def split_reasoning_and_answer(
    tokenizer: Any, generated_ids: list[int]
) -> tuple[str, str, int, bool]:
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    try:
        close_position = generated_ids.index(close_id)
    except (ValueError, TypeError):
        full = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if "</think>" in full:
            reasoning, answer = full.split("</think>", maxsplit=1)
            reasoning = reasoning.replace("<think>", "", 1).strip()
            return (
                reasoning,
                answer.strip(),
                len(tokenizer.encode(reasoning, add_special_tokens=False)),
                True,
            )
        # An unclosed <think> is a right-censored reasoning trace, not a
        # zero-length trace. Keep its observed length and mark it incomplete.
        reasoning = full.replace("<think>", "", 1).strip()
        return reasoning, full, len(generated_ids), False

    reasoning_ids = generated_ids[:close_position]
    final_ids = generated_ids[close_position + 1 :]
    reasoning = tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip()
    final = tokenizer.decode(final_ids, skip_special_tokens=True).strip()
    return reasoning, final, len(reasoning_ids), True
