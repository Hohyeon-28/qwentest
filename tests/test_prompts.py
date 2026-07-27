from src.prompts import split_reasoning_and_answer


class DummyTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "</think>"
        return 99

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if 99 in token_ids:
            return "<think>work</think>answer"
        return "<think>unfinished work"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text.split())))


def test_unclosed_reasoning_keeps_observed_length_and_is_incomplete():
    reasoning, final, count, complete = split_reasoning_and_answer(
        DummyTokenizer(), [1, 2, 3]
    )
    assert reasoning == "unfinished work"
    assert final == "<think>unfinished work"
    assert count == 3
    assert not complete


def test_closed_reasoning_is_complete():
    reasoning, final, count, complete = split_reasoning_and_answer(
        DummyTokenizer(), [1, 2, 99, 3]
    )
    assert reasoning == "<think>unfinished work"
    assert final == "<think>unfinished work"
    assert count == 2
    assert complete
