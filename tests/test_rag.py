from app.rag import build_citations, cited_ids, format_context
from app.schemas import Message
from tests.conftest import DOCS, make_rag


def test_format_context_numbers_chunks_with_location():
    ctx = format_context([d for d, _ in DOCS])
    assert ctx.startswith("[1] (Medical_book.pdf, p. 12)\nAcne is a common skin disease.")
    assert "[2] (Medical_book.pdf, p. 13)" in ctx
    assert "[3] (Medical_book.pdf)\n" in ctx  # no page metadata


def test_cited_ids():
    assert cited_ids("Foo [1]. Bar [3][1].") == {1, 3}
    assert cited_ids("no citations") == set()


def test_build_citations_maps_metadata_and_cited_flags():
    cits = build_citations(DOCS, "Acne is a skin disease [1] caused by pores [2].")
    assert [c.id for c in cits] == [1, 2, 3]
    assert cits[0].page == 12 and isinstance(cits[0].page, int)
    assert cits[0].source == "Medical_book.pdf"
    assert cits[0].score == 0.91
    assert [c.cited for c in cits] == [True, True, False]
    assert cits[2].page is None


def test_snippet_is_truncated():
    from langchain_core.documents import Document

    long = [(Document(page_content="x " * 500, metadata={"source": "a.pdf"}), None)]
    snippet = build_citations(long)[0].snippet
    assert len(snippet) <= 301 and snippet.endswith("…")


async def test_condense_skipped_without_history(settings):
    rag, store = make_rag(settings, ["Answer [1]."])
    answer, cits = await rag.answer("What is acne?", [])
    assert store.queries == ["What is acne?"]
    assert answer == "Answer [1]."
    assert cits[0].cited


async def test_condense_used_with_history(settings):
    rag, store = make_rag(settings, ["What causes acne?", "Clogged pores [2]."])
    history = [Message(role="user", content="What is acne?"), Message(role="assistant", content="A skin disease [1].")]
    answer, _ = await rag.answer("What causes it?", history)
    assert store.queries == ["What causes acne?"]
    assert answer == "Clogged pores [2]."


async def test_stream_event_order(settings):
    rag, _ = make_rag(settings, ["Hi [1]."])
    events = [e async for e in rag.stream("q", [])]
    assert events[0].type == "citations"
    assert all(e.type == "token" for e in events[1:-1])
    assert "".join(e.data for e in events[1:-1]) == "Hi [1]."
    assert events[-1].type == "done"
    assert events[-1].data["answer"] == "Hi [1]."
    assert events[-1].data["citations"][0].cited is True


# --- Relevance threshold ----------------------------------------------------

from app.rag import NO_CONTEXT_ANSWER  # noqa: E402
from tests.conftest import OFF_TOPIC  # noqa: E402


async def test_low_score_chunks_are_filtered(settings):
    rag, _ = make_rag(settings, ["Acne [1]."])  # stub returns DOCS + LOW_DOC (0.20)
    _, cits = await rag.answer("What is acne?", [])
    assert [c.score for c in cits] == [0.91, 0.85, 0.40]  # 0.40 == threshold is kept (>=)
    assert all("Sourdough" not in c.snippet for c in cits)


async def test_out_of_scope_skips_llm(settings):
    # FakeListChatModel with no responses raises if it is ever called.
    rag, store = make_rag(settings, [], results=OFF_TOPIC)
    answer, cits = await rag.answer("Who won the world cup?", [])
    assert answer == NO_CONTEXT_ANSWER
    assert cits == []
    assert store.queries == ["Who won the world cup?"]


async def test_out_of_scope_stream_has_normal_event_shape(settings):
    rag, _ = make_rag(settings, [], results=OFF_TOPIC)
    events = [e async for e in rag.stream("bread?", [])]
    assert [e.type for e in events] == ["citations", "token", "done"]
    assert events[0].data == []
    assert events[1].data == NO_CONTEXT_ANSWER
    assert events[2].data == {"answer": NO_CONTEXT_ANSWER, "citations": []}


async def test_threshold_zero_disables_filter(settings):
    rag, _ = make_rag(settings.model_copy(update={"min_relevance_score": 0.0}), ["x"], results=OFF_TOPIC)
    _, cits = await rag.answer("bread?", [])
    assert len(cits) == 2


# --- Page numbers -------------------------------------------------------------


def test_page_is_not_shifted_again():
    from langchain_core.documents import Document
    from app.rag import _page

    # Stored values are already 1-based (converted at ingest); Pinecone returns floats.
    assert _page(Document(page_content="x", metadata={"page": 43.0})) == 43
    assert _page(Document(page_content="x", metadata={"page": 43})) == 43
    assert _page(Document(page_content="x", metadata={})) is None


# --- Follow-up rewrite (condense) ---------------------------------------------------

from app.rag import (  # noqa: E402
    check_rewrite,
    clean_rewrite,
    condense_prompt,
    format_transcript,
    normalize_citations,
)

# The real output gemma3:4b produced instead of a rewrite.
REAL_BAD_REWRITE = (
    "Teenagers are more likely to develop acne due to hormonal changes [1]. Boys have more "
    "severe acne and develop it more often than girls [1]."
)


def _turns(*pairs):
    msgs = []
    for q, a in pairs:
        msgs += [Message(role="user", content=q), Message(role="assistant", content=a)]
    return msgs


def test_format_transcript_keeps_last_turns_labelled_and_clean():
    history = _turns(*[(f"q{i}", f"a{i} [1][2, 3]") for i in range(5)])
    lines = format_transcript(history).splitlines()
    assert lines == ["User: q2", "Assistant: a2", "User: q3", "Assistant: a3", "User: q4", "Assistant: a4"]


def test_format_transcript_truncates_long_answers():
    text = format_transcript(_turns(("What is acne?", "word " * 200)), max_chars=50)
    assert text.splitlines()[1].endswith("…") and len(text.splitlines()[1]) <= len("Assistant: ") + 51


def test_clean_rewrite_accepts_and_tidies_questions():
    assert clean_rewrite("Who usually gets acne?", "Who usually gets it?") == "Who usually gets acne?"
    assert clean_rewrite('Standalone question: "Who usually gets acne?"\n', "Who usually gets it?") == "Who usually gets acne?"
    assert clean_rewrite("  How is acne treated?  \nextra line", "How is it treated?") == "How is acne treated?"


def test_clean_rewrite_rejects_answers():
    q = "Who usually gets it?"
    assert clean_rewrite(REAL_BAD_REWRITE, q) is None                        # the real failure
    assert clean_rewrite("Who gets acne [1]?", q) is None                    # cites a passage
    assert clean_rewrite("Is it " + "very " * 60 + "common?", q) is None     # far too long
    assert clean_rewrite("   ", q) is None


LONG_STATEMENT = (
    "Acne is a skin condition that teens get and it is common in many people who are young "
    "and it can be mild or bad in some cases for a while"
)


def test_check_rewrite_reasons():
    q = "Who usually gets it?"
    assert check_rewrite("", q)[1] == "empty"
    assert check_rewrite("Teens get acne [1].", q)[1] == "has_citation"
    assert check_rewrite("Is it " + "very " * 60 + "common?", q)[1] == "too_long"
    assert len(LONG_STATEMENT.split()) > 30 and len(LONG_STATEMENT) < 150
    assert check_rewrite(LONG_STATEMENT, q)[1] == "looks_like_answer"
    assert check_rewrite(LONG_STATEMENT + "?", q)[1] is None  # a long *question* is fine


def test_check_rewrite_accepts_short_statements_and_has_a_length_floor():
    """Rules: a statement is only an 'answer' if it's also > 30 words; 3x length has a 150-char floor."""
    assert check_rewrite("Teenagers usually get acne.", "Who usually gets it?")[1] is None
    short_q = "Who gets it?"  # 12 chars: plain 3x would cap a rewrite at 36 chars
    rewrite = "Who usually gets acne vulgaris and rosacea?"
    assert len(rewrite) > 3 * len(short_q)
    assert check_rewrite(rewrite, short_q) == (rewrite, None)


def test_condense_prompt_is_always_two_messages():
    """History goes in as a transcript, never as extra chat turns the model would 'reply' to."""
    for n in (1, 3, 8):
        history = _turns(*[(f"q{i}", f"a{i}") for i in range(n)])
        msgs = condense_prompt.invoke({"transcript": format_transcript(history), "question": "q?"}).to_messages()
        assert [m.type for m in msgs] == ["system", "human"]
        assert "Follow-up question: q?" in msgs[1].content


async def test_condense_falls_back_when_model_answers(settings):
    history = _turns(("What is acne?", "Acne is a skin disease [1]."))
    rag, store = make_rag(settings, [REAL_BAD_REWRITE, "Teens [1]."])
    await rag.answer("Who usually gets it?", history)
    # Dual retrieval: the plain follow-up AND previous question + follow-up (in parallel).
    assert sorted(store.queries) == ["What is acne? Who usually gets it?", "Who usually gets it?"]


async def test_condense_uses_previous_user_question_not_assistant(settings):
    history = _turns(("What is acne?", "A [1]."), ("What causes it?", "Pores [2]."))
    rag, store = make_rag(settings, ["Teenagers do [1].", "x"])
    await rag.answer("Who gets it?", history)
    assert sorted(store.queries) == ["What causes it? Who gets it?", "Who gets it?"]


# --- Citation lists -------------------------------------------------------------------


def test_normalize_citations_lists_and_ranges():
    assert normalize_citations("Foo [1, 3].", 4) == "Foo [1][3]."
    assert normalize_citations("Foo [1,3] bar [2; 4]", 4) == "Foo [1][3] bar [2][4]"
    assert normalize_citations("Foo [1-3] and [2–4]", 4) == "Foo [1][2][3] and [2][3][4]"
    assert normalize_citations("Already [1][2].", 4) == "Already [1][2]."


def test_normalize_citations_leaves_non_citations_alone():
    assert normalize_citations("Published [2023].", 4) == "Published [2023]."
    assert normalize_citations("Out of range [7, 9].", 4) == "Out of range [7, 9]."
    assert normalize_citations("Backwards [3-1].", 4) == "Backwards [3-1]."


async def test_grouped_citations_marked_cited_in_answer(settings):
    rag, _ = make_rag(settings, ["Acne is common [1, 3]."])
    answer, cits = await rag.answer("What is acne?", [])
    assert answer == "Acne is common [1][3]."
    assert [c.cited for c in cits] == [True, False, True]


async def test_grouped_citations_normalized_in_stream_done(settings):
    rag, _ = make_rag(settings, ["Acne [1-2]."])
    events = [e async for e in rag.stream("What is acne?", [])]
    done = events[-1].data
    assert done["answer"] == "Acne [1][2]."
    assert [c.cited for c in done["citations"]] == [True, True, False]
