"""scripts/langfuse_retention.py: pages through old root observations, deletes in batches."""

from types import SimpleNamespace

import pytest

from scripts import langfuse_retention as retention


class FakeApi:
    def __init__(self, pages):
        self.pages = pages
        self.list_calls = []
        self.deleted = []
        self.observations = SimpleNamespace(get_many=self.get_many)
        self.trace = SimpleNamespace(delete_multiple=lambda trace_ids: self.deleted.append(list(trace_ids)))

    def get_many(self, **kwargs):
        self.list_calls.append(kwargs)
        return self.pages[len(self.list_calls) - 1]


def page(trace_ids, cursor=None):
    return SimpleNamespace(data=[SimpleNamespace(trace_id=t) for t in trace_ids], meta=SimpleNamespace(cursor=cursor))


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    retention.get_settings.cache_clear()
    ids = [f"{i:032x}" for i in range(250)]
    api = FakeApi([page(ids[:150], cursor="next"), page(ids[150:] + ids[:5])])  # duplicates on purpose
    import langfuse

    monkeypatch.setattr(langfuse, "Langfuse", lambda **kw: SimpleNamespace(api=api, kwargs=kw))
    yield api, ids
    retention.get_settings.cache_clear()


def test_dry_run_counts_but_deletes_nothing(fake, capsys):
    api, ids = fake
    assert retention.main([]) == 0
    assert api.deleted == []
    assert "Found 250 trace(s)" in capsys.readouterr().out
    first = api.list_calls[0]
    assert first["is_root_observation"] is True and first["limit"] == 1000 and first["cursor"] is None
    assert api.list_calls[1]["cursor"] == "next"


def test_yes_deletes_in_batches_of_100(fake):
    api, ids = fake
    assert retention.main(["--days", "14", "--yes"]) == 0
    assert [len(b) for b in api.deleted] == [100, 100, 50]
    assert sum(api.deleted, []) == ids


def test_missing_keys_is_an_error(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    retention.get_settings.cache_clear()
    try:
        assert retention.main([]) == 2
    finally:
        retention.get_settings.cache_clear()
