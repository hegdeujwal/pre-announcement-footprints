"""On-demand LLM summaries of a filing (src/live/summaries.py).

Display only, after the filing is public. No network here: the EDGAR client
and the Groq session are fakes.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.live import summaries
from src.utils.config import load_config

BODY = (b"<html><body><p>Item 2.02 Results of Operations.</p><p>On October 1, "
        b"2026, Example Inc. issued a press release announcing its results for "
        b"the quarter ended September 30, 2026. A copy is attached as Exhibit "
        b"99.1 and is incorporated herein by reference.</p></body></html>")
EXHIBIT = (b"<html><body><p>Example Inc. reports revenue of $1.2 billion, up 8% "
           b"from a year earlier, and raises its full-year outlook for 2026 "
           b"after a strong quarter in every region.</p></body></html>")
ACC = "0000000001-26-000001"


class FakeClient:
    def __init__(self, files: dict[str, bytes], listing: list[str] | None = None):
        self.files, self.listing, self.calls = files, listing, []

    def get_bytes(self, url):
        self.calls.append(url)
        name = url.rsplit("/", 1)[-1]
        if name not in self.files:
            raise RuntimeError(f"HTTP 404 for {url}")
        return self.files[name]

    def get_json(self, url):
        if self.listing is None:
            raise RuntimeError("no listing")
        return {"directory": {"item": [{"name": n} for n in self.listing]}}


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload, headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, resp):
        self.resp, self.sent = resp, []

    def post(self, url, **kw):
        self.sent.append(kw["json"])
        return self.resp


def ok(text="Example Inc. reported results.\n- Revenue $1.2 billion"):
    return FakeSession(FakeResp(200, {"choices": [{"message": {"content": text}}]}))


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    c = copy.deepcopy(load_config())
    c["live"]["summaries"]["dir"] = str(tmp_path / "summaries")
    monkeypatch.setenv(c["live"]["summaries"]["api_key_env"], "test-key")
    return c


def client():
    return FakeClient({"ex-8k.htm": BODY, "ex99-1.htm": EXHIBIT},
                      ["ex-8k.htm", "ex99-1.htm", "R1.htm", "Financial_Report.xlsx"])


def test_text_is_the_body_then_the_press_release(cfg):
    text, sources = summaries.filing_text(cfg, client(), "1", ACC, "ex-8k.htm")
    assert text.index("Example Inc. issued") < text.index("revenue of $1.2 billion")
    assert [s.rsplit("/", 1)[-1] for s in sources] == ["ex-8k.htm", "ex99-1.htm"]


def test_text_is_cut_to_the_budget(cfg):
    cfg["live"]["summaries"]["max_input_chars"] = 300
    text, _ = summaries.filing_text(cfg, client(), "1", ACC, "ex-8k.htm")
    assert len(text) <= 300


def test_body_alone_when_the_folder_listing_fails(cfg):
    c = FakeClient({"ex-8k.htm": BODY}, listing=None)
    _, sources = summaries.filing_text(cfg, c, "1", ACC, "ex-8k.htm")
    assert len(sources) == 1


@pytest.mark.parametrize("files,doc", [({}, "ex-8k.htm"),
                                       ({"e.htm": b"<p>404</p>"}, "e.htm"),
                                       ({}, None)])
def test_nothing_readable_fails_loudly(cfg, files, doc):
    with pytest.raises(summaries.SummaryError):
        summaries.filing_text(cfg, FakeClient(files, []), "1", ACC, doc)


def test_prompt_carries_items_ticker_and_text(cfg):
    msgs = summaries.messages(cfg, "FILING TEXT", ["2.02"], "EXM")
    user = msgs[1]["content"]
    assert "EXM" in user and "2.02" in user and "FILING TEXT" in user
    assert "investment advice" in msgs[0]["content"]


def test_missing_key_is_named(cfg, monkeypatch):
    monkeypatch.delenv(cfg["live"]["summaries"]["api_key_env"])
    with pytest.raises(summaries.SummaryError, match="GROQ_API_KEY"):
        summaries.call_groq(cfg, [], session=ok())


@pytest.mark.parametrize("resp,match", [
    (FakeResp(401), "API key"),
    (FakeResp(429, headers={"retry-after": "7"}), "7 seconds"),
    (FakeResp(500, {"error": "x"}), "HTTP 500"),
    (FakeResp(200, {"choices": []}), "no summary"),
])
def test_groq_failures_say_why(cfg, resp, match):
    with pytest.raises(summaries.SummaryError, match=match):
        summaries.call_groq(cfg, [], session=FakeSession(resp))


def test_request_uses_the_configured_model(cfg):
    s = ok()
    summaries.call_groq(cfg, [{"role": "user", "content": "x"}], session=s)
    assert s.sent[0]["model"] == cfg["live"]["summaries"]["model"]


def test_generated_once_then_read_from_disk(cfg):
    s = ok()
    rec = summaries.generate(cfg, ACC, "1", "ex-8k.htm", "2.02,9.01",
                             ticker="EXM", client=client(), session=s)
    assert rec["items"] == ["2.02", "9.01"] and rec["model"]
    assert Path(cfg["live"]["summaries"]["dir"], f"{ACC}.json").exists()
    again = summaries.generate(cfg, ACC, "1", "ex-8k.htm", "2.02",
                               client=FakeClient({}), session=FakeSession(None))
    assert again == rec and len(s.sent) == 1
    assert summaries.load(cfg, ACC) == rec


def test_force_regenerates(cfg):
    summaries.generate(cfg, ACC, "1", "ex-8k.htm", "2.02", client=client(),
                       session=ok("first"))
    rec = summaries.generate(cfg, ACC, "1", "ex-8k.htm", "2.02", client=client(),
                             session=ok("second"), force=True)
    assert rec["summary"] == "second" == summaries.load(cfg, ACC)["summary"]


def test_a_failed_call_saves_nothing(cfg):
    with pytest.raises(summaries.SummaryError):
        summaries.generate(cfg, ACC, "1", "ex-8k.htm", "2.02", client=client(),
                           session=FakeSession(FakeResp(429)))
    assert summaries.load(cfg, ACC) is None


def test_no_detector_feature_or_label_code_reads_summaries():
    """The text exists only after t0: nothing that detects may import it."""
    root = Path(__file__).resolve().parents[1] / "src"
    for pkg in ("pipeline", "baselines", "rl", "eval", "collectors"):
        for f in (root / pkg).rglob("*.py"):
            src = f.read_text()
            assert "live.summaries" not in src, f
            assert "live import summaries" not in src, f


def test_missing_key_fetches_nothing_from_sec(cfg, monkeypatch):
    monkeypatch.delenv(cfg["live"]["summaries"]["api_key_env"])
    c = client()
    with pytest.raises(summaries.SummaryError, match="GROQ_API_KEY"):
        summaries.generate(cfg, ACC, "1", "ex-8k.htm", "2.02", client=c, session=ok())
    assert c.calls == []


def test_the_cover_page_is_skipped(cfg):
    cover = (b"<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>"
             b"<p>175 West Jackson Boulevard, Chicago</p>") + BODY
    text, _ = summaries.filing_text(cfg, FakeClient({"a.htm": cover}, []), "1",
                                    ACC, "a.htm")
    assert "Jackson" not in text and "Item 2.02" in text
