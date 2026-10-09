"""Chat evidence remains a draft until every answer is explicitly confirmed."""

from uuid import uuid4

from fastapi.testclient import TestClient

from psych_support_bot.app import app
from psych_support_bot.infra.db.models import ConversationSession, Message
from psych_support_bot.infra.db.session import SessionLocal
from psych_support_bot.services.chat_assisted_assessment import _extract_candidates, _guided_reply


def _conversation(user_id: str, text: str) -> str:
    session_id = str(uuid4())
    with SessionLocal() as db:
        db.add(ConversationSession(id=session_id, user_id=user_id, mode="support", risk_level="low", summary=""))
        db.add(Message(session_id=session_id, role="user", content=text))
        db.commit()
    return session_id


def test_candidates_require_confirmation_and_completion_is_idempotent(monkeypatch) -> None:
    user_id = f"draft-{uuid4().hex[:8]}"
    session_id = _conversation(user_id, "过去两周，我几乎每天都感到紧张焦虑。")
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment._extract_candidates",
        lambda messages, **kwargs: ({"0": {"value": 3, "message_id": messages[0].id}}, "ready"),
    )
    client = TestClient(app)
    started = client.post(
        "/v1/assessments/chat-assisted/drafts",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "consent_acknowledged": True,
            "timeframe_acknowledged": True,
        },
    )
    assert started.status_code == 200
    body = started.json()
    draft_id = body["draft_id"]
    assert body["items"][0]["candidate_value"] == 3
    assert body["items"][0]["confirmed_value"] is None
    assert all(item["candidate_value"] is None for item in body["items"][1:])
    incomplete = client.post(f"/v1/assessments/chat-assisted/drafts/{draft_id}/complete?user_id={user_id}")
    assert incomplete.status_code == 409

    for index in range(7):
        response = client.put(
            f"/v1/assessments/chat-assisted/drafts/{draft_id}/items/{index}",
            json={"user_id": user_id, "value": 1, "timeframe_acknowledged": True},
        )
        assert response.status_code == 200
    first = client.post(f"/v1/assessments/chat-assisted/drafts/{draft_id}/complete?user_id={user_id}")
    second = client.post(f"/v1/assessments/chat-assisted/drafts/{draft_id}/complete?user_id={user_id}")
    assert first.status_code == second.status_code == 200
    assert first.json()["score"] == second.json()["score"] == 7
    history = client.get(f"/v1/assessments?user_id={user_id}").json()
    assert len([item for item in history if item["source"] == "chat_assisted"]) == 1
    exported = client.get(f"/v1/me/export?user_id={user_id}")
    assert exported.status_code == 200
    assert exported.json()["chat_assessment_drafts"][0]["draft_id"] == draft_id
    token = client.post("/v1/me/confirm-intent", json={"user_id": user_id, "action": "clear_records"}).json()[
        "confirm_token"
    ]
    cleared = client.delete(f"/v1/me/records?user_id={user_id}&confirm_token={token}")
    assert cleared.status_code == 200
    assert cleared.json()["deleted"]["chat_assessment_drafts"] == 1
    assert client.get(f"/v1/assessments/chat-assisted/drafts/{draft_id}?user_id={user_id}").status_code == 404


def test_chat_assisted_draft_cannot_use_another_users_conversation(monkeypatch) -> None:
    owner = f"owner-{uuid4().hex[:8]}"
    stranger = f"stranger-{uuid4().hex[:8]}"
    session_id = _conversation(owner, "过去两周经常焦虑。")
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment._extract_candidates",
        lambda messages, **kwargs: ({}, "ready"),
    )
    client = TestClient(app)
    denied = client.post(
        "/v1/assessments/chat-assisted/drafts",
        json={
            "user_id": stranger,
            "session_id": session_id,
            "consent_acknowledged": True,
            "timeframe_acknowledged": True,
        },
    )
    assert denied.status_code == 404
    started = client.post(
        "/v1/assessments/chat-assisted/drafts",
        json={"user_id": owner, "session_id": session_id, "consent_acknowledged": True, "timeframe_acknowledged": True},
    )
    draft_id = started.json()["draft_id"]
    assert client.get(f"/v1/assessments/chat-assisted/drafts/{draft_id}?user_id={stranger}").status_code == 404
    assert (
        client.put(
            f"/v1/assessments/chat-assisted/drafts/{draft_id}/items/0",
            json={"user_id": stranger, "value": 0, "timeframe_acknowledged": True},
        ).status_code
        == 404
    )


def test_chat_analysis_requires_explicit_consent(monkeypatch) -> None:
    user_id = f"consent-{uuid4().hex[:8]}"
    session_id = _conversation(user_id, "我最近有点担心。")
    called = []
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment._extract_candidates", lambda messages: called.append(1)
    )
    response = TestClient(app).post(
        "/v1/assessments/chat-assisted/drafts", json={"user_id": user_id, "session_id": session_id}
    )
    assert response.status_code == 403
    assert called == []


def test_model_candidates_without_owned_evidence_are_discarded(monkeypatch) -> None:
    from types import SimpleNamespace

    from psych_support_bot.infra.config.settings import get_settings

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    user_id = f"evidence-{uuid4().hex[:8]}"
    session_id = _conversation(user_id, "这两周每天有点紧张。")
    with SessionLocal() as db:
        message = db.query(Message).filter(Message.session_id == session_id).one()
        fake_model = SimpleNamespace(
            invoke=lambda _: SimpleNamespace(
                content='{"candidates":[{"item_index":0,"value":3,"message_id":99999999},'
                f'{{"item_index":1,"value":2,"message_id":{message.id}}}]}}'
            )
        )
        monkeypatch.setattr(
            "psych_support_bot.services.chat_assisted_assessment.build_chat_model", lambda **_: fake_model
        )
        candidates, status = _extract_candidates([message])
    assert status == "ready"
    assert candidates == {"1": {"value": 2, "message_id": message.id}}
    get_settings.cache_clear()


def test_guided_chat_collects_multiple_topics_then_reviews_once(monkeypatch) -> None:
    from types import SimpleNamespace

    from psych_support_bot.ai.schemas.messages import RiskResult

    user_id = f"guided-{uuid4().hex[:8]}"
    session_id = _conversation(user_id, "想聊聊这两周的状态。")

    def fake_extract(messages, *, guided=False, focus=None):
        if not guided:
            return {}, "ready"
        message_id = messages[0].id
        return {"0": {"value": 3, "message_id": message_id}, "1": {"value": 2, "message_id": message_id}}, "ready"

    monkeypatch.setattr("psych_support_bot.services.chat_assisted_assessment._extract_candidates", fake_extract)
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment.get_settings",
        lambda: SimpleNamespace(openai_api_key="test"),
    )
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment.classify_risk_llm",
        lambda *args: (
            RiskResult(risk_level="low", risk_types=[], needs_crisis_mode=False, reason="test"),
            None,
        ),
    )
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment._guided_reply", lambda message, focus: "先聊聊别的。"
    )
    client = TestClient(app)
    started = client.post(
        "/v1/assessments/chat-assisted/drafts",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "consent_acknowledged": True,
            "timeframe_acknowledged": True,
        },
    )
    draft_id = started.json()["draft_id"]
    turn = client.post(
        f"/v1/assessments/chat-assisted/drafts/{draft_id}/turns",
        json={"user_id": user_id, "message": "这两周几乎每天都很紧张，也有一半以上的日子停不下担心。"},
    )
    assert turn.status_code == 200
    assert turn.json()["turn_status"] == "guided"
    assert turn.json()["draft"]["items"][0]["candidate_value"] == 3
    assert turn.json()["draft"]["items"][1]["candidate_value"] == 2
    assert turn.json()["draft"]["next_topic_index"] == 2
    assert turn.json()["draft"]["items"][0]["confirmed_value"] is None
    assert client.get(f"/v1/assessments?user_id={user_id}").json() == []
    reviewed = client.post(
        f"/v1/assessments/chat-assisted/drafts/{draft_id}/review",
        json={
            "user_id": user_id,
            "answers": [3, 2, 0, 0, 0, 0, 0],
            "timeframe_acknowledged": True,
            "review_acknowledged": True,
        },
    )
    assert reviewed.status_code == 200
    assert reviewed.json()["score"] == 5
    repeated = client.post(
        f"/v1/assessments/chat-assisted/drafts/{draft_id}/review",
        json={"user_id": user_id, "answers": [0] * 7, "timeframe_acknowledged": True, "review_acknowledged": True},
    )
    assert repeated.json()["score"] == 5
    assert len(client.get(f"/v1/assessments?user_id={user_id}").json()) == 1


def test_guided_chat_pauses_on_crisis_and_does_not_extract(monkeypatch) -> None:
    user_id = f"risk-{uuid4().hex[:8]}"
    session_id = _conversation(user_id, "想聊聊这两周的状态。")
    calls = []

    def fake_extract(messages, *, guided=False, focus=None):
        if guided:
            calls.append(1)
        return {}, "ready"

    monkeypatch.setattr("psych_support_bot.services.chat_assisted_assessment._extract_candidates", fake_extract)
    client = TestClient(app)
    started = client.post(
        "/v1/assessments/chat-assisted/drafts",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "consent_acknowledged": True,
            "timeframe_acknowledged": True,
        },
    )
    draft_id = started.json()["draft_id"]
    turn = client.post(
        f"/v1/assessments/chat-assisted/drafts/{draft_id}/turns",
        json={"user_id": user_id, "message": "我想自杀。"},
    )
    assert turn.status_code == 200
    assert turn.json()["turn_status"] == "crisis"
    assert calls == []
    assert turn.json()["draft"]["status"] == "crisis_paused"
    assert turn.json()["draft"]["next_topic_index"] is None
    blocked = client.post(
        f"/v1/assessments/chat-assisted/drafts/{draft_id}/review",
        json={"user_id": user_id, "answers": [0] * 7, "timeframe_acknowledged": True, "review_acknowledged": True},
    )
    assert blocked.status_code == 409


def test_ambiguous_answer_moves_on_after_two_gentle_prompts(monkeypatch) -> None:
    from types import SimpleNamespace

    from psych_support_bot.ai.schemas.messages import RiskResult

    user_id = f"ambiguous-{uuid4().hex[:8]}"
    session_id = _conversation(user_id, "想聊聊这两周的状态。")
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment._extract_candidates",
        lambda messages, **kwargs: ({}, "ready"),
    )
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment.get_settings",
        lambda: SimpleNamespace(openai_api_key="test"),
    )
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment.classify_risk_llm",
        lambda *args: (
            RiskResult(risk_level="low", risk_types=[], needs_crisis_mode=False, reason="test"),
            None,
        ),
    )
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment._guided_reply", lambda message, focus: "我们接着聊聊。"
    )
    client = TestClient(app)
    started = client.post(
        "/v1/assessments/chat-assisted/drafts",
        json={
            "user_id": user_id,
            "session_id": session_id,
            "consent_acknowledged": True,
            "timeframe_acknowledged": True,
        },
    )
    draft_id = started.json()["draft_id"]
    for _ in range(2):
        turn = client.post(
            f"/v1/assessments/chat-assisted/drafts/{draft_id}/turns",
            json={"user_id": user_id, "message": "也许吧。"},
        )
        assert turn.status_code == 200
    assert turn.json()["draft"]["items"][0]["candidate_value"] is None
    assert turn.json()["draft"]["next_topic_index"] == 1


def test_unsafe_generated_topic_reply_uses_fixed_safe_prompt(monkeypatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment.get_settings",
        lambda: SimpleNamespace(openai_api_key="test"),
    )
    monkeypatch.setattr(
        "psych_support_bot.services.chat_assisted_assessment.build_chat_model",
        lambda **kwargs: SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="你有焦虑症。你最近怎么样？")),
    )
    reply = _guided_reply("我最近紧张", 1)
    assert "焦虑症" not in reply
    assert "担心" in reply
