"""Opt-in GAD-7 drafts derived from a user's own conversation."""

import json
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from psych_support_bot.api.auth import request_user_id
from psych_support_bot.domain.assessments.schemas import AssessmentAnswerSet, AssessmentResult
from psych_support_bot.domain.assessments.service import build_assessment_result
from psych_support_bot.infra.db.session import get_db_session
from psych_support_bot.services.chat_assisted_assessment import (
    complete_draft,
    confirm_item,
    create_draft,
    draft_view,
    get_owned_draft,
    guided_turn,
)

router = APIRouter(prefix="/v1/assessments/chat-assisted", tags=["chat-assisted-assessments"])


class CreateDraftRequest(BaseModel):
    user_id: str = ""
    session_id: str = Field(..., min_length=1, max_length=64)
    consent_acknowledged: bool = False
    timeframe_acknowledged: bool = False


class ConfirmItemRequest(BaseModel):
    user_id: str = ""
    value: int = Field(..., ge=0, le=3)
    timeframe_acknowledged: bool = False


class GuidedTurnRequest(BaseModel):
    user_id: str = ""
    message: str = Field(..., min_length=1, max_length=2000)


class ReviewRequest(BaseModel):
    user_id: str = ""
    answers: list[int] = Field(..., min_length=7, max_length=7)
    timeframe_acknowledged: bool = False
    review_acknowledged: bool = False


@router.post("/drafts")
def start_chat_assisted_draft(
    payload: CreateDraftRequest, request: Request, session: Session = Depends(get_db_session)
) -> dict:
    user_id = request_user_id(request, payload.user_id)
    if not payload.consent_acknowledged or not payload.timeframe_acknowledged:
        raise HTTPException(
            status_code=403, detail="Chat analysis and past-two-weeks timeframe require explicit consent"
        )
    draft, extraction_status = create_draft(session, user_id, payload.session_id)
    return draft_view(session, draft, extraction_status)


@router.post("/drafts/{draft_id}/turns")
def add_guided_turn(
    draft_id: str,
    payload: GuidedTurnRequest,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict:
    user_id = request_user_id(request, payload.user_id)
    draft = get_owned_draft(session, user_id, draft_id)
    reply, turn_status = guided_turn(session, draft, payload.message)
    return {"reply": reply, "turn_status": turn_status, "draft": draft_view(session, draft)}


@router.post("/drafts/{draft_id}/review", response_model=AssessmentResult)
def review_chat_assisted_draft(
    draft_id: str,
    payload: ReviewRequest,
    request: Request,
    session: Session = Depends(get_db_session),
) -> AssessmentResult:
    user_id = request_user_id(request, payload.user_id)
    if not payload.timeframe_acknowledged or not payload.review_acknowledged:
        raise HTTPException(status_code=403, detail="Confirm the timeframe and reviewed answers")
    draft = get_owned_draft(session, user_id, draft_id)
    if draft.status == "crisis_paused":
        raise HTTPException(status_code=409, detail="Assessment paused after a safety event")
    if draft.status != "completed":
        if any(type(value) is not int or value < 0 or value > 3 for value in payload.answers):
            raise HTTPException(status_code=422, detail="Each answer must be an integer from 0 to 3")
        now = datetime.now(UTC).isoformat()
        draft.confirmed_json = json.dumps(
            {str(index): {"value": value, "confirmed_at": now} for index, value in enumerate(payload.answers)}
        )
        complete_draft(session, draft)
    confirmed = json.loads(draft.confirmed_json)
    answers = [confirmed[str(index)]["value"] for index in range(7)]
    return build_assessment_result("gad7", answers=AssessmentAnswerSet(answers=answers), language="zh")


@router.get("/drafts/{draft_id}")
def read_chat_assisted_draft(
    draft_id: str, request: Request, user_id: str = "", session: Session = Depends(get_db_session)
) -> dict:
    user_id = request_user_id(request, user_id)
    return draft_view(session, get_owned_draft(session, user_id, draft_id))


@router.put("/drafts/{draft_id}/items/{item_index}")
def confirm_chat_assisted_item(
    draft_id: str,
    item_index: int,
    payload: ConfirmItemRequest,
    request: Request,
    session: Session = Depends(get_db_session),
) -> dict:
    user_id = request_user_id(request, payload.user_id)
    if not payload.timeframe_acknowledged:
        raise HTTPException(status_code=403, detail="Confirm the past-two-weeks timeframe")
    draft = get_owned_draft(session, user_id, draft_id)
    confirm_item(session, draft, item_index, payload.value)
    return draft_view(session, draft)


@router.post("/drafts/{draft_id}/complete", response_model=AssessmentResult)
def complete_chat_assisted_draft(
    draft_id: str, request: Request, user_id: str = "", session: Session = Depends(get_db_session)
) -> AssessmentResult:
    user_id = request_user_id(request, user_id)
    draft = get_owned_draft(session, user_id, draft_id)
    complete_draft(session, draft)
    confirmed = json.loads(draft.confirmed_json)
    answers = [confirmed[str(index)]["value"] for index in range(7)]
    return build_assessment_result("gad7", answers=AssessmentAnswerSet(answers=answers), language="zh")
