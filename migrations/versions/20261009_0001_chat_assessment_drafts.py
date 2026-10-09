"""Persist chat-assisted assessment drafts separately from scored assessments."""

import sqlalchemy as sa
from alembic import op

revision = "20261009_0001"
down_revision = "20260904_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_assessment_drafts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("assessment_type", sa.String(32), nullable=False),
        sa.Column("candidates_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("confirmed_json", sa.Text(), nullable=False, server_default="{}"),
        sa.Column("status", sa.String(32), nullable=False, server_default="draft"),
        sa.Column("assessment_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chat_assessment_drafts_user_id", "chat_assessment_drafts", ["user_id"])
    op.create_index("ix_chat_assessment_drafts_session_id", "chat_assessment_drafts", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_chat_assessment_drafts_session_id", "chat_assessment_drafts")
    op.drop_index("ix_chat_assessment_drafts_user_id", "chat_assessment_drafts")
    op.drop_table("chat_assessment_drafts")
