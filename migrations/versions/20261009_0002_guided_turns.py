"""Track how often each GAD-7 topic was gently explored."""

import sqlalchemy as sa
from alembic import op

revision = "20261009_0002"
down_revision = "20261009_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_assessment_drafts", sa.Column("asked_json", sa.Text(), nullable=False, server_default="{}"))


def downgrade() -> None:
    op.drop_column("chat_assessment_drafts", "asked_json")
