from __future__ import annotations

from app.tools.data import default_store
from app.tools.registry import tool


@tool(
    arg_descriptions={
        "customer_id": "Customer identifier, if known",
        "subject": "Short summary of the issue",
        "description": "Detailed description of the issue",
        "priority": "low | normal | high",
    }
)
async def create_support_ticket(
    subject: str,
    description: str,
    customer_id: str | None = None,
    priority: str = "normal",
) -> dict:
    """Open a support ticket for a customer issue."""
    if priority not in ("low", "normal", "high"):
        raise ValueError(f"invalid priority {priority!r}")
    ticket = default_store().create_ticket(
        customer_id=customer_id, subject=subject, description=description, priority=priority
    )
    return {
        "ticket_id": ticket.ticket_id,
        "status": ticket.status,
        "priority": ticket.priority,
        "subject": ticket.subject,
    }
