from __future__ import annotations

from app.tools.data import default_store
from app.tools.registry import tool


@tool(
    arg_descriptions={
        "email": "Full or partial email address of the customer",
        "name": "Full or partial name of the customer",
    }
)
async def search_customer(email: str | None = None, name: str | None = None) -> dict:
    """Find a customer by email or name."""
    customer = default_store().find_customer(email=email, name=name)
    if customer is None:
        return {"found": False, "query": {"email": email, "name": name}}
    return {
        "found": True,
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "plan": customer.plan,
    }
