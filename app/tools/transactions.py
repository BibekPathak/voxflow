from __future__ import annotations

from app.tools.data import default_store
from app.tools.registry import tool


@tool(
    arg_descriptions={
        "customer_id": "The customer identifier",
        "limit": "Maximum number of transactions to return",
    }
)
async def get_recent_transactions(customer_id: str, limit: int = 5) -> dict:
    """List the most recent payments/transactions for a customer."""
    payments = default_store().payments_for(customer_id)
    payments = sorted(payments, key=lambda p: p.created_at, reverse=True)[:limit]
    return {
        "customer_id": customer_id,
        "count": len(payments),
        "payments": [
            {
                "payment_id": p.payment_id,
                "amount": p.amount,
                "currency": p.currency,
                "status": p.status,
                "created_at": p.created_at,
            }
            for p in payments
        ],
    }


@tool(
    arg_descriptions={
        "payment_id": "The payment/transaction identifier to inspect",
    }
)
async def inspect_payment(payment_id: str) -> dict:
    """Look up details and status of a single payment."""
    payment = default_store().get_payment(payment_id)
    if payment is None:
        return {"found": False, "payment_id": payment_id}
    return {
        "found": True,
        "payment_id": payment.payment_id,
        "customer_id": payment.customer_id,
        "amount": payment.amount,
        "currency": payment.currency,
        "status": payment.status,
        "decline_reason": payment.decline_reason,
        "created_at": payment.created_at,
    }
