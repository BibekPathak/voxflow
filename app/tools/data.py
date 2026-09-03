from __future__ import annotations

import itertools
from dataclasses import dataclass


@dataclass(slots=True)
class Customer:
    customer_id: str
    name: str
    email: str
    plan: str


@dataclass(slots=True)
class Payment:
    payment_id: str
    customer_id: str
    amount: float
    currency: str
    status: str
    decline_reason: str | None
    created_at: str


@dataclass(slots=True)
class Ticket:
    ticket_id: str
    customer_id: str | None
    subject: str
    description: str
    status: str = "open"
    priority: str = "normal"


class SupportStore:
    """Deterministic in-memory fake of the support backend.

    Keeps tool behavior consistent and repeatable across sessions so the demo
    and the evaluation harness have stable data to operate on.
    """

    def __init__(self) -> None:
        self.customers: dict[str, Customer] = {}
        self.payments: dict[str, Payment] = {}
        self.tickets: dict[str, Ticket] = {}
        self._ticket_seq = itertools.count(1000)
        self._seed()

    def _seed(self) -> None:
        self.customers["cust_1"] = Customer("cust_1", "Alice Morgan", "alice@example.com", "premium")
        self.customers["cust_2"] = Customer("cust_2", "Bob Chen", "bob@example.com", "standard")
        self.payments["pay_101"] = Payment(
            "pay_101", "cust_1", 49.99, "USD", "declined", "insufficient_funds", "2026-08-30T10:12:00Z"
        )
        self.payments["pay_102"] = Payment(
            "pay_102", "cust_1", 129.00, "USD", "succeeded", None, "2026-08-28T14:03:00Z"
        )
        self.payments["pay_103"] = Payment(
            "pay_103", "cust_2", 9.99, "USD", "declined", "card_expired", "2026-08-31T09:45:00Z"
        )
        self.payments["pay_104"] = Payment("pay_104", "cust_2", 59.90, "USD", "succeeded", None, "2026-08-25T16:30:00Z")

    def find_customer(self, *, email: str | None = None, name: str | None = None) -> Customer | None:
        query_email = (email or "").strip().lower()
        query_name = (name or "").strip().lower()
        for customer in self.customers.values():
            if query_email and query_email in customer.email.lower():
                return customer
            if query_name and query_name in customer.name.lower():
                return customer
        return None

    def payments_for(self, customer_id: str) -> list[Payment]:
        return [p for p in self.payments.values() if p.customer_id == customer_id]

    def get_payment(self, payment_id: str) -> Payment | None:
        return self.payments.get(payment_id)

    def create_ticket(
        self, *, customer_id: str | None, subject: str, description: str, priority: str = "normal"
    ) -> Ticket:
        ticket = Ticket(
            ticket_id=f"tkt_{next(self._ticket_seq)}",
            customer_id=customer_id,
            subject=subject,
            description=description,
            priority=priority,
        )
        self.tickets[ticket.ticket_id] = ticket
        return ticket


_default_store: SupportStore | None = None


def default_store() -> SupportStore:
    global _default_store
    if _default_store is None:
        _default_store = SupportStore()
    return _default_store
