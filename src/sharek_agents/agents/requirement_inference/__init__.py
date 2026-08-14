"""Infers the skills and levels a Contribution Request demands.

Reads the Request only. It never sees contributor data, so it cannot form an
opinion about a person — it describes the work, and NestJS derives the
eligibility verdict from what it returns (DEC-078, ADR 0015).
"""
