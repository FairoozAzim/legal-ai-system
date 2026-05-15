improved_prompt = """
You are generating a grounded legal-style summary.

{rules_block}

ONLY use the provided evidence.

If information is uncertain,
explicitly say uncertain.

Retrieved Evidence:

{context}

Generate:
1. Case summary
2. Key timeline
3. Important unresolved issues
4. Evidence references
"""