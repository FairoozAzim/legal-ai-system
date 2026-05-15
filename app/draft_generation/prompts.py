GROUNDING_PROMPT = """You are generating a grounded legal-style summary. 
{rules_block}

ONLY use the provided evidence. 
If information is uncertain explicitly say uncertain or mention that you don't know.

Retrieved Evidence: 

{context} 

Generate: 
1. Case summary 
2. Key timeline 
3. Important unresolved issues 
4. Evidence references

"""