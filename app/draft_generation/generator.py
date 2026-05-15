from groq import Groq

from .prompts import GROUNDING_PROMPT


class GroundedGenerator:

    def __init__(
        self,
        groq_api_key,
        model_name="llama-3.1-8b-instant"
    ):

        self.client = Groq(
            api_key=groq_api_key
        )

        self.model_name = model_name

    def build_context(self, retrieved_chunks):

        context = ""

        for i, chunk in enumerate(retrieved_chunks, start=1):

            context += f"""[Evidence {i}]Source Page: {chunk['metadata']['page']}{chunk['text']}"""

        return context

    def generate(
        self,
        retrieved_chunks,
        rules_block = ""
    ):
        

        context = self.build_context(
            retrieved_chunks
        )
       
        prompt = GROUNDING_PROMPT.format(
            rules_block = rules_block,
            context=context
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate grounded legal-style drafts."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=800
        )

        return response.choices[0].message.content