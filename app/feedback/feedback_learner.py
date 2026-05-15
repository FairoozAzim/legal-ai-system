from groq import Groq


EXTRACT_RULE = """
You are an expert legal writing analyst.

Study how an operator corrected an AI draft
and return ONE concise reusable drafting instruction.

Rules:
- 1-3 sentences maximum
- imperative style
- concise
- no explanation
- no preamble
"""

EXTRACT_USER = """
Original draft:
{original}

Edited version:
{edited}

What single reusable drafting rule does this edit teach?
"""


class FeedbackLearner:

    def __init__(
        self,
        api_key,
        feedback_store,
        model_name="llama-3.1-8b-instant"
    ):

        self.client = Groq(
            api_key=api_key
        )

        self.feedback_store = feedback_store

        self.model_name = model_name

    # -----------------------------------
    # Extract one rule
    # -----------------------------------

    def extract_rule(
        self,
        edit
    ):

        response = (
            self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": EXTRACT_RULE
                    },
                    {
                        "role": "user",
                        "content": EXTRACT_USER.format(
                            original=edit[
                                "original_draft"
                            ],
                            edited=edit[
                                "edited_draft"
                            ]
                        )
                    }
                ],
                temperature=0.2,
                max_tokens=200
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    # -----------------------------------
    # Process pending edits
    # -----------------------------------

    def process_pending_edits(self):

        edits = (
            self.feedback_store.load_edits()
        )

        rules = (
            self.feedback_store.load_rules()
        )

        new_rules = []

        for edit in edits:

            if edit["rule_extracted"]:

                continue

            print(
                f"Extracting rule from "
                f"edit {edit['id']} ..."
            )

            rule_text = (
                self.extract_rule(edit)
            )

            rule = {
                "rule": rule_text,
                "source_edit_id": edit["id"],
                "source_name": edit["pdf_name"],
            }

            rules.append(rule)

            new_rules.append(rule)

            edit["rule_extracted"] = True

            print(f"Rule: {rule_text}")

        self.feedback_store.save_json(
            self.feedback_store.edits_file,
            edits
        )

        self.feedback_store.save_rules(
            rules
        )

        return new_rules

    # -----------------------------------
    # Prompt injection block
    # -----------------------------------

    def build_rules_block(self):

        rules = (
            self.feedback_store.load_rules()
        )

        if not rules:

            return ""

        lines = [
            (
                "Learned operator "
                "preferences — apply all "
                "of these:\n"
            )
        ]

        for i, rule in enumerate(
            rules[-10:][::-1],
            start=1
        ):

            lines.append(
                f"{i}. {rule['rule']}"
            )

        return "\n".join(lines)