import os
from app.feedback import (
    FeedbackStore,
    FeedbackLearner
)
from pathlib import Path
from dotenv import load_dotenv
from app.draft_generation import GroundedGenerator

load_dotenv()  
GROQ_API_KEY = os.getenv("GROQ_KEY")
# -----------------------------------
# Store feedback
# -----------------------------------
EDITS_FILE = Path("edit_records.json")
RULES_FILE = Path("learned_rules.json")


feedback_store = FeedbackStore()

sample_edit= """
Case Summary

This case involves a property ownership dispute between Amelia Harper (Plaintiff) and Westbrook Holdings LLC (Defendant). The dispute arises from alleged conflicting ownership information in a property transfer record dated March 12, 2022. The Plaintiff filed an ownership challenge on January 11, 2024, questioning the validity and consistency of the recorded ownership documentation.

Key Timeline

1. March 12, 2022: Property transfer allegedly recorded.
2. June 4, 2023: Inspection request submitted.
3. January 11, 2024: Plaintiff filed an ownership challenge.

Important Unresolved Issues

1. Conflicting Ownership Records:
   The transfer documentation contains allegedly conflicting ownership information, as asserted by the Plaintiff.

2. Boundary Uncertainty:
   The Inspection Memo (Evidence 2) notes that boundary markers are unclear, creating ambiguity regarding property limits.

3. Document Integrity Concerns:
   The ownership certificate appears incomplete, with a reported signature mismatch on the record copy, as referenced in the Scanned Note (Evidence 4). This raises questions regarding document authenticity.

4. Evidence Quality and Reliability Issues:
   Several evidentiary materials may be affected by quality limitations, including:
   - Presence of photocopies of photocopies (Evidence 6)
   - Low-confidence OCR transcriptions (Evidence 5)
   - Potential gaps or obscured sections in source documents
   These factors may impact the reliability and completeness of the record set.

Evidence References

1. Evidence 1: Transfer Record (Page 3)
   - "Ownership remains under review."

2. Evidence 2: Inspection Memo (Page 5)
   - "Boundary markers unclear."

3. Evidence 3: Witness Statement (Page 7)
   - "Prior owner disputed transaction."

4. Evidence 4: Scanned Note
   - "Ownership certificate appears incomplete. Signature mismatch on record copy. Manual verification recommended prior to filing recommendation."

5. Evidence 5: OCR Extracted Content
   - Content reliability is uncertain due to low-quality transcription and incomplete text recognition.

6. Evidence 6: Internal Review Notes
   - "Several pages appear to be photocopies of photocopies. OCR confidence varies significantly by section. Some evidence may be incomplete or partially obscured."
"""

def feedback_learning(original_draft, pdf_name,  retrieved_chunks, query = "Summarize the case",sample_edit=sample_edit):

    feedback_store.capture_edit(
        original=original_draft,
        edited= sample_edit,
        pdf_name=pdf_name,
        query= query
        )

    # -----------------------------------
    # Learn rules
    # -----------------------------------

    learner = FeedbackLearner(
        api_key=GROQ_API_KEY,
        feedback_store=feedback_store
    )

    learner.process_pending_edits()

    # -----------------------------------
    # Build prompt injection block
    # -----------------------------------

    rules_block = learner.build_rules_block()

    print(rules_block)


    # -----------------------------------
    # Generation
    # -----------------------------------


    generator = GroundedGenerator(
        groq_api_key= GROQ_API_KEY
    )

    improved_draft = generator.generate(
        retrieved_chunks,
        rules_block=rules_block
    )
    return improved_draft