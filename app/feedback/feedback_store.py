import json
import uuid

from datetime import datetime
from pathlib import Path


class FeedbackStore:

    def __init__(
        self,
        storage_dir="data/feedback"
    ):

        self.storage_dir = Path(storage_dir)

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.edits_file = (
            self.storage_dir / "edit_records.json"
        )

        self.rules_file = (
            self.storage_dir / "learned_rules.json"
        )

    # -----------------------------------
    # Generic JSON helpers
    # -----------------------------------

    def load_json(self, path):

        if not path.exists():

            return []

        return json.loads(
            path.read_text()
        )

    def save_json(
        self,
        path,
        data
    ):

        path.write_text(
            json.dumps(data, indent=2)
        )

    # -----------------------------------
    # Edit storage
    # -----------------------------------

    def capture_edit(
        self,
        original,
        edited,
        pdf_name,
        query
    ):

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "pdf_name": pdf_name,
            "query": query,
            "original_draft": original,
            "edited_draft": edited,
            "rule_extracted": False,
        }

        edits = self.load_edits()

        edits.append(record)

        self.save_json(
            self.edits_file,
            edits
        )

        print(
            f"Captured edit: {record['id']}"
        )

        return record

    # -----------------------------------
    # Loaders
    # -----------------------------------

    def load_edits(self):

        return self.load_json(
            self.edits_file
        )

    def load_rules(self):

        return self.load_json(
            self.rules_file
        )

    # -----------------------------------
    # Save rules
    # -----------------------------------

    def save_rules(
        self,
        rules
    ):

        self.save_json(
            self.rules_file,
            rules
        )