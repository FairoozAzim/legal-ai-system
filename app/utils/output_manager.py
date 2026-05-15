import json
from pathlib import Path
from datetime import datetime


class OutputManager:

    def __init__(
        self,
        output_dir="data/outputs"
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_output(
        self,
        pdf_name,
        query,
        generated_draft,
        retrieved_chunks
    ):

        timestamp = datetime.utcnow().strftime(
            "%Y%m%d_%H%M%S"
        )

        output_data = {
            "timestamp": timestamp,
            "pdf_name": pdf_name,
            "query": query,
            "generated_draft": generated_draft,
            "retrieved_evidence": retrieved_chunks
        }

        file_name = (
            f"{Path(pdf_name).stem}_{timestamp}.json"
        )

        save_path = (
            self.output_dir / file_name
        )

        with open(save_path, "w") as f:

            json.dump(
                output_data,
                f,
                indent=2
            )

        return save_path