import fitz
import io

from PIL import Image

from .ocr import extract_text_from_image


class OCRPDFLoader:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        self.doc = fitz.open(pdf_path)

    def extract_native_text(self, page):

        return page.get_text().strip()

    def extract_images_from_page(self, page):

        return page.get_images(full=True)

    def run_ocr_on_images(self, image_list):

        ocr_texts = []

        for img in image_list:

            xref = img[0]

            base_image = self.doc.extract_image(xref)

            image_bytes = base_image["image"]

            image = Image.open(io.BytesIO(image_bytes))

            text = extract_text_from_image(image)

            if text.strip():

                ocr_texts.append(text)

        return ocr_texts

    def load(self):

        documents = []

        for page_num, page in enumerate(self.doc, start=1):

            native_text = self.extract_native_text(page)

            image_list = self.extract_images_from_page(page)

            ocr_texts = self.run_ocr_on_images(image_list)

            final_text = native_text

            if ocr_texts:

                final_text += "\n\n[OCR IMAGE CONTENT]\n"

                final_text += "\n".join(ocr_texts)

            documents.append(
                {
                    "text": final_text,
                    "metadata": {
                        "page": page_num,
                        "source": self.pdf_path,
                        "num_images": len(image_list)
                    }
                }
            )

        return documents