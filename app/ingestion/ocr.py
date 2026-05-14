import pytesseract

from .preprocess import preprocess_image


def extract_text_from_image(image):
    """
    Run OCR on a PIL image.
    """

    processed_image = preprocess_image(image)

    text = pytesseract.image_to_string(processed_image)

    return text