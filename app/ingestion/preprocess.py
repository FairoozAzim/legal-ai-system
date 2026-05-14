from PIL import ImageOps, ImageFilter


def preprocess_image(image):
    """
    Apply preprocessing for better OCR quality.
    """

    image = ImageOps.grayscale(image)

    image = ImageOps.autocontrast(image)

    image = image.filter(ImageFilter.SHARPEN)

    return image