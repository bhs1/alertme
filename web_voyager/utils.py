import os
import dotenv
from PIL import Image
import io
import base64

dotenv.load_dotenv()

async def mask_sensitive_data(text):
    password = os.getenv('GTC_PASSWORD')
    if password:
        masked_password = '*' * len(password)
        return text.replace(password, masked_password)
    return text

async def compress_screenshot(screenshot, scale_factor: int = 1) -> str:
    """Compress the screenshot by reducing its width and height by 4x."""
    image = Image.open(io.BytesIO(screenshot))

    new_size = (image.width // scale_factor, image.height // scale_factor)
    compressed_image = image.resize(new_size, Image.LANCZOS)

    # Convert back to base64
    buffer = io.BytesIO()
    # Save as PNG to avoid JPEG compression
    compressed_image.save(buffer, format="PNG")
    compressed_b64_image = base64.b64encode(buffer.getvalue()).decode()

    return compressed_b64_image

import math
def calculate_distance(bbox, x, y):
    center_x = bbox['x'] + bbox['width'] / 2
    center_y = bbox['y'] + bbox['height'] / 2
    print(f"center_x: {center_x}, center_y: {center_y}")
    return math.sqrt((x - center_x)**2 + (y - center_y)**2)

async def take_screenshot(page):
    screenshot = await page.screenshot()
    compressed_screenshot = await compress_screenshot(screenshot)
    return compressed_screenshot