import os
import dotenv

dotenv.load_dotenv()

async def mask_sensitive_data(text):
    password = os.getenv('GTC_PASSWORD')
    if password:
        masked_password = '*' * len(password)
        return text.replace(password, masked_password)
    return text

