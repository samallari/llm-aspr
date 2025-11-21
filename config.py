import os
from dotenv import load_dotenv

load_dotenv()

OPENREVIEW_EMAIL = os.getenv("EMAIL")
OPENREVIEW_PASSWORD = os.getenv("PASSWORD")