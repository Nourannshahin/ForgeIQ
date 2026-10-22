import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your project's .env file."
    )

client = genai.Client(api_key=api_key)

print("Connecting to Gemini...")
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="In one sentence, explain what preventive maintenance means."
)

print("\nGemini responded successfully!")
print(response.text)