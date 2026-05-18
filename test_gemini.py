import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is required")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    contents="테스트: 안녕하세요! 감정단어|수치 형식으로 아무거나 응답해주세요.",
)
print("API 응답 성공!")
print(response.text)
