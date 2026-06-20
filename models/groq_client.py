from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_APIKEY")
)

def generate(
    prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.3
):
    """
    Generate text using Groq models.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature
    )

    return response.choices[0].message.content