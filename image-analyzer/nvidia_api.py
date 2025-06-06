import requests
import base64
import io
import os
from PIL import Image
from dotenv import load_dotenv
import datetime

load_dotenv()  # Load environment variables from .env

API_KEY = os.getenv("NVIDIA_API_KEY")  # Read the key

INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.2-11b-vision-instruct"

def analyze_image_with_nvidia(image: Image.Image, prompt: str) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]
            }
        ],
        "max_tokens": 512,
        "temperature": 0.8,
        "top_p": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stream": False
    }

    response = requests.post(INVOKE_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        return f"Error: {response.status_code}\n{response.text}"

def log_analysis(prompt: str, result: str):
    with open("analysis_log.txt", "a", encoding="utf-8") as log_file:
        timestamp = datetime.datetime.now().isoformat()
        log_file.write(f"\n---\nTime: {timestamp}\nPrompt: {prompt}\nResult: {result}\n")