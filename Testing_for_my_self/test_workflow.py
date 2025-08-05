from playwright.sync_api import sync_playwright # type: ignore
import os
import requests # type: ignore

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto("https://en.wikipedia.org/wiki/Main_Page")
    content = page.inner_text("body")
    screenshot_path =  f"Testing_for_my_self\screenshot.png"
    page.screenshot(path=str(screenshot_path), full_page=True)
    browser.close()

print("Content = ", content)

from dotenv import load_dotenv # type: ignore
load_dotenv()

HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = "sshleifer/distilbart-cnn-12-6"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

text = content
payload = {"inputs": f"Summarize this: {text[:2000]}"}
response = requests.post(
    f"https://api-inference.huggingface.co/models/{HF_MODEL}",
    headers=HEADERS,
    json=payload
)

print("\nPayload = ", payload)

try:
    result = response.json()
    summary = result[0]["summary_text"] if isinstance(result, list) and "summary_text" in result[0] else "[NO_SUMMARY]"
except Exception as e:
    print("Error calling HF:", e)
    summary = "[ERROR]"

print("\nSummary =", summary)   