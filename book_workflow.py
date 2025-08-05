import os
import json
import uuid
import datetime
from pathlib import Path
from typing import TypedDict, Literal
import requests 
from langgraph.graph import StateGraph, END 
from sentence_transformers import SentenceTransformer, util 
import pyttsx3 
import speech_recognition as sr 
from playwright.sync_api import sync_playwright
import chromadb 
from chromadb.config import Settings 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from dotenv import load_dotenv
load_dotenv()

HF_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
HF_REVIEW_MODEL = "sshleifer/distilbart-cnn-12-6"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)
reward_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB client
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="book_versions")

# ---------------- STATE ----------------
class State(TypedDict):
    url: str
    scraped: str
    written: str
    reviewed: str
    reward: float
    version_id: str
    intent: str

# ---------------- NODES ----------------
def scrape_node(state: State) -> State:
    url = state["url"]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        content = page.inner_text("body")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = output_dir / f"screenshot_{timestamp}.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        browser.close()
        return {"scraped": content}

def writer_node(state: State) -> State:
    text = state["scraped"]
    payload = {
    "inputs": f"Summarize this: {text[:2000]}",
    "parameters": {
        "do_sample": True,
        "temperature": 1.5
    }
}
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL}",
        headers=HEADERS,
        json=payload
    )
    try:
        result = response.json()
        summary = result[0]["summary_text"] if isinstance(result, list) and "summary_text" in result[0] else "[NO_SUMMARY]"
    except Exception as e:
        print("Error calling HF:", e)
        summary = "[ERROR]"
    return {"written": summary}

def review_node(state: State) -> State:
    text = state.get("written", "").strip()

    if not text or "[ERROR]" in text or "[NO_SUMMARY]" in text:
        return {"reviewed": "[REVIEW_FAILED] Cannot review due to previous summarization error."}

    try:
        prompt = (
    "Review the following summary and improve it for clarity, grammar, and conciseness.\n\n"
    "Make it professional, readable, and retain the original meaning. Avoid repetition.\n\n"
    f"{text}"
        )
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_REVIEW_MODEL}",
            headers=HEADERS,
            json={"inputs": prompt},
        )
        result = response.json()
        if isinstance(result, list) and "summary_text" in result[0]:
            reviewed_text = result[0]["summary_text"]
        else:
            reviewed_text = "[REVIEW_FAILED] Unexpected response format."
    except Exception as e:
        print("Error calling HF for review:", e)
        reviewed_text = "[REVIEW_FAILED] HF API call failed."

    return {"reviewed": reviewed_text}


def reward_node(state: State) -> State:
    emb1 = reward_model.encode(state["scraped"], convert_to_tensor=True)
    emb2 = reward_model.encode(state["reviewed"], convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return {"reward": round(score, 4)}

def version_node(state: State) -> State:
    version_id = f"v_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.datetime.now().isoformat()
    filepath = output_dir / f"{version_id}.json"

    # Save to disk
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump({"text": state["reviewed"], "reward": state["reward"], "timestamp": timestamp}, f, indent=2)

    # Save to ChromaDB
    collection.add(
        documents=[state["reviewed"]],
        metadatas=[{"version_id": version_id, "reward": state["reward"], "timestamp": timestamp}],
        ids=[version_id]
    )
    return {"version_id": version_id}

def human_node(state: State) -> State:
    print("\n[MODEL OUTPUT] Model Output:\n", state.get("written", ""))
    print("\n[HUMAN EDIT OR REGENERATED] Current Output:\n", state.get("reviewed", ""))
    while True:
        feedback = input("Suggest edits or type 'stop' to finalize, 'improve' to regenerate: ").strip().lower()
        if feedback in ["stop", "improve"]:
            break
    return {"intent": feedback}

def voice_node(state: State) -> State:
    engine = pyttsx3.init()
    engine.say(f"Summary complete. Reward score is {state['reward']}")
    engine.runAndWait()
    return {}

# ---------------- LANGGRAPH ----------------
workflow = StateGraph(State)
workflow.set_entry_point("Scrape")
workflow.add_node("Scrape", scrape_node)
workflow.add_node("Write", writer_node)
workflow.add_node("Review", review_node)
workflow.add_node("Reward", reward_node)
workflow.add_node("Version", version_node)
workflow.add_node("HumanLoop", human_node)
workflow.add_node("Voice", voice_node)

workflow.add_edge("Scrape", "Write")
workflow.add_edge("Write", "Review")
workflow.add_edge("Review", "Reward")
workflow.add_edge("Reward", "Version")
workflow.add_edge("Version", "HumanLoop")

# Conditional looping based on intent
def route_based_on_feedback(state: State) -> str:
    return state.get("intent", "improve")

workflow.add_conditional_edges("HumanLoop", route_based_on_feedback, {
    "stop": "Voice",
    "improve": "Write"
})

workflow.add_edge("Voice", END)

graph = workflow.compile()

"""def inspect_chromadb():
    print("\n Inspecting ChromaDB Collection:")
    all_collections = client.list_collections()
    print("Available Collections:")
    for col in all_collections:
        print(f"- {col.name}")

    # View all documents in 'book_versions'
    print(f"\ Contents of collection 'book_versions':")
    try:
        docs = collection.get()
        for i, doc in enumerate(docs["documents"]):
            print(f"\n--- Document {i+1} ---")
            print(f"ID: {docs['ids'][i]}")
            print(f"Reward: {docs['metadatas'][i]['reward']}")
            print(f"Timestamp: {docs['metadatas'][i]['timestamp']}")
            print("Summary Preview:", doc[:200], "...\n")
    except Exception as e:
        print("Error retrieving documents:", e)"""


def export_chromadb_versions():
    all_items = collection.get(include=["documents", "metadatas"])
    if not all_items["ids"]:
        print("No data in ChromaDB collection.")
        return

    chroma_export_path = output_dir / "chromadb_book_versions.json"
    with open(chroma_export_path, "w", encoding="utf-8") as f:
        json.dump(all_items, f, indent=2)
    print(f"\n Exported ChromaDB versions to: {chroma_export_path}")


# ---------------- RUN ----------------
result = graph.invoke({
    "url": "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
    })

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
if "reviewed" in result:
    result_path = output_dir / f"final_{timestamp}.txt"
    with open(result_path, "w", encoding='utf-8') as f:
        f.write("Generated Summary:\n" + result["reviewed"] + "\n")
        f.write(json.dumps({"status": "Saved", "path": str(result_path)}, indent=2))
        print(f"\n Workflow complete. Output saved to: {result_path}")
else:
    print("\n Workflow completed. No reviewed content to save (possibly user chose to stop).")

#inspect_chromadb()    

export_chromadb_versions()
