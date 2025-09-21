import requests
import json

API = "http://localhost:8000/ask"
QUESTIONS_PATH = "../questions.json"   # your 8 queries

def run_and_collect(mode):
    # 'mode' is now just a label for this function, not sent to the API
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        qs = json.load(f)
    rows = []
    for item in qs:
        q = item["q"] if isinstance(item, dict) else item
        
        # <-- THIS IS THE FIX
        # The payload now matches the 'AskRequest' Pydantic model in your API.
        payload = {"query": q, "top_k": 3}
        
        try:
            r = requests.post(API, json=payload, timeout=30)
            r.raise_for_status()
            resp = r.json()
            # We use the local 'mode' variable for organizing results
            rows.append({
                "q": q,
                "mode": mode,
                "answer": resp.get("results"), # The API returns a 'results' key
                "abstained": resp.get("abstained", False),
                "top_context": (resp.get("contexts") or [None])[0]
            })
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed for q='{q}' mode='{mode}': {e}")
            rows.append({"q": q, "mode": mode, "answer": None, "abstained": False, "top_context": None})
    return rows

def print_table(rows_b, rows_h):
    # This function may need adjustment depending on what 'answer' contains
    print("| Q | Baseline | Rerank | Notes |")
    print("|---|---|---|---|")
    for i in range(len(rows_b)):
        # Assuming the 'answer' is a list of result dictionaries
        qb_top_result = (rows_b[i]["answer"] or [{}])[0].get("text", "None")
        qh_top_result = (rows_h[i]["answer"] or [{}])[0].get("text", "None")
        
        # Truncate for better display
        qb_display = (qb_top_result[:50] + '...') if len(qb_top_result) > 50 else qb_top_result
        qh_display = (qh_top_result[:50] + '...') if len(qh_top_result) > 50 else qh_top_result

        print(f"| {i+1} | {qb_display} | {qh_display} |  |")

if __name__ == "__main__":
    print("Make sure your API is running at", API)
    # The 'mode' argument here is just for local tracking
    bs = run_and_collect("baseline")
    hs = run_and_collect("rerank")
    print_table(bs, hs)