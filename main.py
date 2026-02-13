"""
MNEMOSYNE — main.py
Connects the Mnemosyne memory engine to Google Gemini.
Run: python main.py
"""

import os
import json
import logging
import logging.handlers
from dotenv import load_dotenv
import google.generativeai as genai
from mnemosyne import MnemosyneEngine

# Logging Setup

_log_handler = logging.handlers.RotatingFileHandler(
    filename="mnemosyne_logs.jsonl",
    maxBytes=2 * 1024 * 1024,   # 2 MB
    backupCount=3,
    mode="a",
    encoding="utf-8",
)
_log_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("mnemosyne")
logger.setLevel(logging.INFO)
logger.addHandler(_log_handler)


def log_turn(result):
    """Writes one JSONL line per turn with full memory audit trail."""
    entry = {
        "turn":               result.turn_number,
        "user_input":         result.user_message,
        "ai_response":        result.response,
        "memories_retrieved": [m.to_dict() for m in result.memories_used],
        "memories_added":     [m.to_dict() for m in result.memories_added],
        "memories_evicted":   result.evicted,
        "latency_ms":         result.latency_ms,
    }
    logger.info(json.dumps(entry))

load_dotenv()

# Configuration

API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY environment variable is not set.\n"
        "Run:  export GEMINI_API_KEY='your_key_here'"
    )


MODEL_ID = "gemini-2.5-flash-preview-04-17"

genai.configure(api_key=API_KEY)


# ─── Gemini Wrapper ───────────────────────────────────────────────────────────

def gemini_wrapper(system_prompt: str, user_message: str, history: list) -> str:
    """
    Bridges the Mnemosyne Engine with Google's Gemini API.

    Args:
        system_prompt: Dynamic prompt from Mnemosyne — contains the
                       <memory_context> block. Changes every turn, so
                       GenerativeModel must be created per-call.
        user_message:  Current message from the user.
        history:       List of previous turns as dicts:
                       [{'role': 'user'|'assistant', 'content': '...'}, ...]
                       The engine appends the CURRENT turn to history AFTER
                       calling this function, so history never includes the
                       current user_message. This is the correct shape for
                       Gemini's start_chat(history=...) + send_message().
    """

    # Must be created per-call: system_instruction contains the dynamic
    # memory block which is different on every turn.
    model = genai.GenerativeModel(
        model_name=MODEL_ID,
        system_instruction=system_prompt,
    )

    # Convert Mnemosyne history format → Gemini format.
    # Mnemosyne: role = "user" | "assistant"
    # Gemini:    role = "user" | "model"
    #
    # Gemini requires history to strictly alternate user/model and end
    # on a "model" turn. The engine guarantees this: history always
    # contains complete pairs (user + assistant) because appending happens
    # after this function returns. Empty history on turn 1 is also valid.
    gemini_history = []
    for turn in history:
        # FIX WARN-4: skip error responses so they do not pollute history
        if turn["content"].startswith("[Gemini API Error"):
            continue
        role = "model" if turn["role"] == "assistant" else "user"
        gemini_history.append({
            "role":  role,
            "parts": [turn["content"]],
        })

    # Ensure history strictly alternates and ends on model turn.
    # If an error was filtered and left history in a bad shape, trim the tail.
    while gemini_history and gemini_history[-1]["role"] != "model":
        gemini_history.pop()

    chat = model.start_chat(history=gemini_history)

    try:
        response = chat.send_message(user_message)
        return response.text

    except Exception as e:
        # FIX WARN-4: log at ERROR level so it stands out in the log file.
        # Return a clearly prefixed string so the engine and caller can detect
        # it is an error, not a real response.
        error_msg = f"[Gemini API Error: {str(e)}]"
        logger.error(json.dumps({"error": str(e), "type": type(e).__name__}))
        return error_msg


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    engine = MnemosyneEngine(
        db_path="mnemosyne_memories.db",
        llm_fn=gemini_wrapper,
        verbose=False,
    )

    print(f"Mnemosyne × {MODEL_ID} — ready")
    print("Logging to 'mnemosyne_logs.jsonl'")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            # FIX WARN-2: skip empty input — avoids ChromaDB query with empty string
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                print("Goodbye.")
                break

            result = engine.chat(user_input)
            print(f"Gemini: {result.response}\n")
            log_turn(result)

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()