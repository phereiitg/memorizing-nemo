"""
MNEMOSYNE — Sentinel Agent (Layer 1)
UPGRADED: Uses Gemini 2.5 Flash for intelligent, semantic memory extraction.
"""

import json
import google.generativeai as genai
from typing import Optional
from .models import MemoryObject, MemoryType, ExtractionResult

# CONFIGURATION
CONFIDENCE_THRESHOLD = 0.60

SYSTEM_PROMPT = """
You are a Memory Extraction Sentinel. Your goal is to extract strictly factual information, user preferences, and constraints from the user's latest message.

Output a JSON object with a key "memories" containing a list of extracted items.
Each item must have:
- "type": One of ["fact", "preference", "constraint", "entity", "commitment"]
- "key": A short, specific snake_case key (e.g., "dietary_restriction", "vacation_preference")
- "value": The extracted fact as a concise string (e.g., "Allergic to peanuts", "Prefers mountains over beaches")
- "confidence": A float (0.0 to 1.0) indicating how explicit the user was.

Rules:
1. IGNORE casual conversation, greetings, or questions that don't reveal facts.
2. EXTRACT distinct facts even if they are in the same sentence.
3. INFER context: "I'm vegan" -> constraint: dietary_restriction = "Vegan".
4. HANDLE negation: "I don't like horror movies" -> preference: movie_dislike = "Horror".
5. BE CONCISE.

Example Input: "I usually prefer mountains over beaches, and make sure to call me after 5pm."
Example Output:
{
  "memories": [
    {"type": "preference", "key": "vacation_preference", "value": "Prefers mountains over beaches", "confidence": 0.9},
    {"type": "constraint", "key": "contact_time", "value": "Call after 5 PM", "confidence": 0.95}
  ]
}
"""

# SENTINEL AGENT

class Sentinel:
    """
    Layer 1: Extracts memory candidates using an LLM (Gemini Flash).
    Runs asynchronously in the background.
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.threshold = confidence_threshold
        # Initialize Gemini Model specifically for JSON output
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"},
            system_instruction=SYSTEM_PROMPT
        )

    def extract(self, user_message: str, turn_number: int) -> ExtractionResult:
        """
        Sends the user message to Gemini Flash to extract memory candidates.
        """
        candidates = []
        raw_text = ""

        try:
            # 1. Call Gemini
            response = self.model.generate_content(
                f"User Message: \"{user_message}\""
            )
            raw_text = response.text
            
            # 2. Parse JSON
            data = json.loads(raw_text)
            raw_memories = data.get("memories", [])

            # 3. Convert to internal MemoryObject format
            for item in raw_memories:
                try:
                    # Validate Type
                    mem_type_str = item.get("type", "fact").lower()
                    # Map common LLM mistakes to valid enums if necessary
                    if mem_type_str not in [t.value for t in MemoryType]:
                        mem_type_str = "fact"
                    
                    mem_type = MemoryType(mem_type_str)
                    
                    # Create Object
                    conf = float(item.get("confidence", 0.0))
                    mem = MemoryObject(
                        type=mem_type,
                        key=item.get("key", "extracted_info"),
                        value=item.get("value", ""),
                        source_turn=turn_number,
                        last_recalled_turn=turn_number,
                        heat=min(1.0, conf),
                        confidence=conf,
                    )
                    candidates.append(mem)
                except Exception as e:
                    # Skip malformed items
                    continue

        except Exception as e:
            # In production, log this error to your file
            print(f"[Sentinel Error] Extraction failed: {e}")

        # 4. Filter Results
        filtered_in  = [c for c in candidates if c.confidence >= self.threshold]
        filtered_out = [c for c in candidates if c.confidence < self.threshold]

        return ExtractionResult(
            candidates=candidates,
            raw_turn=user_message,
            turn_number=turn_number,
            filtered_in=filtered_in,
            filtered_out=filtered_out,
        )