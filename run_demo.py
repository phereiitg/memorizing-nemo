import time
import os
from dotenv import load_dotenv
from mnemosyne.engine import MnemosyneEngine

# Load API Key
load_dotenv()

def run_pipeline_demo():
    print("🐠 --- Starting Memorizing-Nemo Pipeline Demo ---")
    
    # Initialize Engine with verbose=True to see the "thinking"
    engine = MnemosyneEngine(verbose=True)
    
    # The Scripted Conversation
    test_inputs = [
        "Hi, I'm Prakhar. I am a student at IIT Guwahati.",
        "I really prefer mountains over beaches. I find them peaceful.",
        "Where should I go for a weekend trip? Keep my location and preferences in mind."
    ]

    for i, text in enumerate(test_inputs):
        print(f"\n[Turn {i+1}] ------------------------------------------------")
        print(f"👤 User: {text}")
        
        # Run the full pipeline (Sentinel -> Oracle -> Gemini -> Curator)
        result = engine.chat(text)
        
        print(f"🤖 Nemo: {result.response}")
        
        # Small pause to let background threads (Curator) finish printing logs
        time.sleep(2)

    print("\n✅ Demo Complete. Memory pipeline verified.")

if __name__ == "__main__":
    run_pipeline_demo()