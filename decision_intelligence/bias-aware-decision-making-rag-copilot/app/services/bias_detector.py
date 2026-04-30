"""
bias_detector.py

Enhanced version:
- Supports comparison between:
    1. Baseline LLM (no RAG)
    2. RAG-enhanced LLM (with context)

This allows qualitative evaluation of improvement.
"""
import json
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
# Load API key from environment
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def get_project_root():
    """
    Returns absolute path to project root directory.

    Why:
    - Ensures all file paths work regardless of where script is executed
    - Avoids relative path bugs (like the one you're seeing)
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def load_taxonomy():
    """
    Load bias taxonomy from project data folder.
    """
    base_dir = get_project_root()

    file_path = os.path.join(base_dir, "data", "metadata", "bias-taxonomy.json")

    with open(file_path, "r") as f:
        return json.load(f)


def load_prompt():
    """
    Load system prompt from prompts directory.
    """
    base_dir = get_project_root()

    file_path = os.path.join(base_dir, "prompts", "bias_detection_system_prompt.txt")

    with open(file_path, "r") as f:
        return f.read()


#    Temporary RAG simulation.

def load_mock_rag_context(scenario: str):
    if "similar" in scenario or "background" in scenario:
        return """
        Similarity bias occurs when decision-makers favor individuals who resemble themselves.
        This reduces diversity and leads to suboptimal hiring outcomes.
        """
    elif "leader" in scenario:
        return """
        Groupthink and authority bias occur when teams defer to leaders without critical evaluation.
        This suppresses dissent and leads to poor decision quality.
        """
    elif "model" in scenario or "women" in scenario:
        return """
        Historical bias and representation bias arise when models are trained on past data reflecting inequality.
        This leads to systematic discrimination.
        """
    else:
        return ""

import time

def call_llm(system_prompt: str, scenario: str):
    """
    LLM call using Google GenAI SDK with retry + fallback models
    Check the model from here for rate limits
    https://aistudio.google.com/rate-limit
    """

    models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flasg-tts",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview",
        "gemini-flash-latest"
    ]

    full_prompt = f"""
{system_prompt}

---

Scenario:
{scenario}

---

IMPORTANT:
- Output MUST be valid JSON
- Do NOT include markdown
- Follow the schema strictly
"""

    last_error = None

    for model_name in models:
        try:
            print(f"Trying model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt
            )

            return json.loads(response.text)

        except Exception as e:
            print(f"Model {model_name} failed: {e}")

            # Detect 503 / overload errors
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print("Retrying with next model...")
                time.sleep(2)
                last_error = e
                continue
            else:
                # Other errors should fail fast
                raise e

    raise Exception(f"All models failed. Last error: {last_error}")


def detect_bias_comparison(scenario: str):
    """
    Compare bias detection:
    1. Without RAG
    2. With RAG

    Returns:
        dict with both responses
    """

    taxonomy = load_taxonomy()
    base_prompt = load_prompt()

    # -----------------------------
    # WITHOUT RAG (Baseline)
    # -----------------------------
    system_prompt_no_rag = (
        base_prompt
        + "\n\nBias Taxonomy:\n"
        + json.dumps(taxonomy, indent=2)
    )

    response_no_rag = call_llm(system_prompt_no_rag, scenario)

    # -----------------------------
    # WITH RAG (Enhanced)
    # -----------------------------
    rag_context = load_mock_rag_context(scenario)

    system_prompt_with_rag = (
        base_prompt
        + "\n\nRelevant Context from Knowledge Base:\n"
        + rag_context
        + "\n\nBias Taxonomy:\n"
        + json.dumps(taxonomy, indent=2)
    )

    response_with_rag = call_llm(system_prompt_with_rag, scenario)

    return {
        "without_rag": response_no_rag,
        "with_rag": response_with_rag
    }

def compare_outputs(no_rag, with_rag):
    print("\n📊 COMPARISON INSIGHT")

    if no_rag == with_rag:
        print("⚠️ No difference detected")
    else:
        print("✅ RAG introduced differences")

    print("Bias count (no RAG):", len(no_rag.get("biases_identified", [])))
    print("Bias count (with RAG):", len(with_rag.get("biases_identified", [])))

def print_comparison(result: dict):
    """
    Pretty-print comparison output.

    Why:
    - Makes qualitative evaluation easy
    - Useful for demos and interviews
    """

    print("\n" + "=" * 80)
    print("🔴 WITHOUT RAG (Baseline)")
    print("=" * 80)
    print(result["without_rag"])

    print("\n" + "=" * 80)
    print("🟢 WITH RAG (Enhanced)")
    print("=" * 80)
    print(result["with_rag"])
