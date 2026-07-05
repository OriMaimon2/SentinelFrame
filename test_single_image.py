#!/usr/bin/env python3
"""
Test script: Generate a single image for prompt adjustment
Supports OpenAI and Google Gemini APIs
"""

import urllib.request
import base64
from pathlib import Path
from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
)

# ============================================================================
# CONFIG - CHOOSE YOUR API
# ============================================================================

MODEL_PROVIDER = "google"  # Options: "openai" or "google"

# Common Config (API keys imported from config.py)
IMAGE_SIZE = "1024x1024"
OUTPUT_DIR = "gemini_output"

# ============================================================================
# EDIT THIS PROMPT
# ============================================================================

LABEL = "right_hand_in_bag"
"""PROMPT = (
    "Person holding open shopping bag, right hand inside bag reaching, "
    "supermarket CCTV camera view, good resolution, "
    "wide-angle overhead camera, supermarket shelves visible"
)"""

PROMPT = (

    "A realistic wide shot frame from a high ceiling-mounted supermarket security camera, looking down at a natural steep angle over a long shopping aisle. "
    "An ordinary customer is captured from a realistic, natural distance down the aisle, browsing the shelves. "
    "The person is small to medium in the frame, ensuring a true surveillance perspective. "
    "Their right hand is clearly visible and exposed outside of their body outline as they stand naturally. "
    "Bright and authentic supermarket environment with fully stocked shelves, colorful products, and clear overhead fluorescent lighting. "
    "Completely unposed and candid composition, captured from a normal high vantage point, looking like genuine, unedited store footage."
)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_output_dir() -> None:
    """Create output directory."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output directory ready: {OUTPUT_DIR}/")


def generate_with_openai(prompt: str) -> str | None:
    """Generate image using OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            response = client.images.generate(
                model=OPENAI_MODEL,
                prompt=prompt,
                size=IMAGE_SIZE,
                n=1,
            )
            return response.data[0].url
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠ Attempt {attempt + 1} failed: {str(e)[:80]}... Retrying...")
            else:
                print(f"  ✗ Failed after {MAX_RETRIES} attempts: {str(e)}")
                return None


def generate_with_google(prompt: str) -> bytes | None:
    """Generate image using Google Gemini API."""
    try:
        from google import genai
    except ImportError:
        print("❌ Google genai library not installed. Install with:")
        print("   pip install google-genai")
        return None

    client = genai.Client(api_key=GOOGLE_API_KEY)

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            interaction = client.interactions.create(
                model=GOOGLE_MODEL,
                input=prompt,
            )

            if interaction.output_image and interaction.output_image.data:
                return base64.b64decode(interaction.output_image.data)
            return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠ Attempt {attempt + 1} failed: {str(e)[:80]}... Retrying...")
            else:
                print(f"  ✗ Failed after {MAX_RETRIES} attempts: {str(e)}")
                return None


def save_image(data, filepath: Path) -> bool:
    """Save image data (bytes, URL, or PIL Image)."""
    try:
        if isinstance(data, bytes):
            # Write raw bytes
            with open(filepath, "wb") as f:
                f.write(data)
        elif hasattr(data, 'save'):
            # PIL Image
            data.save(str(filepath))
        elif isinstance(data, str):
            # URL
            urllib.request.urlretrieve(data, str(filepath))
        return True
    except Exception as e:
        print(f"  ✗ Save failed: {str(e)[:80]}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def test_generate() -> None:
    """Generate single test image."""
    setup_output_dir()

    print(f"\n🎯 Label: {LABEL}")
    print(f"📝 Prompt:\n   {PROMPT}\n")

    if MODEL_PROVIDER == "openai":
        print(f"⏳ Generating with OpenAI ({OPENAI_MODEL}, {IMAGE_SIZE})...")
        data = generate_with_openai(PROMPT)
    elif MODEL_PROVIDER == "google":
        print(f"⏳ Generating with Google ({GOOGLE_MODEL}, {IMAGE_SIZE})...")
        data = generate_with_google(PROMPT)
    else:
        print(f"❌ Unknown provider: {MODEL_PROVIDER}")
        return

    if data is None:
        print("❌ Generation failed")
        return

    filepath = Path(OUTPUT_DIR) / f"{LABEL}_{MODEL_PROVIDER}_test.png"
    if save_image(data, filepath):
        print(f"✅ Success! Saved to: {filepath}")
    else:
        print("❌ Save failed")


if __name__ == "__main__":
    if MODEL_PROVIDER == "google" and GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("❌ ERROR: Set your Google API key")
        print("   GOOGLE_API_KEY = 'your-key-here'")
    elif MODEL_PROVIDER == "openai" and OPENAI_API_KEY == "PUT_YOUR_KEY_HERE":
        print("❌ ERROR: Set your OpenAI API key")
        print("   OPENAI_API_KEY = 'sk-...'")
    else:
        test_generate()
