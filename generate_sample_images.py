#!/usr/bin/env python3
"""
Generate 1 sample image per label for SentinelFrame project.
Uses Google Gemini API with structured prompt templates.
"""

import json
import base64
from pathlib import Path
from google import genai
from config import GOOGLE_API_KEY, GOOGLE_MODEL, OUTPUT_DIR

# ============================================================================
# CONFIG (imported from config.py)
# ============================================================================

# ============================================================================
# LABEL SPACE (FROM PROJECT)
# ============================================================================

LABELS = [
    "hand_in_pocket",
    "hand_in_bag",
    "hand_under_clothing",
    "object_in_hand",
    "interacting_with_shelf",
    "hand_occluded_generic",
    "both_hands_not_visible",
]

# ============================================================================
# PROMPT STRUCTURE
# ============================================================================

# ============================================================================
# PROMPT STRUCTURE (UPDATED FOR MAXIMUM NATURAL SURVEILLANCE REALISM)
# ============================================================================

PROMPT_TEMPLATE = {
    "base_context": (
        "A realistic wide shot frame from a standard ceiling-mounted supermarket security camera, "
        "looking down at a natural steep angle over a shopping aisle. "
        "Bright and authentic supermarket environment with fully stocked shelves, colorful products, "
        "and clear overhead fluorescent lighting. Crucially, the aisle is completely empty of any other people, "
        "containing only a solitary single ordinary customer in casual clothes, captured from "
        "a realistic, natural distance down the aisle, ensuring a true surveillance perspective with "
        "completely unposed, candid, and organic framing. Only one single isolated human is visible."
    ),
"prompts": {
        "hand_in_pocket": {
            # מניעת בלבול: מדגישים כיס של מכנסיים ולא של קפוצ'ון, ומגדירים שהאגודל או חלק מהיד נכנסים פנימה ברור.
            "action": "The customer is standing by the shelves with their right hand firmly pushed inside their pants pocket (jeans or trousers). The fabric of the pocket is visibly bulging from the hand inside.",
            "framing": "Full-body view from a natural distance, showing a relaxed, casual shopping posture where the arm goes straight down into the pocket.",
        },
        "hand_in_bag": {
            # מניעת בלבול: מדגישים שהיד ממש *בתוך* חלל התיק (ולא רק מחזיקה את הרצועה).
            "action": "The customer is holding an open shopping bag or backpack, with their right hand and wrist completely plunged inside the dark main compartment, actively rummaging or reaching deep into the bottom of the bag.",
            "framing": "The bag is held forward or to the side, clearly showing the hand disappearing past the opening of the bag from the high vantage point.",
        },
        "hand_under_clothing": {
            "action": "The customer has their right hand slipped directly underneath the front fabric of their jacket, tucked flat against their torso, hidden completely beneath the outerwear layer. There are no pockets involved.",
            "framing": "The arm disappears under the main zipper area of the jacket, visible from a realistic distance down the aisle.",
        },
        "object_in_hand": {
            # מניעת בלבול: מגדירים אחיזה ברורה (Grip) של מוצר קטן, כדי שהמודל לא יצייר סתם יד פתוחה ליד מוצר.
            "action": "The customer is holding a specific retail item (like a soda can, juice bottle, or snack box) with a clear, firm grip. Their fingers are wrapped tightly around the product, showcasing ownership of the item.",
            "framing": "The object is held away from the shelves, making it clear it has been taken, with the hand and object fully exposed from the ceiling angle.",
        },
        "interacting_with_shelf": {
            # מניעת בלבול: מפרידים בין "סתם עמידה ליד" לבין אינטראקציה פיזית (מגע, הזזה, שליפה).
            "action": "The customer is actively reaching out, their arm extended forward, with their fingers physically touching, sliding, or pulling a specific box from the middle shelf.",
            "framing": "The physical contact between the fingertips and the product packaging is the focal point, captured from a natural distance down the aisle.",
        },
        "hand_occluded_generic": {
            # מניעת בלבול: מסבירים ל-AI *למה* לא רואים את היד (בגלל זווית הגוף/המדף) כדי שלא יחביא אותה בכיס בטעות.
            "action": "The customer is turned sideways or has their back partially to the camera. Their right arm is positioned behind their own torso or blocked entirely by a large shelf column, creating a natural physical obstruction.",
            "framing": "A natural wide shot where the person is fully visible, but one hand is completely blocked from the camera's line of sight by an external object or their own body.",
        },
        "both_hands_not_visible": {
            # מניעת בלבול: מגדירים תנוחה ספציפית שמעלימה את שתי הידיים (כמו ידיים מאחורי הגב או שלובות עמוק), ללא שימוש בכיסים.
            "action": "The customer is browsing with both of their arms held flat behind their lower back, or with both arms tightly crossed over their chest with hands tucked completely out of sight. No pockets or bags are used.",
            "framing": "A clear surveillance view of the shopper from a distance, showing a complete full-body profile but with zero hands or fingers visible anywhere.",
        },
    },
}


def build_prompt(label: str) -> str:
    """Build complete prompt from template into a natural, seamless paragraph."""
    base = PROMPT_TEMPLATE["base_context"]
    label_info = PROMPT_TEMPLATE["prompts"][label]

    # שרשור טבעי כפסקה זורמת ללא כותרות קשיחות שמבלבלות את ה-AI
    prompt = f"{base} {label_info['action']} {label_info['framing']}"
    return prompt


# ============================================================================
# GENERATION
# ============================================================================

def setup_output_dir() -> None:
    """Create output directory structure."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    for label in LABELS:
        label_dir = output_path / label
        label_dir.mkdir(exist_ok=True)

    print(f"✓ Output directory structure ready: {OUTPUT_DIR}/")


def generate_image_with_retry(
    client: genai.Client,
    prompt: str,
    max_retries: int = 3,
) -> bytes | None:
    """Generate image and return bytes."""
    for attempt in range(max_retries):
        try:
            interaction = client.interactions.create(
                model=GOOGLE_MODEL,
                input=prompt,
            )

            if interaction.output_image and interaction.output_image.data:
                return base64.b64decode(interaction.output_image.data)
            return None
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠ Attempt {attempt + 1} failed: {str(e)[:80]}... Retrying...")
            else:
                print(f"  ✗ Failed after {max_retries} attempts: {str(e)}")
                return None


def generate_samples() -> None:
    """Generate one sample image per label."""
    client = genai.Client(api_key=GOOGLE_API_KEY)
    setup_output_dir()

    metadata = []
    output_path = Path(OUTPUT_DIR)

    print(f"\n🎬 Generating {len(LABELS)} sample images (1 per label)...\n")

    for label_idx, label in enumerate(LABELS, 1):
        print(f"📌 [{label_idx}/{len(LABELS)}] {label}")

        # Build prompt
        prompt = build_prompt(label)
        print(f"   Prompt: {prompt[:100]}...")

        # Generate image
        image_bytes = generate_image_with_retry(client, prompt)
        if image_bytes is None:
            print(f"   ✗ Failed to generate")
            continue

        # Save image
        image_path = output_path / label / "sample_001.png"
        try:
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            print(f"   ✓ Saved to {image_path}")
        except Exception as e:
            print(f"   ✗ Save failed: {str(e)[:80]}")
            continue

        # Record metadata
        metadata.append({
            "file": str(image_path),
            "label": label,
            "prompt_template": PROMPT_TEMPLATE["prompts"][label],
            "full_prompt": prompt,
            "model": GOOGLE_MODEL,
        })

    # Save metadata
    metadata_path = output_path / "sample_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n" + "=" * 70)
    print(f"✅ Sample generation complete!")
    print(f"   Output: {OUTPUT_DIR}/")
    print(f"   Metadata: {OUTPUT_DIR}/sample_metadata.json")
    print(f"=" * 70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("❌ ERROR: Set your Google API key")
        print("   GOOGLE_API_KEY = 'your-key-here'")
    else:
        generate_samples()
