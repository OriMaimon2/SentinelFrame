#!/usr/bin/env python3
"""
Generate sample images per label for SentinelFrame project.
Uses Google Gemini API with structured prompt templates.
"""

import json
import base64
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
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
    "no_visible_hand_interaction",
]

# ============================================================================
# GENERATION PARAMETERS
# ============================================================================
# Choose which labels to generate and how many images to create for each one.
# Labels with a count of 0 (or omitted here) are skipped entirely.

IMAGES_PER_LABEL = {
    "hand_in_pocket": 20,
    "hand_in_bag": 20,
    "hand_under_clothing": 20,
    "object_in_hand": 20,
    "interacting_with_shelf": 20,
    "hand_occluded_generic": 20,
    "both_hands_not_visible": 20,
    "no_visible_hand_interaction": 20,
}

# Set to an int for a fully reproducible run (same seed -> same sequence of
# variable choices across every generated image). Leave None to use a fresh
# seed each run — the actual seed used is always logged so any run can be
# reproduced later by pinning RANDOM_SEED to that value.
RANDOM_SEED = 42

# Number of images generated concurrently (each is a slow, network-bound API
# call, so threads — not processes — are the right tool here). Tune this down
# if you hit API rate limits, or up if the API comfortably handles more.
MAX_WORKERS = 2

# ============================================================================
# VARIABLE POOLS (RANDOMIZED PROMPT TOKENS)
# ============================================================================
# Each key corresponds to a {placeholder} token used somewhere in
# PROMPT_TEMPLATE. One value per key is chosen at random for every generated
# image via resolve_variables().

VARIABLE_POOLS = {
    "subject": ["man", "woman"],
    "age": ["young adult", "middle-aged", "elderly"],
    "body_type": ["a slim build", "an average build", "a heavy build"],
    "clothing_style": [
        "jeans and a t-shirt",
        "a hoodie and sweatpants",
        "a winter jacket and jeans",
        "sportswear",
        "a casual dress",
        "business casual attire",
        "shorts and a t-shirt",
        "a long coat",
    ],
    "hand_side": ["right", "left"],
    "outerwear": ["jacket", "hoodie", "coat"],
    "bag_type": [
        "plastic grocery bag", "paper shopping bag", "reusable shopping bag",
        "backpack", "shoulder bag", "crossbody bag", "small handbag",
        "large handbag", "tote bag", "shopping basket",
    ],
    "pocket_location": [
        "front pocket of their pants",
        "side pocket of their jacket",
        "pocket of their hoodie",
        "back pocket of their pants",
    ],
    "product": [
        "soda can", "juice bottle", "snack box", "cereal box", "bag of chips",
        "bottle of shampoo", "loaf of bread", "carton of eggs", "bottle of water",
        "chocolate bar",
    ],
    "shelf_level": ["top", "middle", "bottom"],
    "shelf_item": ["box", "can", "jar", "bottle", "package"],
    "occlusion_source": [
        "a large shelf column", "a tall shelving unit",
        "a promotional display stand", "another row of shelves",
    ],
    "both_hands_pose": [
        "with both of their arms held flat behind their lower back",
        "with both arms tightly crossed over their chest, hands tucked completely out of sight",
        "with both hands clasped together behind their back",
        "with both arms folded down at their sides and hands tucked into opposite sleeves",
        "with both arms wrapped around a shopping basket held low against their waist, hands hidden beneath it",
    ],
    "framing_realism": [
        "The subject is well-centered and clearly visible within the frame.",
        "The subject is slightly off-center within the frame, as is common in real, unattended CCTV coverage.",
        "The subject is partially cropped near the edge of the frame, with an imperfect, natural security-camera composition.",
        "The subject is captured at a slightly greater distance than usual, appearing a little smaller within the frame, typical of a wide-coverage security camera.",
        "The subject is angled partly away from the camera, so not every detail is perfectly visible, consistent with a real overhead security angle.",
    ],
    "surrounding_products": [
        "colorful snack food and confectionery packaging",
        "canned goods and dry pantry staples",
        "fresh produce displayed in open bins and crates",
        "refrigerated dairy products and cartons",
        "meat and deli products in sealed packaging",
        "bottled beverages and soft drinks",
        "household cleaning supplies and detergents",
        "bakery items and bread on open shelves",
        "frozen food packages in chest freezers",
        "pharmacy and personal care products",
    ],
    "camera_view": [
        "a ceiling-mounted CCTV camera installed in the corner of a supermarket aisle, looking diagonally downward",
        "a ceiling-mounted wide-angle security camera overlooking a supermarket aisle from above",
        "a high corner surveillance camera capturing customers walking along a shopping aisle",
        "a ceiling-mounted surveillance camera monitoring a long supermarket aisle from one end",
        "a wide-angle CCTV camera mounted near the junction of two shopping aisles",
        "a security camera mounted above the endcap of a supermarket aisle, looking diagonally across the shelves",
        "a ceiling-mounted surveillance camera providing an oblique view of a shopping aisle",
        "a fixed overhead CCTV camera covering a supermarket aisle with a wide field of view"
    ],
    "shelf_arrangement": [
        "fully stocked, neatly organized shelves",
        "densely packed shelves with products of many different colors",
        "sparsely stocked shelves with only a few items per row",
        "shelves of varying height, from low to reaching near the ceiling",
        "shelves featuring a large nearby promotional end-cap display",
    ],
    "lighting": [
        "cool white LED lighting",
        "warm white LED lighting",
        "slightly dim aisle lighting",
        "brightly illuminated aisle lighting",
        "uneven lighting with visible shadows",
        "lighting with visible reflections off a polished floor",
        "mild natural sunlight spilling in from the store entrance",
        "slightly overexposed lighting",
        "slightly underexposed, darker lighting",
    ],
    "camera_imperfections": [
        "The footage shows mild JPEG compression artifacts typical of a security DVR recording.",
        "The image has a subtle layer of sensor noise and grain, consistent with a low-cost CCTV camera.",
        "The footage is slightly low-bitrate, with soft blocky compression visible in flatter areas of the image.",
        "There is a faint motion blur on the moving subject, typical of a CCTV camera's slower shutter speed.",
        "The image shows a slight rolling-shutter skew, as is common with cheaper CCTV sensors.",
        "The camera lens appears slightly dirty or dusty, giving the footage a faint hazy quality.",
        "There is mild chromatic aberration visible near the edges of the frame.",
        "The corners of the frame show subtle vignetting, darker than the center.",
        "The footage shows a slight automatic-exposure fluctuation, as if the camera is adjusting to lighting changes.",
        "The white balance appears slightly off, giving the image a faint color cast typical of budget CCTV hardware.",
        "There is minor oversharpening visible in the footage, giving edges a slightly artificial halo.",
        "The image has a subtle softness from imperfect autofocus, typical of a fixed-focus security camera.",
        "The footage shows faint interlacing artifacts near moving edges, consistent with an older analog CCTV system.",
    ],
}

# ============================================================================
# PROMPT STRUCTURE (UPDATED FOR MAXIMUM NATURAL SURVEILLANCE REALISM)
# ============================================================================

PROMPT_TEMPLATE = {
    "base_context": (
        "A realistic wide shot frame captured by {camera_view}. "
        "Bright and authentic supermarket environment with {shelf_arrangement} lined with {surrounding_products}, "
        "and {lighting}. Crucially, the aisle is completely empty of any other people, "
        "containing only a solitary, ordinary {age} {subject} with {body_type}, wearing {clothing_style}, captured from "
        "a realistic, natural distance down the aisle, ensuring a true surveillance perspective with "
        "completely unposed, candid, and organic framing. {framing_realism} Only one single isolated human is visible. "
        "{camera_imperfections}"
    ),
"prompts": {
        "hand_in_pocket": {
            "variants": [
                {
                    "action": "A single {subject} standing casually by the shelves while shopping, with their {hand_side} hand naturally resting inside the {pocket_location}. The arm is relaxed and straight, and the hand is fully inside the pocket in a typical idle posture, with a slight natural deformation of the fabric around the pocket area.",
                    "framing": "Full-body CCTV-style surveillance view from above at a natural supermarket aisle angle, showing a normal, unposed shopping moment with a relaxed stance."
                },
            ],
        },
        "hand_in_bag": {
            "variants": [
                {
                    # מניעת בלבול: מדגישים שהיד ממש *בתוך* חלל התיק (ולא רק מחזיקה את הרצועה).
                    "action": "The {subject} is holding an open {bag_type}, with their {hand_side} hand and wrist completely plunged inside the dark main compartment, actively rummaging or reaching deep into the bottom of the bag.",
                    "framing": "The bag is held forward or to the side, with the hand disappearing past the opening of the bag visible from the high vantage point, though not always perfectly centered or fully unobstructed.",
                },
            ],
        },
        "hand_under_clothing": {
            "variants": [
                {
                    "action": (
                        "The {subject}'s {hand_side} hand is fully inserted beneath the outer {outerwear} layer, "
                        "between the {outerwear} and the torso. The fabric of the {outerwear} clearly sits on top of the hand, "
                        "fully covering it, so the hand is not visible as a shape from the outside. "
                        "There is no contact with any pockets, and no hand outline is visible on the outer surface of the clothing. "
                        "The arm entry point is at the open front seam of the {outerwear}, and the hand is completely concealed underneath the garment layer."
                    ),
                    "framing": (
                        "Only the arm entry point is slightly implied near the {outerwear} opening; "
                        "the rest of the hand is fully hidden under the clothing with no external bulging or visible grasping shape."
                    ),
                },
                {
                    "action": (
                        "The {subject}'s {hand_side} hand is slipped horizontally underneath their shirt, entering from the side at waist height "
                        "and resting flat against the torso beneath the fabric. There is no contact with any pockets, and the hand is not visible as a shape from outside."
                    ),
                    "framing": (
                        "The shirt fabric drapes naturally over the hidden hand, with only a subtle, natural fold visible where the arm enters at the side; "
                        "no bulge or grasping shape is visible from outside the fabric."
                    ),
                },
                {
                    "action": (
                        "The {subject}'s {hand_side} hand has reached up underneath their shirt from the open bottom hem, moving vertically upward along the torso "
                        "until the hand is completely hidden beneath the fabric. There is no contact with any pockets."
                    ),
                    "framing": (
                        "Only a faint vertical ripple in the shirt is visible where the arm enters near the waistline; "
                        "the hand itself is completely hidden with no external bulging shape."
                    ),
                },
                {
                    "action": (
                        "The {subject} is using both hands together underneath their shirt, holding a concealed object flat against their torso. "
                        "Both hands entered from the open front or bottom of the shirt and are completely hidden beneath the fabric. "
                        "In this pose the {subject} is not pushing a shopping cart or trolley, since both hands are occupied under the clothing."
                    ),
                    "framing": (
                        "A clear surveillance view showing both of the {subject}'s arms disappearing under the front of the shirt, "
                        "with no visible hands, fingers, or bulging shape from the fabric, and no cart or trolley being handled."
                    ),
                },
            ],
        },
        "object_in_hand": {
            "variants": [
                {
                    # מניעת בלבול: מגדירים אחיזה ברורה (Grip) של מוצר קטן, כדי שהמודל לא יצייר סתם יד פתוחה ליד מוצר.
                    "action": "The {subject} is holding a specific retail item, a {product}, in their {hand_side} hand with a clear, firm grip. Their fingers are wrapped tightly around the product, showcasing ownership of the item.",
                    "framing": "The object is held away from the shelves, showing that it has been taken; the hand and object are visible from the ceiling angle, though not always perfectly centered, fully unobstructed, or facing the camera.",
                },
            ],
        },
        "interacting_with_shelf": {
            "variants": [
                {
                    # מניעת בלבול: מפרידים בין "סתם עמידה ליד" לבין אינטראקציה פיזית (מגע, הזזה, שליפה).
                    "action": "The {subject} is actively reaching out with their {hand_side} arm extended forward, with their fingers physically touching, sliding, or pulling a specific {shelf_item} from the {shelf_level} shelf.",
                    "framing": "The physical contact between the fingertips and the product packaging is visible, captured from a natural distance down the aisle, though not always perfectly centered or fully unobstructed by other items on the shelf.",
                },
            ],
        },
        "hand_occluded_generic": {
            "variants": [
                {
                    # מניעת בלבול: מסבירים ל-AI *למה* לא רואים את היד (בגלל זווית הגוף/המדף) כדי שלא יחביא אותה בכיס בטעות.
                    "action": "The {subject} is turned sideways or has their back partially to the camera. Their {hand_side} arm is positioned behind their own torso or blocked entirely by {occlusion_source}, creating a natural physical obstruction.",
                    "framing": "A natural wide shot where the {subject} is fully visible, but their {hand_side} hand is completely blocked from the camera's line of sight by an external object or their own body.",
                },
            ],
        },
        "both_hands_not_visible": {
            "variants": [
                {
                    # מניעת בלבול: מגדירים תנוחה ספציפית שמעלימה את שתי הידיים (כמו ידיים מאחורי הגב או שלובות עמוק), ללא שימוש בכיסים.
                    "action": "The {subject} is browsing {both_hands_pose}. No pockets or bags are used.",
                    "framing": "A surveillance view of the {subject} from a distance, with the full-body profile sometimes partially cropped or off-center, but with zero hands or fingers visible anywhere.",
                },
            ],
        },
        "no_visible_hand_interaction": {
            "variants": [
                {
                    "action": "The {subject} is walking calmly down the aisle with both hands swinging naturally and visibly at their sides, empty and open, not touching any product, pocket, bag, or their own body in any way.",
                    "framing": "A natural, everyday CCTV moment showing ordinary walking behavior; both hands remain visible throughout, with no concealment, occlusion, or object handling of any kind.",
                },
                {
                    "action": "The {subject} is pushing a shopping cart down the aisle with both hands resting normally on the handle, palms and fingers visible and relaxed, not gripping any product or reaching toward their body.",
                    "framing": "A routine surveillance view of ordinary cart-pushing behavior, with both hands visible on the handle at all times and no interaction with shelves, pockets, or clothing.",
                },
                {
                    "action": "The {subject} is holding a phone in their {hand_side} hand at chest height, glancing casually at the screen while walking, with both hands visible and not touching any shelf, pocket, or clothing.",
                    "framing": "A normal, everyday CCTV moment of casual phone use while shopping; both hands stay visible and there is no concealment or shelf interaction.",
                },
                {
                    "action": "The {subject} is carrying a shopping basket by its handle in their {hand_side} hand, arm relaxed at their side, with both hands visible and empty of any concealment, simply walking through the aisle.",
                    "framing": "A routine surveillance view showing ordinary basket-carrying behavior, with both hands visible and no interaction with products, pockets, or clothing.",
                },
            ],
        },
    },
}


def resolve_variables(label: str) -> dict:
    """Pick one random value per VARIABLE_POOLS key, plus which text variant to use, for a single image."""
    chosen = {key: random.choice(options) for key, options in VARIABLE_POOLS.items()}
    variants = PROMPT_TEMPLATE["prompts"][label]["variants"]
    chosen["_variant_index"] = random.randrange(len(variants))
    return chosen


def build_prompt(label: str, chosen: dict) -> str:
    """Build complete prompt from template into a natural, seamless paragraph, substituting {placeholder} tokens with chosen values."""
    base = PROMPT_TEMPLATE["base_context"].format(**chosen)
    variant = PROMPT_TEMPLATE["prompts"][label]["variants"][chosen["_variant_index"]]

    # שרשור טבעי כפסקה זורמת ללא כותרות קשיחות שמבלבלות את ה-AI
    action = variant["action"].format(**chosen)
    framing = variant["framing"].format(**chosen)
    prompt = f"{base} {action} {framing}"
    return prompt


# ============================================================================
# LOGGING
# ============================================================================

def setup_logger(output_dir: str) -> logging.Logger:
    """Configure a logger that writes progress and failures to console and a log file."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"generation_{datetime.now():%Y%m%d_%H%M%S}.log"

    logger = logging.getLogger("generate_sample_images")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    logger.info(f"📝 Log file: {log_path}")
    return logger


# ============================================================================
# GENERATION
# ============================================================================

def setup_output_dir(labels) -> None:
    """Create output directory structure for the given labels."""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)

    for label in labels:
        label_dir = output_path / label
        label_dir.mkdir(exist_ok=True)


def next_sample_index(label_dir: Path) -> int:
    """Find the next unused sample_NNN index in a label directory, so existing images aren't overwritten."""
    indices = []
    for f in label_dir.glob("sample_*.png"):
        try:
            indices.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=0) + 1


PROGRESS_FILENAME = "generation_progress.json"


def load_progress(output_path: Path, seed: int, targets: dict) -> dict | None:
    """Load a saved run-plan checkpoint if it matches the current seed and label/count plan exactly."""
    progress_path = output_path / PROGRESS_FILENAME
    if not progress_path.exists():
        return None
    try:
        with open(progress_path) as f:
            progress = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if progress.get("seed") != seed or progress.get("images_per_label") != targets:
        return None
    return progress


def save_progress(output_path: Path, progress: dict) -> None:
    """Persist the current run-plan checkpoint (seed, plan, per-label start index, completed rounds)."""
    progress_path = output_path / PROGRESS_FILENAME
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def generate_image_with_retry(
    client: genai.Client,
    prompt: str,
    logger: logging.Logger,
    max_retries: int = 3,
) -> tuple[bytes | None, str | None]:
    """Generate image and return (image_bytes, failure_reason). failure_reason is None on success."""
    failure_reason = "Unknown error"
    for attempt in range(max_retries):
        try:
            interaction = client.interactions.create(
                model=GOOGLE_MODEL,
                input=prompt,
            )

            if interaction.output_image and interaction.output_image.data:
                return base64.b64decode(interaction.output_image.data), None

            failure_reason = "API returned no image data"
            logger.warning(f"  ⚠ Attempt {attempt + 1}: {failure_reason}")
        except Exception as e:
            failure_reason = str(e)
            if attempt < max_retries - 1:
                logger.warning(f"  ⚠ Attempt {attempt + 1} failed: {failure_reason[:120]}... Retrying...")
            else:
                logger.error(f"  ✗ Failed after {max_retries} attempts: {failure_reason}")

    return None, failure_reason


def generate_samples() -> None:
    """Generate images for each configured label according to IMAGES_PER_LABEL."""
    seed = RANDOM_SEED if RANDOM_SEED is not None else random.randrange(2**32)

    targets = {}
    unknown_labels = []
    for label, count in IMAGES_PER_LABEL.items():
        if label not in LABELS:
            unknown_labels.append(label)
            continue
        if count > 0:
            targets[label] = count

    if not targets:
        # Nothing configured to do — bail out before any logging/IO side effects.
        for label in unknown_labels:
            print(f"⚠ Skipping unknown label in IMAGES_PER_LABEL: {label}")
        print("⚠ No labels with count > 0 in IMAGES_PER_LABEL. Nothing to generate.")
        return

    output_path = Path(OUTPUT_DIR)
    progress_path = output_path / PROGRESS_FILENAME
    progress = load_progress(output_path, seed, targets)
    total_requested = sum(targets.values())

    if progress is not None:
        completed_rounds = {label: set(rounds) for label, rounds in progress["completed_rounds"].items()}
        already_done = sum(len(rounds) for rounds in completed_rounds.values())
        if already_done >= total_requested:
            # Fully completed in a previous run. Return before touching the filesystem at
            # all (no log file, no output directories, no API client) so repeatedly calling
            # generate_samples() on a finished plan is a true, side-effect-free no-op.
            print(f"✅ This exact plan ({total_requested} images) was already fully completed in a previous "
                  f"run. Nothing to do — see {progress_path}.")
            return

    # From here on there is real work to do, so it's fine to set up logging and the client.
    logger = setup_logger(OUTPUT_DIR)
    random.seed(seed)
    logger.info(f"🎲 Random seed: {seed} (set RANDOM_SEED = {seed} to reproduce this exact run)")
    for label in unknown_labels:
        logger.warning(f"⚠ Skipping unknown label in IMAGES_PER_LABEL: {label}")

    setup_output_dir(targets.keys())
    client = genai.Client(api_key=GOOGLE_API_KEY)

    metadata_path = output_path / "sample_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = []

    label_dirs = {label: output_path / label for label in targets}

    if progress is not None:
        start_indices = progress["start_indices"]
        logger.info("🔁 Resuming: saved checkpoint matches this seed and label/count plan.")
    else:
        if progress_path.exists():
            logger.info("⚠ Saved checkpoint found but seed or label/count plan differs — starting a fresh plan "
                        "(existing files are still safe; new images continue numbering after them).")
        start_indices = {label: next_sample_index(label_dirs[label]) for label in targets}
        completed_rounds = {label: set() for label in targets}
        progress = {
            "seed": seed,
            "images_per_label": targets,
            "start_indices": start_indices,
            "completed_rounds": {label: [] for label in targets},
        }
        save_progress(output_path, progress)

    already_done = sum(len(rounds) for rounds in completed_rounds.values())
    stats = {label: {"succeeded": 0, "failed": 0, "failures": []} for label in targets}
    max_count = max(targets.values())

    logger.info(f"\n🎬 Plan: {total_requested} images across {len(targets)} label(s), round-robin, "
                f"{max_count} round(s), {MAX_WORKERS} worker thread(s). {already_done} already completed, "
                f"{total_requested - already_done} remaining.\n")

    # Shared mutable state (metadata, stats, completed_rounds, progress, and their on-disk
    # writes) is only ever touched by a worker thread while holding this lock. Variable
    # resolution and the completed-slot skip check happen in the main thread below, outside
    # any thread — so the RNG draw order stays strictly sequential regardless of how workers
    # get scheduled or complete.
    state_lock = threading.Lock()

    def process_slot(label: str, round_num: int, count: int, sample_index: int, chosen: dict) -> None:
        logger.info(f"📌 {label} [{round_num + 1}/{count}] -> sample_{sample_index:03d}  (round {round_num + 1}/{max_count})")
        logger.info(f"   🎲 Variables: {chosen}")

        prompt = build_prompt(label, chosen)
        image_bytes, failure_reason = generate_image_with_retry(client, prompt, logger)

        if image_bytes is None:
            logger.error(f"   ✗ Failed to generate: {failure_reason}")
            with state_lock:
                stats[label]["failed"] += 1
                stats[label]["failures"].append({
                    "sample": f"sample_{sample_index:03d}",
                    "reason": failure_reason,
                    "variables": chosen,
                })
            return

        image_path = label_dirs[label] / f"sample_{sample_index:03d}.png"
        try:
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            logger.info(f"   ✓ Saved to {image_path}")
        except Exception as e:
            reason = f"Save failed: {e}"
            logger.error(f"   ✗ {reason}")
            with state_lock:
                stats[label]["failed"] += 1
                stats[label]["failures"].append({
                    "sample": f"sample_{sample_index:03d}",
                    "reason": reason,
                    "variables": chosen,
                })
            return

        with state_lock:
            stats[label]["succeeded"] += 1
            metadata.append({
                "file": str(image_path),
                "label": label,
                "prompt_template": PROMPT_TEMPLATE["prompts"][label],
                "full_prompt": prompt,
                "variables": chosen,
                "model": GOOGLE_MODEL,
            })

            # Written after every image (not just at the end) so metadata for
            # already-generated images survives even if the run is stopped early.
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # A slot only counts as completed once its image is actually saved, so a
            # failed attempt is retried (with the same variables) on the next run.
            completed_rounds[label].add(round_num)
            progress["completed_rounds"][label] = sorted(completed_rounds[label])
            save_progress(output_path, progress)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for round_num in range(max_count):
            for label, count in targets.items():
                if round_num >= count:
                    continue

                sample_index = start_indices[label] + round_num

                # Always draw this slot's variables, even for a slot already completed in
                # an earlier run, so the RNG sequence stays identical to an uninterrupted
                # run — this must stay sequential in the main thread, never inside a worker.
                chosen = resolve_variables(label)

                if round_num in completed_rounds[label]:
                    continue

                futures.append(executor.submit(process_slot, label, round_num, count, sample_index, chosen))

        for future in as_completed(futures):
            future.result()  # re-raise anything unexpected (workers don't normally raise)

    total_succeeded = sum(s["succeeded"] for s in stats.values())
    total_failed = sum(s["failed"] for s in stats.values())

    logger.info("\n" + "=" * 70)
    logger.info("✅ Generation complete!")
    logger.info(f"   Attempted this run: {total_succeeded + total_failed}   Succeeded: {total_succeeded}   Failed: {total_failed}")
    for label, s in stats.items():
        logger.info(f"   - {label}: {s['succeeded']} succeeded, {s['failed']} failed")
        for failure in s["failures"]:
            logger.info(f"       ✗ {failure['sample']}: {failure['reason']} (variables: {failure['variables']})")
    logger.info(f"   Output: {OUTPUT_DIR}/")
    logger.info(f"   Metadata: {metadata_path}")
    logger.info(f"   Progress checkpoint: {progress_path}")
    if total_failed:
        logger.info(f"   {total_failed} image(s) failed and will be retried automatically on the next run "
                     f"with the same seed/plan.")
    logger.info("=" * 70 + "\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
        print("❌ ERROR: Set your Google API key")
        print("   GOOGLE_API_KEY = 'your-key-here'")
    else:
        generate_samples()
