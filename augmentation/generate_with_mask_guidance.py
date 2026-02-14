"""
MASK-GUIDED IMAGE GENERATION USING GEMINI NANO BANANA

This script uses segmentation masks as visual guides to generate eye images with Gemini.
The masks are overlaid on the original images with colored regions, and a detailed prompt
explains what each color represents (pupil, iris, sclera).

The goal is to see if providing explicit mask guidance helps the model generate more
anatomically accurate variations or transformations of eye images.

Requirements:
    pip install google-genai==1.41.0 Pillow==11.3.0 tqdm==4.67.1 numpy opencv-python

Setup:
    export GEMINI_API_KEY=<your_api_key>

Usage:
    # Basic usage - generate variants with mask guidance
    python augmentation/generate_with_mask_guidance.py \\
        --images_dir data/images \\
        --masks_dir data/masks \\
        --out_dir data/mask_guided_generation

    # With specific transformation
    python augmentation/generate_with_mask_guidance.py \\
        --images_dir data/images \\
        --masks_dir data/masks \\
        --out_dir data/generated \\
        --prompt "colorize this grayscale eye image realistically" \\
        --variants 3

    # Using config file
    python augmentation/generate_with_mask_guidance.py --config mask_guide_config.json

Arguments:
    --images_dir, -i     : Directory with source eye images (required)
    --masks_dir, -m      : Directory with corresponding segmentation masks (required)
    --out_dir, -o        : Output directory (default: mask_guided_output)
    --prompt, -p         : Transformation/generation prompt (default: maintain structure)
    --variants, -v       : Number of variants per image (default: 1)
    --temperature, -t    : Creativity level 0.0-1.0 (default: 0.7)
    --mask_opacity, -a   : Mask overlay opacity 0.0-1.0 (default: 0.4)
    --show_legend        : Add color legend to guide image (default: False)
    --config, -c         : JSON config file (optional)
    --api_key            : Gemini API key (or set GEMINI_API_KEY env var)

Mask Format:
    Expects masks in MOBIUS format:
    - Class 0 (Background/Sclera): Black or any non-iris/pupil color
    - Class 1 (Iris): Typically green in visualization
    - Class 2 (Pupil): Typically blue in visualization

Output:
    - Creates timestamped output directory
    - Saves mask-guided images (image + mask overlay)
    - Saves generated variants
    - Saves config and logs
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ========================= LOGGING SETUP =========================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.WARNING,
    force=True,
)

logger = logging.getLogger("mask_guided_generation")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
logger.propagate = False

# ========================= DEFAULTS =========================

DEFAULT_MODEL = "gemini-2.5-flash-image"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_VARIANTS = 1
DEFAULT_OUT_DIR = Path("mask_guided_output")
DEFAULT_MASK_OPACITY = 0.4
DEFAULT_ASPECT_RATIO = "1:1"

# Color scheme for mask visualization (RGB)
MASK_COLORS = {
    0: (50, 50, 50),      # Background/Sclera: Dark gray
    1: (0, 255, 0),       # Iris: Green
    2: (255, 0, 0),       # Pupil: Red
}

MASK_NAMES = {
    0: "Background/Sclera (white of the eye)",
    1: "Iris (colored ring around pupil)",
    2: "Pupil (dark center)",
}

# ========================= CONFIG DATACLASS =========================


@dataclass
class MaskGuidedConfig:
    images_dir: Optional[Path] = None
    masks_dir: Optional[Path] = None
    out_dir: Path = DEFAULT_OUT_DIR
    prompt: Optional[str] = None
    variants: int = DEFAULT_VARIANTS
    temperature: float = DEFAULT_TEMPERATURE
    mask_opacity: float = DEFAULT_MASK_OPACITY
    show_legend: bool = False
    model: str = DEFAULT_MODEL


# ========================= MASK PROCESSING =========================


def create_colored_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    opacity: float = 0.4,
    show_legend: bool = False
) -> Image.Image:
    """
    Create an image with colored mask overlay and optional legend.

    Args:
        image: Original image (H, W, 3) in RGB
        mask: Segmentation mask (H, W) with class indices
        opacity: Opacity of mask overlay (0.0 = transparent, 1.0 = opaque)
        show_legend: Whether to add color legend

    Returns:
        PIL Image with mask overlay
    """
    # Create colored mask
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in MASK_COLORS.items():
        colored_mask[mask == class_id] = color

    # Ensure image is RGB
    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Blend image and colored mask
    blended = cv2.addWeighted(image, 1.0, colored_mask, opacity, 0)

    # Convert to PIL
    result = Image.fromarray(blended)

    # Add legend if requested
    if show_legend:
        result = add_color_legend(result)

    return result


def add_color_legend(image: Image.Image) -> Image.Image:
    """
    Add a color legend to the image showing what each color represents.

    Args:
        image: PIL Image

    Returns:
        PIL Image with legend added
    """
    # Create new image with space for legend
    legend_height = 80
    new_height = image.height + legend_height
    new_img = Image.new('RGB', (image.width, new_height), (255, 255, 255))
    new_img.paste(image, (0, 0))

    # Draw legend
    draw = ImageDraw.Draw(new_img)

    try:
        # Try to use a nice font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        # Fallback to default
        font = ImageFont.load_default()

    y_offset = image.height + 10
    x_start = 10

    for class_id in [2, 1, 0]:  # Reverse order (pupil, iris, background)
        color = MASK_COLORS[class_id]
        name = MASK_NAMES[class_id]

        # Draw color box
        box_size = 20
        draw.rectangle(
            [x_start, y_offset, x_start + box_size, y_offset + box_size],
            fill=color,
            outline=(0, 0, 0)
        )

        # Draw text
        draw.text((x_start + box_size + 10, y_offset + 3), name, fill=(0, 0, 0), font=font)

        y_offset += 25

    return new_img


def load_mask(mask_path: Path) -> np.ndarray:
    """
    Load a segmentation mask from file.

    Assumes MOBIUS format:
    - Can be RGB with color-coded regions
    - Or already in class index format

    Args:
        mask_path: Path to mask file

    Returns:
        Mask array (H, W) with class indices
    """
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask_img is None:
        raise ValueError(f"Could not load mask: {mask_path}")

    # If RGB, convert to class indices
    if len(mask_img.shape) == 3:
        mask_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        h, w = mask_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # MOBIUS format: Green = Iris, Blue = Pupil, Red/Other = Background
        green_mask = (mask_rgb[:, :, 0] == 0) & (mask_rgb[:, :, 1] == 255) & (mask_rgb[:, :, 2] == 0)
        blue_mask = (mask_rgb[:, :, 0] == 0) & (mask_rgb[:, :, 1] == 0) & (mask_rgb[:, :, 2] == 255)

        mask[green_mask] = 1  # Iris
        mask[blue_mask] = 2   # Pupil
        # Everything else stays 0 (background)

    else:
        # Already in class index format
        mask = mask_img

    return mask


# ========================= PROMPT BUILDER =========================


def build_mask_guidance_prompt(user_prompt: Optional[str] = None) -> str:
    """
    Build a detailed prompt explaining the mask overlay and requested transformation.

    Args:
        user_prompt: User's transformation request (optional)

    Returns:
        Complete prompt string
    """
    base_explanation = """You are seeing an eye image with a colored overlay that shows the anatomical segmentation:

COLORED REGIONS EXPLANATION:
- RED region: PUPIL (the dark center of the eye, should be black or very dark)
- GREEN region: IRIS (the colored ring around the pupil - can be brown, blue, green, etc.)
- GRAY region: BACKGROUND/SCLERA (the white of the eye and surrounding skin)

CRITICAL REQUIREMENTS:
1. Maintain the EXACT same anatomical structure and boundaries shown by the colored regions
2. The pupil (RED region) must remain in the exact same position and size
3. The iris (GREEN region) must keep its exact circular/elliptical shape and position
4. The sclera/background (GRAY region) should remain in its current areas
5. Do NOT move, resize, or distort these anatomical regions"""

    if user_prompt:
        task_prompt = f"""

YOUR TASK:
{user_prompt}

IMPORTANT: While performing this task, you MUST preserve the anatomical structure shown by the colored overlay. The colored regions are a guide that shows where each anatomical part should be located."""
    else:
        task_prompt = """

YOUR TASK:
Generate a natural, realistic eye image that follows the anatomical structure shown by the colored overlay.
Make it photorealistic with proper lighting, texture, and detail."""

    style_requirements = """

STYLE REQUIREMENTS:
- High photographic realism
- Natural lighting
- Sharp anatomical details
- Proper eye anatomy (correct pupil, iris, and sclera)
- Remove the colored overlay in the final image (it's just a guide)"""

    return base_explanation + task_prompt + style_requirements


# ========================= MAIN GENERATION FUNCTION =========================


def generate_with_mask_guidance(
    config: MaskGuidedConfig,
    api_key: Optional[str] = None
) -> None:
    """
    Generate eye images using mask guidance with Gemini.

    Args:
        config: MaskGuidedConfig with all parameters
        api_key: Gemini API key (optional, reads from env if not provided)
    """
    logger.info("=== MASK-GUIDED IMAGE GENERATION ===")

    # Validate API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise SystemExit("Set GEMINI_API_KEY env var or pass --api_key")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Validate directories
    if config.images_dir is None or not config.images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {config.images_dir}")
    if config.masks_dir is None or not config.masks_dir.is_dir():
        raise SystemExit(f"Masks directory not found: {config.masks_dir}")

    # Get all images
    image_files = [
        p for p in config.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]

    if not image_files:
        raise SystemExit(f"No images found in {config.images_dir}")

    logger.info(f"Found {len(image_files)} images to process")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = config.out_dir / f"mask_guided_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    guides_dir = output_dir / "mask_guides"
    generated_dir = output_dir / "generated"
    guides_dir.mkdir(exist_ok=True)
    generated_dir.mkdir(exist_ok=True)

    # Setup file logging
    log_file = output_dir / "run.log"
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    logger.info(f"Saved config to {config_path}")

    # Build prompt
    prompt = build_mask_guidance_prompt(config.prompt)
    logger.info(f"Prompt length: {len(prompt)} chars")
    logger.info(f"First 200 chars: {prompt[:200]}...")

    # Track stats
    start_time = datetime.now()
    success_count = 0
    error_count = 0
    skipped_count = 0

    # Process images
    logger.info(f"Processing {len(image_files)} images...")

    for img_path in tqdm(image_files, desc="Generating with mask guidance", unit="img"):
        # Find corresponding mask
        mask_path = config.masks_dir / img_path.name

        # Try common mask naming conventions
        if not mask_path.exists():
            # Try with _mask suffix
            mask_path = config.masks_dir / f"{img_path.stem}_mask{img_path.suffix}"
        if not mask_path.exists():
            # Try with .png extension
            mask_path = config.masks_dir / f"{img_path.stem}.png"

        if not mask_path.exists():
            logger.warning(f"✗ No mask found for {img_path.name}, skipping")
            skipped_count += 1
            continue

        try:
            # Load image and mask
            image = cv2.imread(str(img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = load_mask(mask_path)

            # Create mask-guided image
            guided_img = create_colored_mask_overlay(
                image_rgb,
                mask,
                opacity=config.mask_opacity,
                show_legend=config.show_legend
            )

            # Save mask-guided image
            guide_path = guides_dir / f"{img_path.stem}_guide{img_path.suffix}"
            guided_img.save(guide_path)

            # Generate variants
            for variant_idx in range(config.variants):
                try:
                    # Call Gemini API
                    response = client.models.generate_content(
                        model=config.model,
                        contents=[prompt, guided_img],
                        config=types.GenerateContentConfig(
                            image_config=types.ImageConfig(
                                aspect_ratio=DEFAULT_ASPECT_RATIO,
                            ),
                            temperature=config.temperature,
                        ),
                    )

                    # Extract and save images
                    saved = False
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            # Create output filename
                            if config.variants == 1:
                                out_name = f"{img_path.stem}_generated{img_path.suffix}"
                            else:
                                out_name = f"{img_path.stem}_generated_v{variant_idx:02d}{img_path.suffix}"

                            out_path = generated_dir / out_name

                            # Save generated image
                            generated = Image.open(BytesIO(part.inline_data.data))
                            generated.save(out_path)
                            saved = True
                            success_count += 1
                            logger.info(f"✓ Generated: {out_name}")

                    if not saved:
                        logger.warning(f"✗ No image in response for {img_path.name} variant {variant_idx}")
                        error_count += 1

                except Exception as e:
                    logger.error(f"✗ Error generating {img_path.name} variant {variant_idx}: {e}")
                    error_count += 1

        except Exception as e:
            logger.error(f"✗ Error processing {img_path.name}: {e}")
            error_count += 1

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 50)
    logger.info("GENERATION COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"  - Mask guides: {guides_dir}")
    logger.info(f"  - Generated images: {generated_dir}")
    logger.info(f"Total images processed: {len(image_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Skipped (no mask): {skipped_count}")
    logger.info(f"Duration: {duration}")
    logger.info("=" * 50)

    # Cleanup
    logger.removeHandler(file_handler)
    file_handler.close()


# ========================= CLI PARSING =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images with mask guidance using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--images_dir", "-i",
        type=Path,
        help="Directory with source eye images"
    )
    parser.add_argument(
        "--masks_dir", "-m",
        type=Path,
        help="Directory with corresponding segmentation masks"
    )
    parser.add_argument(
        "--out_dir", "-o",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Transformation/generation prompt (optional)"
    )
    parser.add_argument(
        "--variants", "-v",
        type=int,
        default=DEFAULT_VARIANTS,
        help=f"Number of variants per image (default: {DEFAULT_VARIANTS})"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Creativity level 0.0-1.0 (default: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--mask_opacity", "-a",
        type=float,
        default=DEFAULT_MASK_OPACITY,
        help=f"Mask overlay opacity 0.0-1.0 (default: {DEFAULT_MASK_OPACITY})"
    )
    parser.add_argument(
        "--show_legend",
        action="store_true",
        help="Add color legend to guide images"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        help="JSON config file (optional)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model (default: {DEFAULT_MODEL})"
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> MaskGuidedConfig:
    """
    Load configuration from file and merge with command-line args.

    Args:
        args: Parsed command-line arguments

    Returns:
        MaskGuidedConfig with merged values
    """
    # Start with defaults
    config = MaskGuidedConfig()

    # Load from config file if provided
    if args.config is not None:
        if not args.config.is_file():
            raise SystemExit(f"Config file not found: {args.config}")

        with open(args.config, "r") as f:
            file_config = json.load(f)

        logger.info(f"Loaded config from {args.config}")

        # Update config with file values
        for key, value in file_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Override with command-line args (they take precedence)
    for key, value in vars(args).items():
        if value is not None and key not in ["config", "api_key"]:
            if hasattr(config, key):
                setattr(config, key, value)

    # Convert path strings to Path objects
    if config.images_dir is not None:
        config.images_dir = Path(config.images_dir)
    if config.masks_dir is not None:
        config.masks_dir = Path(config.masks_dir)
    if config.out_dir is not None:
        config.out_dir = Path(config.out_dir)

    return config


# ========================= MAIN ENTRY POINT =========================


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)
    generate_with_mask_guidance(config, api_key=args.api_key)


if __name__ == "__main__":
    main()
