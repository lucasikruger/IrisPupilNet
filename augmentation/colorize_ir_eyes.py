"""
IR EYE COLORIZATION USING GEMINI NANO BANANA

This script uses Gemini's image generation capabilities to colorize grayscale/IR eye images,
transforming them into realistic colored eye images for data augmentation.

The script processes all images in an input directory and saves the colorized versions
to an output directory, maintaining the original filenames.

Requirements:
    pip install google-genai==1.41.0 Pillow==11.3.0 tqdm==4.67.1

Setup:
    export GEMINI_API_KEY=<your_api_key>

Usage:
    # Basic usage
    python augmentation/colorize_ir_eyes.py \\
        --in_dir data/ir_eyes \\
        --out_dir data/colorized_eyes

    # With custom settings
    python augmentation/colorize_ir_eyes.py \\
        --in_dir data/ir_eyes \\
        --out_dir data/colorized_eyes \\
        --variants 3 \\
        --temperature 0.8 \\
        --eye_color brown

    # Using a config file
    python augmentation/colorize_ir_eyes.py --config colorize_config.json

Arguments:
    --in_dir, -i         : Input directory with IR/grayscale eye images (required)
    --out_dir, -o        : Output directory for colorized images (default: colorized_eyes)
    --variants, -v       : Number of colorized versions per image (default: 1)
    --temperature, -t    : Creativity level 0.0-1.0 (default: 0.7)
    --eye_color, -e      : Desired eye color: auto, brown, blue, green, hazel, gray (default: auto)
    --reference_image, -r: Reference image for style guidance (optional)
    --batch_size, -b     : Process images in batches (default: process all)
    --config, -c         : JSON config file (optional)
    --api_key            : Gemini API key (or set GEMINI_API_KEY env var)
    --model, -m          : Gemini model (default: gemini-2.5-flash-image)

Example config.json:
    {
        "in_dir": "data/ir_eyes",
        "out_dir": "data/colorized_eyes",
        "variants": 2,
        "temperature": 0.7,
        "eye_color": "brown"
    }

Output:
    - Creates timestamped output directory: <out_dir>/colorized_YYYY-MM-DD_HH-MM-SS/
    - Preserves original filenames with variant suffix
    - Saves config.json and run.log for reproducibility
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

# ========================= LOGGING SETUP =========================

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.WARNING,
    force=True,
)

logger = logging.getLogger("ir_colorizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)
logger.propagate = False

# ========================= DEFAULTS =========================

DEFAULT_MODEL = "gemini-2.5-flash-image"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_VARIANTS = 1
DEFAULT_OUT_DIR = Path("colorized_eyes")
DEFAULT_ASPECT_RATIO = "1:1"  # Eyes are typically square crops

# Eye color options
EYE_COLORS = {
    "auto": "realistic natural eye colors appropriate for the person",
    "brown": "warm brown eyes with natural depth and detail",
    "blue": "vivid blue eyes with realistic iris patterns",
    "green": "striking green eyes with natural variations",
    "hazel": "hazel eyes with brown and green color mixing",
    "gray": "cool gray eyes with subtle blue undertones",
}

# ========================= CONFIG DATACLASS =========================


@dataclass
class ColorizeConfig:
    in_dir: Optional[Path] = None
    out_dir: Path = DEFAULT_OUT_DIR
    variants: int = DEFAULT_VARIANTS
    temperature: float = DEFAULT_TEMPERATURE
    eye_color: str = "auto"
    reference_image: Optional[Path] = None
    batch_size: Optional[int] = None
    model: str = DEFAULT_MODEL


# ========================= PROMPT BUILDER =========================


def build_colorization_prompt(eye_color: str = "auto") -> str:
    """
    Build a detailed prompt for colorizing IR eye images.

    Args:
        eye_color: Desired eye color (auto, brown, blue, green, hazel, gray)

    Returns:
        Detailed prompt string for Gemini
    """
    color_spec = EYE_COLORS.get(eye_color.lower(), EYE_COLORS["auto"])

    prompt = f"""Transform this infrared/grayscale eye image into a realistic, natural-looking colored photograph of an eye.

Requirements:
- Colorize with {color_spec}
- Maintain the exact same iris and pupil structure and patterns
- Add realistic skin tones around the eye (eyelids, corners, lashes)
- Preserve all anatomical details and boundaries
- Create photorealistic lighting and texture
- Keep the same composition and framing
- Ensure the pupil remains black/very dark
- Add subtle color variations in the iris for realism
- Include natural eyelash coloring (dark brown/black)
- Add realistic sclera color (white with subtle veins)

Style: High-quality photographic realism, natural lighting, sharp details, professional eye photography."""

    return prompt


# ========================= MAIN COLORIZATION FUNCTION =========================


def colorize_ir_eyes(config: ColorizeConfig, api_key: Optional[str] = None) -> None:
    """
    Colorize IR/grayscale eye images using Gemini API.

    Args:
        config: ColorizeConfig with all parameters
        api_key: Gemini API key (optional, reads from env if not provided)
    """
    logger.info("=== IR EYE COLORIZATION ===")

    # Validate API key
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        raise SystemExit("Set GEMINI_API_KEY env var or pass --api_key")

    # Initialize Gemini client
    client = genai.Client(api_key=api_key)

    # Validate input directory
    if config.in_dir is None or not config.in_dir.is_dir():
        raise SystemExit(f"Input directory not found: {config.in_dir}")

    # Get all images
    images = [
        p for p in config.in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    ]

    if not images:
        raise SystemExit(f"No images found in {config.in_dir}")

    logger.info(f"Found {len(images)} images to process")

    # Load reference image if provided
    reference_img = None
    if config.reference_image is not None:
        if not config.reference_image.is_file():
            raise SystemExit(f"Reference image not found: {config.reference_image}")
        reference_img = Image.open(config.reference_image)
        logger.info(f"Using reference image: {config.reference_image}")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = config.out_dir / f"colorized_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

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
    prompt = build_colorization_prompt(config.eye_color)
    logger.info(f"Using eye color: {config.eye_color}")
    logger.info(f"Prompt: {prompt[:100]}...")

    # Track stats
    start_time = datetime.now()
    success_count = 0
    error_count = 0

    # Process images
    logger.info(f"Processing {len(images)} images...")

    for img_path in tqdm(images, desc="Colorizing eyes", unit="img"):
        try:
            # Load source image
            src_img = Image.open(img_path)

            # Prepare contents for API
            contents = [prompt, src_img]
            if reference_img is not None:
                contents.append(reference_img)

            # Generate variants
            for variant_idx in range(config.variants):
                try:
                    # Call Gemini API
                    response = client.models.generate_content(
                        model=config.model,
                        contents=contents,
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
                                out_name = f"{img_path.stem}_colorized{img_path.suffix}"
                            else:
                                out_name = f"{img_path.stem}_colorized_v{variant_idx:02d}{img_path.suffix}"

                            out_path = output_dir / out_name

                            # Save colorized image
                            colorized = Image.open(BytesIO(part.inline_data.data))
                            colorized.save(out_path)
                            saved = True
                            success_count += 1
                            logger.info(f"✓ Saved: {out_name}")

                    if not saved:
                        logger.warning(f"✗ No image in response for {img_path.name} variant {variant_idx}")
                        error_count += 1

                except Exception as e:
                    logger.error(f"✗ Error processing {img_path.name} variant {variant_idx}: {e}")
                    error_count += 1

        except Exception as e:
            logger.error(f"✗ Error loading {img_path.name}: {e}")
            error_count += 1

    # Final summary
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 50)
    logger.info("COLORIZATION COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total images processed: {len(images)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Duration: {duration}")
    logger.info("=" * 50)

    # Cleanup
    logger.removeHandler(file_handler)
    file_handler.close()


# ========================= CLI PARSING =========================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Colorize IR/grayscale eye images using Gemini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--in_dir", "-i",
        type=Path,
        help="Input directory with IR/grayscale eye images"
    )
    parser.add_argument(
        "--out_dir", "-o",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--variants", "-v",
        type=int,
        default=DEFAULT_VARIANTS,
        help=f"Number of colorized versions per image (default: {DEFAULT_VARIANTS})"
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Creativity level 0.0-1.0 (default: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--eye_color", "-e",
        type=str,
        default="auto",
        choices=list(EYE_COLORS.keys()),
        help="Desired eye color (default: auto)"
    )
    parser.add_argument(
        "--reference_image", "-r",
        type=Path,
        help="Reference image for style guidance (optional)"
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        help="Process in batches (optional)"
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
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model (default: {DEFAULT_MODEL})"
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> ColorizeConfig:
    """
    Load configuration from file and merge with command-line args.

    Args:
        args: Parsed command-line arguments

    Returns:
        ColorizeConfig with merged values
    """
    # Start with defaults
    config = ColorizeConfig()

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
    if config.in_dir is not None:
        config.in_dir = Path(config.in_dir)
    if config.out_dir is not None:
        config.out_dir = Path(config.out_dir)
    if config.reference_image is not None:
        config.reference_image = Path(config.reference_image)

    return config


# ========================= MAIN ENTRY POINT =========================


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config(args)
    colorize_ir_eyes(config, api_key=args.api_key)


if __name__ == "__main__":
    main()
