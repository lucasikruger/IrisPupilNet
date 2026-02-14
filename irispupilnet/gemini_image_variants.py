"""
GEMINI IMAGE GENERATION / EDITING (NANO BANANA)
This script uses the Gemini API to generate variations of input images based on a text prompt and optional reference images.
It can be used for data augmentation or creative image editing.

Requirements:
- Python 3.8+
- Install the Gemini API client, Pillow, and tqdm:

    pip install google-genai==1.41.0 Pillow==11.3.0 tqdm==4.67.1


- Set the GEMINI_API_KEY environment variable with your API key.

    export GEMINI_API_KEY=<your_api_key>

Output dir:
The script creates an output directory, within it a named and timestamped subfolder with the generated images.

Config file:
You can provide a JSON config file with parameters, or pass them as command-line arguments. Command-line args override those in the config file.
A new config file will be saved in the output directory for reference. You can have a config file only with the parameters you want to set, the rest will take default values or command-line args.

Logs:
Logs are printed to console and saved to a run.log file in the output directory.

Avoid using reference images:
You can use the flags --avoid_reference_one and --avoid_reference_two to avoid using the reference images if they were given in the config file or command-line args. This is useful if you want to quickly disable the reference images without modifying the config file.

Arguments:
- `--config_file` / `-c` : Path to a JSON config file with parameters (optional, command-line args override those in the config)
- `--in_dir` / `-i` : Directory with source images (required if not in config file)
- `--out_dir` / `-o` : Directory to save outputs (default: ./out)
- `--prompt` / `-p` : Edit instruction, e.g., "make it rainy" (required if not in config file)
- `--image_reference_one` / `-r1` : Reference image 1 to guide style (optional)
- `--image_reference_two` / `-r2` : Reference image 2 to guide style (optional)
- `--avoid_reference_one` / `-ar1` : Avoid using reference image 1 if given in the config or command-line args (optional)
- `--avoid_reference_two` / `-ar2` : Avoid using reference image 2 if given in the config or command-line args (optional)
- `--variants` / `-v` : Number of outputs per image in the input dir (default: 1)
- `--name` / `-n` : Name for this batch, used in output folder (default: "batch-edit")
- `--api_key` : `Gemini` API key, or set GEMINI_API_KEY env var
- `--temperature` / `-t` : Creativity level from 0.0 (more deterministic) to 1.0 (more creative) (default: 0.7)
- `--model` / `-m` : Gemini model to use (default: "gemini-2.5-flash-image")
- `--aspect_ratio` / `-a` : Aspect ratio for generated images, options: "1:1", "4:5", "3:4", "9:16", "16:9" (default: "16:9")
- `--help` / `-h` : Show help message and exit

Examples:

    python gemini_image_variants.py --in_dir path/to/images --prompt "make it rainy, wet streets, visible raindrops" --variants 3

    python gemini_image_variants.py --in_dir ./input_images --prompt_file ./prompt.txt --image_reference_one ./style1.jpg --image_reference_two ./style2.jpg --variants 2 --name "rainy_street"

    python gemini_image_variants.py -i ./input -p "make it rainy, wet streets, visible raindrops" -r1 ./ref_style.jpg -v 4 -n "raindrops" -m "gemini-2.5-flash-image" -a "4:5" -t 0.5

    python gemini_image_variants.py -c ./config.json

    python gemini_image_variants.py -c ./config.json --in_dir ./override_input --prompt "sunny day" --avoid_reference_one --variants 2

Prompting:

Best Practices (To elevate your results from good to great, incorporate these professional strategies into your workflow):
- Be Hyper-Specific: The more detail you provide, the more control you have. Instead of "fantasy armor," describe it: "ornate elven plate armor, etched with silver leaf patterns, with a high collar and pauldrons shaped like falcon wings."
- Provide Context and Intent: Explain the purpose of the image. The model's understanding of context will influence the final output. For example, "Create a logo for a high-end, minimalist skincare brand" will yield better results than just "Create a logo."
- Iterate and Refine: Don't expect a perfect image on the first try. Use the conversational nature of the model to make small changes. Follow up with prompts like, "That's great, but can you make the lighting a bit warmer?" or "Keep everything the same, but change the character's expression to be more serious."
- Use Step-by-Step Instructions: For complex scenes with many elements, break your prompt into steps. "First, create a background of a serene, misty forest at dawn. Then, in the foreground, add a moss-covered ancient stone altar. Finally, place a single, glowing sword on top of the altar."
- Use "Semantic Negative Prompts": Instead of saying "no cars," describe the desired scene positively: "an empty, deserted street with no signs of traffic."
- Control the Camera: Use photographic and cinematic language to control the composition. Terms like wide-angle shot, macro shot, low-angle perspective.

Limitations:
- For best performance, use the following languages: EN, es-MX, ja-JP, zh-CN, hi-IN.
- Image generation does not support audio or video inputs.
- The model won't always follow the exact number of image outputs that the user explicitly asks for.
- The model works best with up to 3 images as an input.
- When generating text for an image, Gemini works best if you first generate the text and then ask for an image with the text.
- Uploading images of children is not currently supported in EEA, CH, and UK.
- All generated images include a SynthID watermark.

References:

https://ai.google.dev/gemini-api/docs/image-generation
https://ai.google.dev/gemini-api/docs/imagen#imagen-configuration
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path

# Gemini API
from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm

# ========================= SETUP LOGGING =========================

# Make root quiet (hides google/*, httpx, etc.)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s :line: %(lineno)d | %(message)s",
    level=logging.WARNING,  # root at WARNING (was INFO)
    force=True,
)

# Our app logger emits INFO without propagating to root
logger = logging.getLogger("nano_banana")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s :line: %(lineno)d | %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


# ========================= GLOBAL DEFAULT VARS ==============================

DEFAULT_MODEL = "gemini-2.5-flash-image"  # image generation/editing (aka Nano Banana)
DEFAULT_ASPECT_RATIO = "16:9"  # "1:1", "4:5", "3:4", "9:16", "16:9"
DEFAULT_TEMPERATURE = 0.7  # 0.0 (more deterministic) to 1.0 (more creative)
DEFAULT_BATCH_NAME = "batch-edit"
DEFAULT_VARIANTS_NUMBER = 1  # Number of variants to generate per input image
DEFAULT_OUT_DIR_PATH = Path("out")  # Where to save outputs

# ======================== DEFAULT CONFIG ========================


@dataclass
class ImageGenerationConfig:
    in_dir: Path = None
    "Path to directory with source images"

    out_dir: Path = DEFAULT_OUT_DIR_PATH
    "Path to directory where output images will be saved"

    prompt: str | None = None
    "Text prompt to guide image generation/editing"

    image_reference_one: Path | None = None
    "Path to first image reference"

    image_reference_two: Path | None = None
    "Path to second image reference"

    variants: int = DEFAULT_VARIANTS_NUMBER
    "Number of image variants to generate per input image"

    name: str = DEFAULT_BATCH_NAME
    "Name for this batch, used in output folder"

    temperature: float = DEFAULT_TEMPERATURE
    "Creativity level from 0.0 (more deterministic) to 1.0 (more creative)"

    model: str = DEFAULT_MODEL
    "Gemini model to use for image generation/editing"

    aspect_ratio: str = DEFAULT_ASPECT_RATIO
    "Aspect ratio for generated images, options: '1:1', '4:5', '3:4', '9:16', '16:9'"


# ========================= MAIN FUNCTION ========================


def main():
    # Load command-line args
    args = parse_arguments()

    # Get the configs
    config = load_args_and_config(args=args, config_file=args.config)

    # Run the image generation with the final config
    generate_gemini_image_variants(config=config, api_key=args.api_key)


# ========================= GEMINI IMAGE GENERATION FUNCTION ========================


def generate_gemini_image_variants(
    config: ImageGenerationConfig, api_key: str = None
) -> None:
    """
    Generate image variants using the Gemini API based on the provided config.
    It creates an output directory, within it a named and timestamped subfolder with the generated images.
    For each image in the input directory, the API will generate the specified number of variants based on the text prompt and optional reference images.


    Args:
        config (ImageGenerationConfig): Configuration parameters for image generation.

        The config should include:
            - in_dir: Directory with source images
            - out_dir: Directory to save outputs
            - prompt: Edit instruction, e.g., "make it rainy"
            - image_reference_one: Reference image 1 to guide style (optional: Path or None)
            - image_reference_two: Reference image 2 to guide style (optional: Path or None)
            - variants: Number of outputs per image in the input dir
            - name: Name for this batch, used in output folder
            - api_key: Gemini API key, or set GEMINI_API_KEY env var
            - temperature: Creativity level from 0.0 (more deterministic) to 1.0 (more creative)
            - model: Gemini model to use
            - aspect_ratio: Aspect ratio for generated images

        api_key (str, optional): Gemini API key. If not provided, it will be read from the GEMINI_API_KEY environment variable.
    """
    logger.info(f"=== GEMINI IMAGE VARIANTS GENERATION ===")
    logger.info("Validating config...")

    # === Check batch_name ===
    if not config.name or not config.name.strip():
        raise SystemExit("Batch name (--name) cannot be empty")
    if any(c in config.name for c in "=_."):
        raise SystemExit(
            f"Batch name (--name) contains invalid characters (=, _, .): {config.name}"
        )

    # === Check Gemini API key ===
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY", None)

    if api_key is None:
        raise SystemExit("Set GEMINI_API_KEY env var or pass --api_key")

    # === Init Gemini client ===
    client = genai.Client(api_key=api_key)

    # === Check input dir ===
    if not config.in_dir.is_dir():
        raise SystemExit(f"Input dir not found: {config.in_dir}")

    # === Check variants ===
    if config.variants < 1:
        raise SystemExit("Variants >= 1")

    # === Input images ===
    imgs = [
        p
        for p in config.in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]

    # Check we have images
    if len(imgs) == 0:
        raise SystemExit(f"No images found in {config.in_dir}")

    # === Reference images ===
    image_reference_one = config.image_reference_one
    if image_reference_one is not None:
        if not image_reference_one.is_file():
            raise SystemExit(f"Reference image 1 not found: {image_reference_one}")
        image_reference_one = Image.open(image_reference_one)

    image_reference_two = config.image_reference_two
    if image_reference_two is not None:
        if not image_reference_two.is_file():
            raise SystemExit(f"Reference image 2 not found: {image_reference_two}")
        image_reference_two = Image.open(image_reference_two)

    # === Prepare prompt ===
    prompt_text = config.prompt

    # Validate prompt text
    if prompt_text is None:
        raise SystemExit("No prompt provided")

    if not prompt_text.strip():
        raise SystemExit("Inline prompt is empty")

    # === Timestamp for this generation run ===
    start_generation_time = datetime.now().astimezone()

    # === Where to save this image's outputs ===
    base_out = (
        config.out_dir
        / f"{config.name}_{start_generation_time.strftime('%Y-%m-%d_%H:%M:%S')}"
    )
    base_out.mkdir(parents=True, exist_ok=True)

    # === Activate file logging ===
    file_handler = activate_file_logging(base_out)

    # === Starting time ===
    starting_time = datetime.now().astimezone()

    # === Log config ===
    logger.info("=== STARTING GEMINI IMAGE VARIANTS GENERATION ===")
    logger.info(f"Start time: {starting_time.isoformat()}")
    logger.info(f"Config: {json.dumps(config.__dict__, indent=2, default=str)}")

    # === Save config file ===
    with open(base_out / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)

    logger.info(f"Saved config to {base_out / 'config.json'}")

    # === Process each image ===
    for img_path in tqdm(imgs, desc="Image generation", unit="img"):

        # Read the source image once
        src_img = Image.open(img_path)

        # Add the prompt and the source image to the contents
        contents = [prompt_text, src_img]

        # Add reference images if given
        if config.image_reference_one is not None:
            contents.append(image_reference_one)

        if config.image_reference_two is not None:
            contents.append(image_reference_two)

        # Generate the requested number of variants
        for variant_number in range(config.variants):

            # Call the Gemini API
            try:
                resp = client.models.generate_content(
                    model=config.model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(
                            aspect_ratio=config.aspect_ratio,
                        ),
                        temperature=config.temperature,
                    ),
                )
            except Exception as e:
                logger.warning(f"{img_path.name}: request failed: {e}")
                continue

            # Prepare output file path
            out_file = (
                base_out
                / f"{img_path.stem}_gen={config.name}_variant={variant_number}_gentime={start_generation_time.isoformat()}.png"
            )

            # Get the image(s) from the response (it could have text parts too)
            for part in resp.candidates[0].content.parts:
                if part.text is not None:
                    logger.info(f"Text part of {out_file.stem}: {part.text}")
                elif part.inline_data is not None:
                    image = Image.open(BytesIO(part.inline_data.data))
                    image.save(out_file)

    # === Finished ===
    finishing_time = datetime.now().astimezone()

    # Summary log
    logger.info(f"Finished. Outputs saved to: {base_out}")
    logger.info(f"Finishing time: {finishing_time.isoformat()}")
    logger.info(f"Total duration: {finishing_time - starting_time}")
    logger.info(f"Total images processed: {len(imgs)}")
    logger.info("All done!!!")

    logger.removeHandler(file_handler)
    file_handler.close()


# ========================= HELPER FUNCTIONS ========================


def load_args_and_config(
    args: argparse.Namespace, config_file: Path | None
) -> ImageGenerationConfig:
    """
    Load the configuration from a JSON file and override with command-line arguments.
    Args:
        args (argparse.Namespace): Command-line arguments.
        config_file (Path | None): Path to a JSON config file.
    Returns:
        ImageGenerationConfig: The final configuration dataclass with merged values.

    """

    # Create a config dataclass with default values
    config = ImageGenerationConfig()

    # Save the default key,values as a copy to check later which args were overridden
    config.__dict__.copy()

    # -- CONFIG FILE --

    # Load config file if given
    if config_file is not None:
        if not config_file.is_file():
            raise SystemExit(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            file_config = json.load(f)
        logger.info(f"Loaded config from {config_file}")
        logger.info(f"Config file contents: {json.dumps(file_config, indent=2)}")

        # Update dataclass with file config
        config = ImageGenerationConfig(**file_config)

    # -- COMMAND-LINE ARGS --

    # Override config values with command-line args if given
    for key, value in vars(args).items():
        # Only override if the command-line arg is not None and not one of the special keys
        if value is not None and key not in [
            "config",
            "api_key",
            "avoid_reference_one",
            "avoid_reference_two",
        ]:
            setattr(config, key, value)
            logger.info(
                f"Overriding config key '{key}' with command-line arg value: {value}"
            )

    # Avoid using reference images if the flags are set, even if they were given in config or command-line args
    if args.avoid_reference_one:
        config.image_reference_one = None
        logger.info(f"Overriding config key 'image_reference_one' to avoid using it")
    if args.avoid_reference_two:
        config.image_reference_two = None
        logger.info(f"Overriding config key 'image_reference_two' to avoid using it")

    # Convert the path-like strings to Path objects
    if config.in_dir is not None:
        config.in_dir = Path(config.in_dir)
    if config.out_dir is not None:
        config.out_dir = Path(config.out_dir)
    if config.image_reference_one is not None:
        config.image_reference_one = Path(config.image_reference_one)
    if config.image_reference_two is not None:
        config.image_reference_two = Path(config.image_reference_two)

    return config


def activate_file_logging(base_out: Path) -> logging.FileHandler:
    """
    Attach a file handler to the existing logger to log messages to a file in the specified output directory.
    Args:
        base_out (Path): The base output directory where the log file will be saved.

    Returns:
        logging.FileHandler: The file handler that was added to the logger.
    """
    log_path = base_out / "run.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s :line: %(lineno)d | %(message)s")
    )
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(f"File logging activated -> {log_path}")
    return file_handler


def parse_arguments() -> argparse.Namespace:
    """
    Parse the command-line arguments.

    The script accepts the following command-line arguments:
        - `--config` or `-c`: Path to a JSON config file with parameters, if provided the command-line args will override those in the config (optional)
        - `--in_dir` or `-i`: Directory with source images (required if not in config file)
        - `--out_dir` or `-o`: Directory to save outputs (default: ./out)
        - `--prompt` or `-p`: Edit instruction, e.g., "make it rainy" (required if not in config file)
        - `--image_reference_one` or `-r1`: Reference image 1 to guide style (optional)
        - `--image_reference_two` or `-r2`: Reference image 2 to guide style (optional)
        - `--variants` or `-v`: Number of outputs per image in the input dir (default: 1)
        - `--name` or `-n`: Name for this batch, used in output folder (default: "batch-edit")
        - `--api_key`: Gemini API key, or set GEMINI_API_KEY env var (required if not set in env)
        - `--temperature` or `-t`: Creativity level from 0.0 (more deterministic) to 1.0 (more creative) (default: 0.7)
        - `--model` or `-m`: Gemini model to use (default: "gemini-2.5-flash-image")
        - `--aspect_ratio` or `-a`: Aspect ratio for generated images, options: "1:1", "4:5", "3:4", "9:16", "16:9" (default: "16:9")
        - `--help` or `-h`: Show help message and exit

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    ap = argparse.ArgumentParser(
        description="GEMINI IMAGE DEFAULT_VARIANTS_NUMBER GENERATION"
    )

    ap.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to a JSON config file with parameters (optional, command-line args ignored if given)",
    )

    ap.add_argument(
        "--in_dir",
        "-i",
        required=False,
        type=Path,
        help="Path to directory with source images",
    )
    ap.add_argument(
        "--out_dir",
        "-o",
        type=Path,
        help=f"Path to directory to save outputs, default: {DEFAULT_OUT_DIR_PATH}",
    )
    ap.add_argument(
        "--prompt", "-p", type=str, help="Edit instruction, e.g., 'make it rainy'"
    )
    ap.add_argument(
        "--image_reference_one",
        "-r1",
        type=Path,
        help="Path to reference image 1 to guide style",
    )
    ap.add_argument(
        "--image_reference_two",
        "-r2",
        type=Path,
        help="Path to reference image 2 to guide style",
    )
    ap.add_argument(
        "--avoid_reference_one",
        action="store_true",
        help="Avoid using reference image 1 if given in the config or command-line args",
    )
    ap.add_argument(
        "--avoid_reference_two",
        action="store_true",
        help="Avoid using reference image 2 if given in the config or command-line args",
    )
    ap.add_argument(
        "--variants",
        "-v",
        type=int,
        help=f"Number of outputs per image in the input dir. Default: {DEFAULT_VARIANTS_NUMBER}",
    )
    ap.add_argument(
        "--name",
        "-n",
        type=str,
        help=f"Name for this batch, used in output folder. Default: {DEFAULT_BATCH_NAME}",
    )
    ap.add_argument(
        "--temperature",
        "-t",
        type=float,
        help=f"Creativity level from 0.0 (more deterministic) to 1.0 (more creative). Default: {DEFAULT_TEMPERATURE}",
    )
    ap.add_argument(
        "--model",
        "-m",
        type=str,
        help=f"Gemini model to use, default: {DEFAULT_MODEL}",
    )
    ap.add_argument(
        "--aspect_ratio",
        "-a",
        type=str,
        help=f"Aspect ratio for generated images. Default: {DEFAULT_ASPECT_RATIO}",
    )

    ap.add_argument(
        "--api_key",
        type=str,
        help="Gemini API key, or set GEMINI_API_KEY env var",
    )

    return ap.parse_args()


# ========================= MAIN ENTRY POINT =========================

if __name__ == "__main__":
    main()
