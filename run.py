import argparse
import os

parser = argparse.ArgumentParser(description="Run the RAG Ops application.")
parser.add_argument(
    "--provider",
    choices=["lm-studio", "gemini", "groq"],
    required=True,
    help="Specify the LLM provider to use.",
)

# Use parse_known_args() to be compatible with uvicorn's reloader,
# which might add its own arguments.
args, _ = parser.parse_known_args()

# Set the environment variables based on the provider
os.environ["LITELLM_MODEL"] = args.provider
os.environ["LITELLM_GUARDRAIL_MODEL"] = f"{args.provider}-guardrail"


import uvicorn
from src.config.settings import SETTINGS
from src.utils.logger import logger
from dotenv import load_dotenv

load_dotenv()


def main():
    logger.info(f"Using provider: {SETTINGS.LITELLM_MODEL}")
    logger.info(f"HOST: {SETTINGS.HOST}")
    logger.info(f"PORT: {SETTINGS.PORT}")

    # Configure Uvicorn settings
    uvicorn_config = {
        "app": "src.main:app",
        "host": SETTINGS.HOST,
        "port": SETTINGS.PORT,
        "reload": True,
    }

    # Start Uvicorn server
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
