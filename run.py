import uvicorn
from src.config.settings import SETTINGS
from src.utils.logger import logger
import os

os.environ["LANGFUSE_PUBLIC_KEY"] = SETTINGS.LANGFUSE_PUBLIC_KEY
os.environ["LANGFUSE_SECRET_KEY"] = SETTINGS.LANGFUSE_SECRET_KEY
os.environ["LANGFUSE_HOST"] = SETTINGS.LANGFUSE_HOST


def main():
    # Log configuration
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
