import asyncio
from pathlib import Path

import cognee
from dotenv import load_dotenv

load_dotenv()


async def cognee_qa_demo():
    cognee.config.system_root_directory(str(Path(__file__).resolve().parent / ".cognee_system"))
    cognee.config.data_root_directory(str(Path(__file__).resolve().parent / ".data_storage"))

    file_path = Path(__file__).resolve().parent / "knowledge" / "alice_in_wonderland.txt"

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    # WARNING: wipes all cognee data on every run
    await cognee.forget(everything=True)

    try:
        # Call Cognee to process document
        await cognee.remember(data=str(file_path), dataset_name="alice_in_wonderland")

        answer = await cognee.recall(datasets=["alice_in_wonderland"], query_text="List me all the important characters in Alice in Wonderland.")
        # Query Cognee for information from provided document
        print(answer)

        answer = await cognee.recall(datasets=["alice_in_wonderland"], query_text="How did Alice end up in Wonderland?")
        print(answer)

    except Exception as e:
        raise RuntimeError(f"Cognee pipeline failed: {e}") from e


# Cognee is an async library, it has to be called in an async context
if __name__ == "__main__":
    asyncio.run(cognee_qa_demo())