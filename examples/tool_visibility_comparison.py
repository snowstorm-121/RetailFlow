"""Paired development diagnostics; not an unseen benchmark or success-rate estimate."""
import argparse
import asyncio
import json
from pathlib import Path

from retailflow.runtime import RetailRuntime


CASES = {
    "query": "I am Yara Muller, ZIP 85041. What is the status of order #W5056519?",
    "cancel_proposal": "I am Yara Muller, ZIP 85041. Please cancel order #W5056519 because I ordered it by mistake.",
}


async def compare(config_path, output):
    config = json.loads(config_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "base-config.json").write_text(json.dumps(config, indent=2))
    for case, message in CASES.items():
        for enabled in (False, True):
            label = f"{case}-{'filtered' if enabled else 'baseline'}"
            directory = output / label
            worker = RetailRuntime(directory / "business.sqlite", directory / "traces")
            result = await worker.run_turn(label, {"kind": "message", "text": message},
                                           {**config, "state_tool_filter": enabled})
            # Persist every result, including failures. No confirmations are submitted.
            (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(label, result["status"], result["failure_reason"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(compare(args.config, args.output))
