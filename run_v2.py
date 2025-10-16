from __future__ import annotations
from v2.orchestrator import run_once

if __name__ == "__main__":
    result = run_once()
    if isinstance(result, tuple):
        episode = result[0] if len(result) > 0 else {}
        summary = result[1] if len(result) > 1 else None
        print(episode)
        if summary is not None:
            print(summary)
    else:
        print(result)
