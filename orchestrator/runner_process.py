from __future__ import annotations

import argparse
import json
import signal
import threading

from .carla_runner.models import SimulationRunRequest
from .carla_runner.simulation_service import _simulation_worker


class StdoutEnvelopeQueue:
    def put(self, envelope: dict) -> None:
        print(json.dumps(envelope), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-file", required=True)
    parser.add_argument("--runtime-settings-file", required=True)
    args = parser.parse_args()

    request = SimulationRunRequest.model_validate_json(open(args.request_file, "r", encoding="utf-8").read())
    runtime_settings = json.loads(open(args.runtime_settings_file, "r", encoding="utf-8").read())

    stop_event = threading.Event()
    pause_event = threading.Event()

    def handle_signal(signum, frame):  # type: ignore[no-untyped-def]
        stop_event.set()

    def handle_pause(signum, frame):  # type: ignore[no-untyped-def]
        pause_event.set()

    def handle_resume(signum, frame):  # type: ignore[no-untyped-def]
        pause_event.clear()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGUSR1, handle_pause)
    signal.signal(signal.SIGUSR2, handle_resume)

    _simulation_worker(
        request.model_dump(),
        runtime_settings,
        StdoutEnvelopeQueue(),
        stop_event,
        pause_event,
    )


if __name__ == "__main__":
    main()
