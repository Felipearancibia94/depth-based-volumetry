"""Local notifications: desktop pop-ups, a loopback webhook, or plain log lines.

Notifications are built from deterministic facts (``PetEvent.summary``), never from model
text, and are rate-limited per condition. A failing notifier is logged and ignored so it
cannot stop the pipeline.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TextIO

from perception_lab.core.config import NotifyConfig
from perception_lab.events.model import PetEvent
from perception_lab.vlm.providers import post_json, require_loopback

log = logging.getLogger(__name__)


class Notifier(Protocol):
    def notify(self, event: PetEvent) -> None: ...


class DesktopNotifier:
    """Uses ``notify-send`` (libnotify) when it is installed; otherwise does nothing."""

    def __init__(self, runner: Callable[..., Any] = subprocess.run) -> None:
        self.command = shutil.which("notify-send")
        self._run = runner
        if self.command is None:
            log.info("notify-send not found; desktop notifications disabled")

    def notify(self, event: PetEvent) -> None:
        if self.command is None:
            return
        urgency = "critical" if event.rule in ("zone_entry", "long_inactivity") else "normal"
        self._run(
            [
                self.command,
                "--app-name=lab",
                f"--urgency={urgency}",
                "Pet activity",
                event.summary(),
            ],
            timeout=5,
            check=False,
        )


class WebhookNotifier:
    def __init__(self, url: str, *, allow_remote: bool = False, post=post_json) -> None:
        require_loopback(url, allow_remote)
        self.url = url
        self._post = post

    def notify(self, event: PetEvent) -> None:
        self._post(self.url, {"summary": event.summary(), "event": event.to_dict()}, 5.0)


class LogNotifier:
    def __init__(self, stream: TextIO = sys.stdout) -> None:
        self.stream = stream

    def notify(self, event: PetEvent) -> None:
        print(f"[{event.detected_at}] {event.summary()}", file=self.stream, flush=True)


class NotificationPolicy:
    """Fan out to notifiers, filtered by rule and limited to one message per condition per
    ``min_interval_s`` of source time."""

    def __init__(self, notifiers: Sequence[Notifier], config: NotifyConfig) -> None:
        self.notifiers = list(notifiers)
        self.config = config
        self._last: dict[tuple[str, tuple[int, ...]], float] = {}
        self.sent = 0

    def notify(self, event: PetEvent) -> bool:
        if self.config.rules and event.rule not in self.config.rules:
            return False
        key = (event.rule, event.track_ids)
        last = self._last.get(key)
        if last is not None and event.timestamp_end_s - last < self.config.min_interval_s:
            return False
        self._last[key] = event.timestamp_end_s
        for notifier in self.notifiers:
            try:
                notifier.notify(event)
            except Exception as exc:  # a broken notifier must not stop monitoring
                log.warning("notifier %s failed: %s", type(notifier).__name__, exc)
        self.sent += 1
        return True


def build_notifier(config: NotifyConfig, *, log_stream: TextIO | None = None) -> NotificationPolicy:
    notifiers: list[Notifier] = []
    if log_stream is not None:
        notifiers.append(LogNotifier(log_stream))
    if config.desktop:
        notifiers.append(DesktopNotifier())
    if config.webhook_url:
        notifiers.append(
            WebhookNotifier(config.webhook_url, allow_remote=config.allow_remote_webhook)
        )
    return NotificationPolicy(notifiers, config)
