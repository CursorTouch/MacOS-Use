"""Event emitter service for the agent."""

from __future__ import annotations

from typing import Callable, Union

from macos_use.agent.events.views import AgentEvent, EventType
from macos_use.agent.events.subscriber import BaseEventSubscriber

EventSubscriber = Union[BaseEventSubscriber, Callable[[AgentEvent], None]]


class Event:
    """Manages event subscribers and dispatches events to them."""

    def __init__(self) -> None:
        self._subscribers: list[EventSubscriber] = []

    def add_subscriber(self, subscriber: EventSubscriber) -> None:
        self._subscribers.append(subscriber)

    def remove_subscriber(self, subscriber: EventSubscriber) -> None:
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def emit(self, event: AgentEvent) -> None:
        for subscriber in self._subscribers:
            try:
                if isinstance(subscriber, BaseEventSubscriber):
                    subscriber.invoke(event)
                else:
                    subscriber(event)
            except Exception:
                pass

    def close(self) -> None:
        for subscriber in self._subscribers:
            if isinstance(subscriber, BaseEventSubscriber):
                try:
                    subscriber.close()
                except Exception:
                    pass
        self._subscribers.clear()
