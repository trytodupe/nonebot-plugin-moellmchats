from dataclasses import dataclass
import itertools
import time
from typing import Any, Callable, Optional

PENDING_TTL_SECONDS = 120


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: int
    user_id: str
    user_name: str
    scope: str
    prompt_preview: str
    started_at: float

    def elapsed_seconds(self, now: float) -> int:
        return max(0, int(now - self.started_at))


@dataclass
class PendingRequest:
    user_id: str
    scope: str
    prompt_preview: str
    created_at: float
    payload: Any
    confirmation_message_id: Optional[int] = None


@dataclass(frozen=True)
class BeginResult:
    request: Optional[RequestSnapshot] = None
    active_request: Optional[RequestSnapshot] = None
    pending_exists: bool = False

    @property
    def started(self) -> bool:
        return self.request is not None


class RequestRegistry:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._ids = itertools.count(1)
        self._active: dict[int, RequestSnapshot] = {}
        self._pending: dict[str, PendingRequest] = {}

    def begin(
        self,
        *,
        user_id: str,
        user_name: str,
        scope: str,
        prompt_preview: str,
        allow_concurrent: bool = False,
    ) -> BeginResult:
        self._discard_expired_pending()
        active = self._first_active_for_user(user_id)
        if active is not None and not allow_concurrent:
            return BeginResult(active_request=active, pending_exists=user_id in self._pending)

        request = RequestSnapshot(
            request_id=next(self._ids),
            user_id=user_id,
            user_name=user_name,
            scope=scope,
            prompt_preview=prompt_preview,
            started_at=self._clock(),
        )
        self._active[request.request_id] = request
        return BeginResult(request=request)

    def finish(self, request_id: int) -> None:
        self._active.pop(request_id, None)

    def add_pending(
        self,
        *,
        user_id: str,
        scope: str,
        prompt_preview: str,
        payload: Any,
    ) -> Optional[PendingRequest]:
        self._discard_expired_pending()
        if user_id in self._pending:
            return None
        pending = PendingRequest(
            user_id=user_id,
            scope=scope,
            prompt_preview=prompt_preview,
            created_at=self._clock(),
            payload=payload,
        )
        self._pending[user_id] = pending
        return pending

    def bind_confirmation(self, user_id: str, message_id: int) -> None:
        pending = self._pending.get(user_id)
        if pending is not None:
            pending.confirmation_message_id = message_id

    def remove_pending(self, user_id: str) -> None:
        self._pending.pop(user_id, None)

    def has_valid_confirmation(self, *, user_id: str, scope: str, reply_message_id: int) -> bool:
        self._discard_expired_pending()
        pending = self._pending.get(user_id)
        return bool(pending is not None and pending.scope == scope and pending.confirmation_message_id == reply_message_id)

    def take_confirmed(self, *, user_id: str, scope: str, reply_message_id: int) -> Optional[PendingRequest]:
        if not self.has_valid_confirmation(
            user_id=user_id,
            scope=scope,
            reply_message_id=reply_message_id,
        ):
            return None
        return self._pending.pop(user_id)

    def snapshot(self, scope: str) -> list[RequestSnapshot]:
        return sorted(
            (request for request in self._active.values() if request.scope == scope),
            key=lambda request: request.started_at,
        )

    def now(self) -> float:
        return self._clock()

    def _first_active_for_user(self, user_id: str) -> Optional[RequestSnapshot]:
        return next((request for request in self._active.values() if request.user_id == user_id), None)

    def _discard_expired_pending(self) -> None:
        now = self._clock()
        expired = [user_id for user_id, pending in self._pending.items() if now - pending.created_at >= PENDING_TTL_SECONDS]
        for user_id in expired:
            self._pending.pop(user_id, None)


request_registry = RequestRegistry()
