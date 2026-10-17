"""Managed MemMachine adapter for immutable operational-memory records."""

from __future__ import annotations

import hashlib
import json
from types import TracebackType
from typing import Any, Literal, Self, TypeVar

import httpx2
from pydantic import BaseModel, ConfigDict, SecretStr

from memory_distillation.benchmark import TaskInput
from memory_distillation.operational_memory.models import ReviewedEpisode
from memory_distillation.versioning.models import (
    CandidateVersion,
    RetrievalPolicy,
    VersionView,
)
from memory_distillation.versioning.retrieval import retrieve_behavior_cards

RecordT = TypeVar("RecordT", bound=BaseModel)


class MemMachineError(RuntimeError):
    """Sanitized managed-memory transport or contract failure."""


class MemMachineAtomicityError(MemMachineError):
    """Raised when promotion requires an unsupported conditional pointer update."""


class _ProjectResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    project_id: str


class _AddMemoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    uid: str


class _AddMemoriesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    results: tuple[_AddMemoryResult, ...]


class _Episode(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    uid: str
    content: str
    metadata: dict[str, Any] | None = None
    filterable_metadata: dict[str, Any] | None = None

    @property
    def record_metadata(self) -> dict[str, Any]:
        return self.filterable_metadata or self.metadata or {}


class _ListContent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    episodic_memory: tuple[_Episode, ...] | None = None


class _ListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    content: _ListContent
    status: int = 0


class _AcceptedPointer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["accepted-pointer.v1"] = "accepted-pointer.v1"
    version_id: str
    version_content_hash: str
    revision: Literal[0] = 0


class MemMachineOperationalMemory:
    """Store typed immutable records in one managed MemMachine project.

    MemMachine's public API has no conditional update operation. Consequently,
    this adapter persists the bootstrap Accepted Version pointer but refuses a
    promotion update rather than presenting a non-atomic append as compare-and-set.
    """

    atomic_pointer_updates_supported = False

    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr,
        project_id: str,
        page_size: int = 100,
        timeout_seconds: float = 30,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        if not project_id:
            raise ValueError("project_id is required")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        self._project_id = project_id
        self._page_size = page_size
        self._client = httpx2.Client(
            base_url=base_url.rstrip("/") + "/",
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=httpx2.Timeout(timeout_seconds),
            transport=transport,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def initialize(self) -> None:
        """Idempotently create the project and its immutable bootstrap records."""

        response = self._post(
            "v2/projects",
            {
                "project_id": self._project_id,
                "description": "Memory-Distilled Agent Team operational memory",
            },
        )
        project = self._parse(_ProjectResponse, response)
        if project.project_id != self._project_id:
            raise MemMachineError("MemMachine returned a different project identity")
        bootstrap = VersionView(
            version_id="accepted-v000",
            cards=(),
            retrieval_policy=RetrievalPolicy(
                top_k=0,
                minimum_score=0,
                token_budget=0,
                calibration_snapshot_hash="0" * 64,
            ),
            content_hash="0" * 64,
        )
        self._freeze_view(bootstrap)
        self._put_immutable(
            record_type="accepted_pointer",
            record_id="accepted",
            record=_AcceptedPointer(
                version_id=bootstrap.version_id,
                version_content_hash=bootstrap.content_hash,
            ),
        )

    def append_episode(self, episode: ReviewedEpisode) -> None:
        validated = ReviewedEpisode.model_validate(episode.model_dump(mode="json"))
        self._put_immutable(
            record_type="reviewed_episode",
            record_id=validated.episode_id,
            record=validated,
        )

    def get_episode(self, episode_id: str) -> ReviewedEpisode | None:
        return self._get_record("reviewed_episode", episode_id, ReviewedEpisode)

    def snapshot_eligible_episodes(self) -> tuple[ReviewedEpisode, ...]:
        episodes = self._load_records("reviewed_episode", ReviewedEpisode)
        return tuple(
            episode
            for episode in sorted(episodes, key=lambda item: item.episode_id)
            if episode.eligible_for_consolidation
        )

    def freeze_candidate(self, candidate: CandidateVersion) -> VersionView:
        validated = CandidateVersion.model_validate(candidate.model_dump(mode="json"))
        _validate_answer_free_cards(validated)
        self._put_immutable(
            record_type="candidate_version",
            record_id=validated.version_id,
            record=validated,
        )
        view = VersionView(
            version_id=validated.version_id,
            cards=validated.cards,
            retrieval_policy=validated.retrieval_policy,
            content_hash=validated.content_hash,
        )
        self._freeze_view(view)
        return view

    def get_candidate(self, version_id: str) -> CandidateVersion:
        candidate = self._get_record("candidate_version", version_id, CandidateVersion)
        if candidate is None:
            raise KeyError(version_id)
        return candidate

    def _freeze_view(self, view: VersionView) -> None:
        validated = VersionView.model_validate(view.model_dump(mode="json"))
        card_ids = [card.card_id for card in validated.cards]
        if len(card_ids) != len(set(card_ids)):
            raise ValueError("version contains duplicate Behavior Card identifiers")
        self._put_immutable(
            record_type="version_view",
            record_id=validated.version_id,
            record=validated,
        )

    def get_view(self, version_id: str) -> VersionView:
        view = self._get_record("version_view", version_id, VersionView)
        if view is None:
            raise KeyError(version_id)
        return view

    def retrieve(self, *, version_id: str, task: TaskInput) -> tuple[str, ...]:
        """Retrieve cards only from the exact immutable version requested."""

        return retrieve_behavior_cards(self.get_view(version_id), task)

    @property
    def accepted_version_id(self) -> str:
        pointer = self._get_record("accepted_pointer", "accepted", _AcceptedPointer)
        if pointer is None:
            raise MemMachineError("Accepted Version pointer is not initialized")
        view = self.get_view(pointer.version_id)
        if view.content_hash != pointer.version_content_hash:
            raise MemMachineError("Accepted Version pointer content hash does not match")
        return pointer.version_id

    def compare_and_set_accepted(self, *, expected: str, replacement: str) -> bool:
        if self.accepted_version_id != expected:
            return False
        self.get_view(replacement)
        raise MemMachineAtomicityError(
            "MemMachine public API has no conditional update primitive; "
            "refusing a non-atomic Accepted Version pointer change"
        )

    def _put_immutable(
        self,
        *,
        record_type: str,
        record_id: str,
        record: BaseModel,
    ) -> None:
        content = _canonical_json(record)
        existing = self._matching_episodes(record_type, record_id)
        if existing:
            self._require_one_identical(existing, content)
            return
        response = self._post(
            "v2/memories",
            {
                "project_id": self._project_id,
                "messages": [
                    {
                        "content": content,
                        "producer": "memory-distillation",
                        "produced_for": "operational-memory",
                        "role": "system",
                        "metadata": {
                            "md_record_type": record_type,
                            "md_record_id": record_id,
                            "md_content_hash": _sha256(content),
                        },
                    }
                ],
                "types": ["episodic"],
            },
        )
        added = self._parse(_AddMemoriesResponse, response)
        if len(added.results) != 1 or not added.results[0].uid:
            raise MemMachineError("MemMachine did not acknowledge exactly one record")
        self._require_one_identical(self._matching_episodes(record_type, record_id), content)

    def _get_record(
        self,
        record_type: str,
        record_id: str,
        model: type[RecordT],
    ) -> RecordT | None:
        matches = self._matching_episodes(record_type, record_id)
        if not matches:
            return None
        content = self._require_one_identical(matches)
        return self._parse_content(model, content)

    def _load_records(self, record_type: str, model: type[RecordT]) -> tuple[RecordT, ...]:
        records: list[RecordT] = []
        by_id: dict[str, list[_Episode]] = {}
        for episode in self._list_episodes(record_type):
            record_id = str(episode.record_metadata.get("md_record_id", ""))
            if not record_id:
                raise MemMachineError("MemMachine record is missing its stable identity")
            by_id.setdefault(record_id, []).append(episode)
        for record_id in sorted(by_id):
            content = self._require_one_identical(by_id[record_id])
            records.append(self._parse_content(model, content))
        return tuple(records)

    def _matching_episodes(self, record_type: str, record_id: str) -> tuple[_Episode, ...]:
        return tuple(
            episode
            for episode in self._list_episodes(record_type)
            if episode.record_metadata.get("md_record_id") == record_id
        )

    def _list_episodes(self, record_type: str) -> tuple[_Episode, ...]:
        episodes: list[_Episode] = []
        for page_num in range(1_000):
            response = self._post(
                "v2/memories/list",
                {
                    "project_id": self._project_id,
                    "page_size": self._page_size,
                    "page_num": page_num,
                    "filter": f"metadata.md_record_type={record_type}",
                    "type": "episodic",
                },
            )
            page = self._parse(_ListResponse, response)
            if page.status != 0:
                raise MemMachineError("MemMachine list operation returned a failure status")
            items = page.content.episodic_memory or ()
            episodes.extend(items)
            if len(items) < self._page_size:
                return tuple(episodes)
        raise MemMachineError("MemMachine pagination exceeded the safety limit")

    def _require_one_identical(
        self,
        episodes: tuple[_Episode, ...] | list[_Episode],
        expected_content: str | None = None,
    ) -> str:
        if len(episodes) != 1:
            raise MemMachineError("immutable record identity is duplicated")
        episode = episodes[0]
        content_hash = episode.record_metadata.get("md_content_hash")
        if content_hash != _sha256(episode.content):
            raise MemMachineError("immutable record content hash does not match")
        if expected_content is not None and episode.content != expected_content:
            raise ValueError("record id already stores different immutable content")
        return episode.content

    def _post(self, path: str, payload: dict[str, Any]) -> httpx2.Response:
        try:
            response = self._client.post(path, json=payload)
        except httpx2.HTTPError as error:
            raise MemMachineError(
                f"MemMachine transport failed ({type(error).__name__})"
            ) from error
        if not response.is_success:
            raise MemMachineError(f"MemMachine request failed (status={response.status_code})")
        return response

    @staticmethod
    def _parse(model: type[RecordT], response: httpx2.Response) -> RecordT:
        try:
            return model.model_validate_json(response.content)
        except ValueError as error:
            raise MemMachineError("MemMachine response failed schema validation") from error

    @staticmethod
    def _parse_content(model: type[RecordT], content: str) -> RecordT:
        try:
            return model.model_validate_json(content)
        except ValueError as error:
            raise MemMachineError("MemMachine record failed schema validation") from error


def _canonical_json(record: BaseModel) -> str:
    return json.dumps(record.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _validate_answer_free_cards(candidate: CandidateVersion) -> None:
    content = " ".join(
        f"{card.trigger} {card.instruction}" for card in candidate.cards
    ).casefold()
    markers = ("correct answer", "ground truth", "hidden label", "chartqa:", "task_id")
    if any(marker in content for marker in markers):
        raise ValueError("Candidate Version contains answer or benchmark leakage")
