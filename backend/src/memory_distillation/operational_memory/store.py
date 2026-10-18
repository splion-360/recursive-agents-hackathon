"""Deterministic Operational Memory adapter used before live MemMachine wiring."""

from memory_distillation.operational_memory.models import ReviewedEpisode


class InMemoryOperationalMemory:
    """Append immutable episodes with idempotent same-content retries."""

    def __init__(self) -> None:
        self._episodes: dict[str, ReviewedEpisode] = {}

    def append_episode(self, episode: ReviewedEpisode) -> None:
        episode = ReviewedEpisode.model_validate(episode.model_dump(mode="json"))
        existing = self._episodes.get(episode.episode_id)
        if existing is not None and existing != episode:
            raise ValueError("episode id already stores different content")
        self._episodes[episode.episode_id] = episode

    def get_episode(self, episode_id: str) -> ReviewedEpisode | None:
        return self._episodes.get(episode_id)

    def snapshot_collection_episodes(self) -> tuple[ReviewedEpisode, ...]:
        """Return every immutable Collection episode, including ineligible evidence."""

        return tuple(
            episode
            for _, episode in sorted(self._episodes.items())
            if episode.provenance.purpose == "collection"
        )

    def snapshot_eligible_episodes(self) -> tuple[ReviewedEpisode, ...]:
        return tuple(
            episode
            for _, episode in sorted(self._episodes.items())
            if episode.eligible_for_consolidation
        )
