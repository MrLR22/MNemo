# modules/event_record.py
# Immutable value object representing one bound event.
# Serialisable to dict for ChromaDB / SQLite downstream.

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class EventRecord:
    action: str
    object_label: str
    location_room: str
    location_surface: str
    timestamp_start: float
    timestamp_end: float
    confidence: float
    flow_direction: str
    video_source: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.timestamp_start:.2f}s–{self.timestamp_end:.2f}s] "
            f"{self.action} | obj={self.object_label} | "
            f"room={self.location_room} | surface={self.location_surface} | "
            f"flow={self.flow_direction} | conf={self.confidence:.3f}"
        )
