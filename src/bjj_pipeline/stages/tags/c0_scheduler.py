from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional


class CadenceState(str, Enum):
    SEEKING = "SEEKING"
    VERIFIED = "VERIFIED"
    RAMP_UP = "RAMP_UP"


@dataclass(frozen=True)
class Candidate:
    """Join-first scheduling candidate.

    Carries join keys + optional gating metadata. In M2, gating may cause
    immediate skip decisions before cadence evaluation.
    """

    frame_index: int
    timestamp_ms: int
    detection_id: str
    tracklet_id: str

    # Optional ROI + gating fields
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    det_conf: float = 1.0
    on_mat: Optional[bool] = None

    # Optional metric geometry (from Stage A overlays when available)
    x_m: Optional[float] = None
    y_m: Optional[float] = None

    scannable: bool = True
    gate_reason: str = "ok"
    gate_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackletScheduleState:
    state: CadenceState = CadenceState.SEEKING
    last_attempt_frame: Optional[int] = None
    last_success_frame: Optional[int] = None
    ramp_until_frame: Optional[int] = None
    last_trigger_frame: Optional[int] = None


@dataclass(frozen=True)
class Decision:
    candidate: Candidate
    decision: Literal["attempt", "skip"]
    # cadence_due / cadence_not_due / skip_* (gating)
    reason: str
    state_before: str
    state_after: str
    gate_meta: Optional[Dict[str, Any]] = None


class C0Scheduler:
    """Deterministic per-tracklet cadence scheduler (Milestone 2).

    M2 responsibilities:
    - Maintain SEEKING / VERIFIED / RAMP_UP state per tracklet_id.
    - Emit attempt-vs-skip decisions based on frame cadence only.

    Explicit non-goals in M2:
    - no scannability gating (blur/contrast/tag-fit) -> M3
    - no triggers (overlap/velocity jumps) -> M4
    - no decoding -> M5
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        k_seek: int = 1,
        k_verify: int = 30,
        n_ramp: int = 60,
        trigger_cooldown_frames: int = 15,
        extend_ramp_on_retrigger: bool = True,
    ):
        if k_seek <= 0:
            raise ValueError("k_seek must be > 0")
        if k_verify <= 0:
            raise ValueError("k_verify must be > 0")
        if n_ramp <= 0:
            raise ValueError("n_ramp must be > 0")

        self.enabled = bool(enabled)
        self.k_seek = int(k_seek)
        self.k_verify = int(k_verify)
        self.n_ramp = int(n_ramp)

        # Milestone 4 trigger controls
        self.trigger_cooldown_frames = int(trigger_cooldown_frames)
        self.extend_ramp_on_retrigger = bool(extend_ramp_on_retrigger)

        self._state: Dict[str, TrackletScheduleState] = {}

    def configure_triggers(self, *, trigger_cooldown_frames: int = 15, extend_ramp_on_retrigger: bool = True) -> None:
        self.trigger_cooldown_frames = int(trigger_cooldown_frames)
        self.extend_ramp_on_retrigger = bool(extend_ramp_on_retrigger)

    # -------------------------
    # Public hooks
    # -------------------------

    def on_decode_success(self, tracklet_id: str, frame_index: int) -> None:
        """Transition a tracklet to VERIFIED after a successful decode.

        Not used in M2 runtime; used for unit testing + later milestones.
        """
        st = self._state.get(tracklet_id)
        if st is None:
            st = TrackletScheduleState()
            self._state[tracklet_id] = st
        st.state = CadenceState.VERIFIED
        st.last_success_frame = int(frame_index)
        st.ramp_until_frame = None

    def apply_triggers(
        self,
        *,
        frame_index: int,
        timestamp_ms: int,
        trigger_events: List[Dict[str, Any]],
    ) -> None:
        _ = int(timestamp_ms)

        for ev in trigger_events:
            tid = str(ev.get("tracklet_id", ""))
            if not tid:
                continue
            st = self._state.get(tid)
            if st is None:
                st = TrackletScheduleState()
                self._state[tid] = st

            if st.last_trigger_frame is not None and self.trigger_cooldown_frames > 0:
                if (int(frame_index) - int(st.last_trigger_frame)) < int(self.trigger_cooldown_frames):
                    continue

            st.last_trigger_frame = int(frame_index)

            if st.state == CadenceState.VERIFIED:
                st.state = CadenceState.RAMP_UP
                st.ramp_until_frame = int(frame_index) + int(self.n_ramp)
            elif st.state == CadenceState.RAMP_UP and self.extend_ramp_on_retrigger:
                st.ramp_until_frame = int(frame_index) + int(self.n_ramp)

    def step(self, frame_index: int, timestamp_ms: int, candidates: List[Candidate]) -> List[Decision]:
        """Compute cadence-only attempt/skip decisions for the current frame."""

        fi = int(frame_index)
        # timestamp_ms is carried through for audit join keys; unused for cadence in M2.
        _ = int(timestamp_ms)

        decisions: List[Decision] = []

        for c in candidates:
            tid = c.tracklet_id
            st = self._state.get(tid)
            if st is None:
                st = TrackletScheduleState()
                self._state[tid] = st

            before = st.state.value

            # RAMP_UP expiry logic (present in M2; trigger to enter RAMP_UP is M4)
            if (
                st.state == CadenceState.RAMP_UP
                and st.ramp_until_frame is not None
                and fi >= int(st.ramp_until_frame)
            ):
                st.state = CadenceState.VERIFIED
                st.ramp_until_frame = None

            # Cadence period by state
            if st.state in (CadenceState.SEEKING, CadenceState.RAMP_UP):
                period = self.k_seek
            else:
                period = self.k_verify

            if not self.enabled:
                # Keep reasons minimal in M2 (no gating yet).
                decisions.append(
                    Decision(
                        candidate=c,
                        decision="skip",
                        reason="cadence_not_due",
                        state_before=before,
                        state_after=st.state.value,
                    )
                )
                continue

            if not bool(getattr(c, "scannable", True)):
                decisions.append(
                    Decision(
                        candidate=c,
                        decision="skip",
                        reason=str(getattr(c, "gate_reason", "skip_not_scannable")),
                        gate_meta=dict(getattr(c, "gate_meta", {}) or {}),
                        state_before=before,
                        state_after=st.state.value,
                    )
                )
                continue

            last = st.last_attempt_frame
            due = (last is None) or ((fi - int(last)) >= int(period))

            if due:
                st.last_attempt_frame = fi
                decisions.append(
                    Decision(
                        candidate=c,
                        decision="attempt",
                        reason="cadence_due",
                        state_before=before,
                        state_after=st.state.value,
                    )
                )
            else:
                decisions.append(
                    Decision(
                        candidate=c,
                        decision="skip",
                        reason="cadence_not_due",
                        state_before=before,
                        state_after=st.state.value,
                    )
                )

        return decisions
