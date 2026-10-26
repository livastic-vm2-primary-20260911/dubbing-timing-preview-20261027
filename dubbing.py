"""Timing and PCM-stem mixing for explicitly supplied, reviewed translations.

This is not ASR, translation, speaker recognition, voice cloning or stem separation.
All processing is local; no API, wallet or customer credentials are needed.
"""
from __future__ import annotations
from array import array
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Sequence
import math
import re
import sys
import wave

class TimingError(ValueError):
    """The requested dub cannot meet the declared timing/consent contract."""

def number(value: object, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TimingError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise TimingError(f"{name} must be a finite number") from None
    if not result.is_finite():
        raise TimingError(f"{name} must be a finite number")
    return result

@dataclass(frozen=True)
class Segment:
    id: str
    speaker: str
    start: float
    window_end: float
    audio_duration: float
    tempo: float
    planned_end: float
    original: str
    translation: str

def plan(duration: object, rows: Sequence[Mapping], audio_lengths: Mapping[str, object],
         *, rights_confirmed: bool = False, voice_mode: str = "stock",
         speaker_consent: bool = False, max_tempo: object = 1.25,
         protected_terms: Sequence[str] = ()) -> list[Segment]:
    if rights_confirmed is not True:
        raise TimingError("Source rights must be explicitly confirmed")
    if voice_mode not in ("stock", "cloned"):
        raise TimingError("Unknown voice mode")
    if voice_mode == "cloned" and speaker_consent is not True:
        raise TimingError("A recreated speaker voice requires explicit speaker consent")
    total = number(duration, "duration")
    speed_limit = number(max_tempo, "max_tempo")
    if total <= 0 or total > 3600:
        raise TimingError("Duration must be in (0, 3600] seconds")
    if not Decimal(1) <= speed_limit <= Decimal("1.25"):
        raise TimingError("This preview limits speed-up to 1.25x")
    if not rows or len(rows) > 5000:
        raise TimingError("Supply 1..5000 reviewed utterances")
    if any(not isinstance(t, str) or not t.strip() for t in protected_terms):
        raise TimingError("Protected terms must be nonempty strings")
    result = []
    ids = set()
    prior_end = Decimal(0)
    for row in rows:
        if not isinstance(row, Mapping):
            raise TimingError("Each utterance must be an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier):
            raise TimingError("Invalid segment id")
        if identifier in ids:
            raise TimingError("Duplicate segment id")
        ids.add(identifier)
        speaker = row.get("speaker")
        if not isinstance(speaker, str) or not speaker.strip():
            raise TimingError("A speaker label is required")
        start, end = number(row.get("start"), "start"), number(row.get("end"), "end")
        if start < 0 or end <= start or end > total or start < prior_end:
            raise TimingError("Unsorted, overlapping or out-of-range source windows")
        prior_end = end
        original, translated = row.get("original"), row.get("translation")
        if any(not isinstance(t, str) or not t.strip() or "\n" in t or "\r" in t
               for t in (original, translated)):
            raise TimingError("Supply nonempty, single-line reviewed text")
        for term in protected_terms:
            pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
            if len(re.findall(pattern, original)) != len(re.findall(pattern, translated)):
                raise TimingError(f"Protected term changed: {term}")
        if identifier not in audio_lengths:
            raise TimingError(f"Missing synthesized audio: {identifier}")
        raw_duration = number(audio_lengths[identifier], "audio_duration")
        if raw_duration <= 0:
            raise TimingError("Empty synthesized audio")
        window = end - start
        tempo = max(Decimal(1), raw_duration / window)
        if tempo > speed_limit:
            raise TimingError(f"{identifier}: translated speech requires {tempo:.3f}x; "
                              "request a shorter faithful translation, do not truncate")
        result.append(Segment(identifier, speaker, float(start), float(end),
                              float(raw_duration), float(tempo),
                              float(start + raw_duration / tempo), original, translated))
    if set(audio_lengths) != ids:
        raise TimingError("Unmatched synthesized audio must not be silently ignored")
    return result

def pcm_read(path: Path, rate: int) -> array:
    with wave.open(str(path), "rb") as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getcomptype()) != (1, 2, rate, "NONE"):
            raise TimingError("Expected mono PCM16 at the declared sample rate")
        samples = array("h", wav.readframes(wav.getnframes()))
    if sys.byteorder != "little":
        samples.byteswap()
    return samples

def pcm_write(path: Path, samples: Sequence[int], rate: int) -> None:
    data = array("h", samples)
    if sys.byteorder != "little":
        data.byteswap()
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
        wav.writeframes(data.tobytes())

def mix_stem(background: Sequence[int], voices: Mapping[str, Sequence[int]],
             segments: Sequence[Segment], rate: int) -> tuple[array, list[dict]]:
    """Keep the supplied clean stem unchanged; reject overflow and overruns.

    The original full mix is NOT used: it would retain the original speaker.
    Production source separation is deliberately outside this preview.
    """
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise TimingError("Invalid sample rate")
    if set(voices) != {s.id for s in segments}:
        raise TimingError("Voice files and segments do not match")
    mixed = array("h", background)
    receipt = []
    previous_end = 0
    for s in segments:
        start = round(s.start * rate)
        stop = round(s.window_end * rate)
        speech = voices[s.id]
        if not speech or start < previous_end or stop > len(mixed):
            raise TimingError("Empty audio, overlap or background stem too short")
        actual_end = start + len(speech)
        if actual_end > stop:
            raise TimingError(f"{s.id}: measured audio exceeds the window; no truncation performed")
        for i, sample in enumerate(speech):
            value = int(mixed[start + i]) + int(sample)
            if not -32768 <= value <= 32767:
                raise TimingError("Mix would clip; lower a stem before rendering")
            mixed[start + i] = value
        previous_end = actual_end
        receipt.append({"id":s.id, "start_frame":start, "end_frame":actual_end,
                        "window_end_frame":stop, "actual_start":start/rate,
                        "actual_end":actual_end/rate,
                        "start_error_seconds":abs(start/rate - s.start),
                        "translation":s.translation})
    return mixed, receipt

def srt_timestamp(seconds: float) -> str:
    value = number(seconds, "subtitle timestamp")
    if value < 0:
        raise TimingError("Negative subtitle timestamp")
    ms = int((value * 1000).to_integral_value(rounding="ROUND_HALF_UP"))
    h, rem = divmod(ms, 3600000); m, rem = divmod(rem, 60000); sec, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"

def subtitles(receipts: Sequence[Mapping]) -> str:
    blocks = []
    prior_end = Decimal(0)
    for index, row in enumerate(receipts, 1):
        start, end = number(row["actual_start"], "start"), number(row["actual_end"], "end")
        text = row["translation"]
        if end <= start or start < prior_end or not isinstance(text,str) or not text.strip() or "\n" in text or "\r" in text:
            raise TimingError("Invalid subtitle cue")
        prior_end = end
        blocks.append(f"{index}\n{srt_timestamp(float(start))} --> {srt_timestamp(float(end))}\n{text}\n")
    return "\n".join(blocks)
