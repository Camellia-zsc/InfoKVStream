from __future__ import annotations
import math
import re
from typing import Iterator, List, Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image

def resize_frame(
    frame_bgr,
    max_side: int,
) -> Image.Image:
    height, width = frame_bgr.shape[:2]

    scale = min(
        max_side / max(height, width),
        1.0,
    )

    if scale < 1.0:
        new_width = max(
            1,
            int(round(width * scale)),
        )
        new_height = max(
            1,
            int(round(height * scale)),
        )
        frame_bgr = cv2.resize(
            frame_bgr,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )
    return Image.fromarray(frame_rgb)

FrameChunk = Tuple[
    List[Image.Image],
    bool,
    float,
]

def iter_chunks_until_query(
    video_path: str,
    query_time: float,
    sample_fps: float,
    frames_per_turn: int,
    max_side: int,
    start_time: float = 0.0,
) -> Iterator[FrameChunk]:
    """
    Backward / Realtime:
    stream frames only up to the query time.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )
    if source_fps <= 0:
        source_fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    start_frame_index = max(0, int(round(start_time * source_fps)))
    query_frame_index = start_frame_index + max(
        0,
        int(math.ceil(query_time * source_fps) - 1),
    )

    if total_frames > 0:
        query_frame_index = min(
            query_frame_index,
            total_frames - 1,
        )

    frame_step = max(
        1,
        int(round(source_fps / sample_fps)),
    )

    buffer: List[Image.Image] = []
    buffer_time = 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    frame_index = start_frame_index
    last_frame_bgr = None
    last_frame_index = -1
    last_sampled_index = -1

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            last_frame_bgr = frame_bgr
            last_frame_index = frame_index

            if frame_index < query_frame_index:
                if (frame_index - start_frame_index) % frame_step == 0:
                    buffer.append(
                        resize_frame(
                            frame_bgr,
                            max_side,
                        )
                    )
                    buffer_time = (
                        (frame_index - start_frame_index) / source_fps
                    )
                    last_sampled_index = frame_index

                    if len(buffer) >= frames_per_turn:
                        yield (
                            buffer,
                            False,
                            buffer_time,
                        )
                        buffer = []
            else:
                if buffer:
                    yield (
                        buffer,
                        False,
                        buffer_time,
                    )
                    buffer = []

                query_image = resize_frame(
                    frame_bgr,
                    max_side,
                )
                query_frame_time = (
                    (frame_index - start_frame_index) / source_fps
                )

                yield (
                    [query_image],
                    True,
                    query_frame_time,
                )
                return

            frame_index += 1

        if last_frame_bgr is None:
            raise RuntimeError(
                f"Video has no valid frames: {video_path}"
            )

        if (
            buffer
            and last_sampled_index == last_frame_index
        ):
            buffer.pop()

        if buffer:
            yield (
                buffer,
                False,
                buffer_time,
            )

        yield (
            [
                resize_frame(
                    last_frame_bgr,
                    max_side,
                )
            ],
            True,
            max(
                0.0,
                (last_frame_index - start_frame_index) / source_fps,
            ),
        )

    finally:
        cap.release()

def iter_frame_chunks(
    video_path: str,
    sample_fps: float,
    frames_per_turn: int,
    max_side: int,
    max_frames: Optional[int] = None,
) -> Iterator[List[Image.Image]]:
    """
    Forward:
    sample frames through the whole already-cut evaluation clip.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    source_fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )
    if source_fps <= 0:
        source_fps = 30.0

    frame_step = max(
        1,
        int(round(source_fps / sample_fps)),
    )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = None
    if max_frames is not None and max_frames > 0 and total_frames > 0:
        target_indices = set(
            np.linspace(
                0,
                total_frames - 1,
                max_frames,
            ).round().astype(int).tolist()
        )

    buffer: List[Image.Image] = []
    frame_index = 0

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break

            if (
                target_indices is not None
                and frame_index in target_indices
            ) or (
                target_indices is None
                and frame_index % frame_step == 0
            ):
                buffer.append(
                    resize_frame(
                        frame_bgr,
                        max_side,
                    )
                )

                if len(buffer) >= frames_per_turn:
                    yield buffer
                    buffer = []

            frame_index += 1

        if buffer:
            yield buffer

    finally:
        cap.release()

def mark_last_chunk(
    chunks: Iterator[List[Image.Image]],
) -> Iterator[
    Tuple[List[Image.Image], bool]
]:
    iterator = iter(chunks)

    try:
        current = next(iterator)
    except StopIteration:
        return

    for following in iterator:
        yield current, False
        current = following

    yield current, True

def clean_reply(
    reply: Optional[str],
) -> str:
    if reply is None:
        return ""

    reply = reply.strip()

    prefix = "I must reply."
    if reply.startswith(prefix):
        reply = reply[len(prefix):].strip()

    return reply

def is_no_reply(
    reply: Optional[str],
) -> bool:
    if reply is None:
        return True

    normalized = reply.strip().upper()

    return (
        normalized == ""
        or normalized == "NO"
        or normalized.startswith("NO REPLY")
    )

def extract_option(
    reply: str,
) -> str:
    normalized = reply.strip().upper()

    start_match = re.match(
        r"^\s*([A-E])(?:\s|[.):,;]|$)",
        normalized,
    )
    if start_match:
        return start_match.group(1)

    option_match = re.search(
        r"(?<![A-Z])([A-E])(?![A-Z])",
        normalized,
    )
    if option_match:
        return option_match.group(1)

    return reply.strip()

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}

def normalize_forward_reply(
    reply: Optional[str],
    task: str,
) -> str:
    reply = clean_reply(reply)

    if task == "REC":
        number_match = re.search(
            r"\b\d+\b",
            reply,
        )
        if number_match:
            return number_match.group(0)

        normalized = reply.lower()
        for word, number in NUMBER_WORDS.items():
            if re.search(
                rf"\b{word}\b",
                normalized,
            ):
                return number

        return reply

    if task in {"SSR", "CRR"}:
        if is_no_reply(reply):
            return "No"

        normalized = reply.lower()

        yes_match = re.search(
            r"\byes\b",
            normalized,
        )
        no_match = re.search(
            r"\bno\b",
            normalized,
        )

        if yes_match and not no_match:
            return "Yes"
        if no_match and not yes_match:
            return "No"
        if normalized.startswith("yes"):
            return "Yes"
        if normalized.startswith("no"):
            return "No"

        return reply

    return reply
