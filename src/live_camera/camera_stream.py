"""
camera_stream.py
=================
Low-latency camera acquisition subsystem for QR Shield's live pipeline.

Problem this module solves
---------------------------
``cv2.VideoCapture`` internally queues frames (its own decode buffer,
plus — for network sources — OS/driver/socket buffers). If a consumer
reads slower than the producer, frames pile up in that buffer. Every
subsequent ``cap.read()`` call then returns the *oldest* undrained
frame rather than the newest one, so displayed/analysed video runs
seconds behind real time. This shows up as roughly zero-latency on a
local laptop webcam (near-zero buffering, driver hands frames straight
through) but as multi-second latency on an IP-camera / MJPEG stream,
where network + decode buffering is much deeper and one slow frame in
the analysis pipeline (Tamper Analysis, Risk Assessment, etc.) lets the
backlog grow without bound.

Fix: decouple *acquisition* from *consumption*. A dedicated background
thread does nothing but call ``cap.read()`` in a tight loop and
immediately overwrite a single shared "latest frame" slot — never a
queue, never a list. Whatever the analysis pipeline last stored is
always discarded the instant a newer frame arrives. Consumers call
:meth:`CameraStream.get_latest_frame` whenever they're ready and always
get the most recent frame available at that instant, no matter how far
behind their own processing has fallen. Frames are never queued for
later consumption, so there is nothing to fall behind on — old frames
are dropped, not delayed.

Supported sources
------------------
    * Laptop / built-in webcams   — int index, e.g. ``0``
    * USB webcams                 — int index, e.g. ``1``, ``2``, ...
    * IP camera HTTP/MJPEG streams — str URL,
      e.g. ``"http://192.168.1.5:8080/video"``
    * Local video files            — str path, e.g. ``"sample.mp4"``

Usage
-----
    stream = CameraStream(0)              # laptop/USB webcam
    stream = CameraStream("http://...")   # IP camera
    stream = CameraStream("clip.mp4")     # video file

    stream.start()
    frame = stream.get_latest_frame()     # None until the first frame arrives
    ...
    stream.stop()

``CameraStream`` also works as a context manager:

    with CameraStream(0) as stream:
        frame = stream.get_latest_frame()

Thread-safety
--------------
A single ``threading.Lock`` guards the shared frame slot. The reader
thread holds it only for the instant it takes to swap in a new frame;
:meth:`get_latest_frame` holds it only for the instant it takes to copy
the reference out. Neither side ever blocks the other for the duration
of a `cv2.VideoCapture.read()` call (which is the actually slow part).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np

logger = logging.getLogger("live_camera.camera_stream")

# A camera source is either an OpenCV device index (webcams) or a string
# (IP-camera URL or local video file path) — anything ``cv2.VideoCapture``
# itself accepts.
CameraSource = Union[int, str]

# Network/streaming URL prefixes. A string source that does NOT start with
# one of these is treated as a local video file (see _looks_like_file).
_STREAM_URL_PREFIXES = ("http://", "https://", "rtsp://", "rtsps://")


def _looks_like_file(source: CameraSource) -> bool:
    """Best-effort heuristic: is *source* a local video file path?

    Int sources (webcam indices) and string URLs (IP/RTSP streams) are
    never treated as files. Any other string is assumed to be a path.
    """
    if not isinstance(source, str):
        return False
    return not source.lower().startswith(_STREAM_URL_PREFIXES)


class CameraStream:
    """Threaded camera reader that always exposes only the newest frame.

    Parameters
    ----------
    source : int or str
        OpenCV camera index, IP-camera stream URL, or local video file
        path. Passed straight through to ``cv2.VideoCapture``.
    api_preference : int, optional
        Optional OpenCV backend hint (e.g. ``cv2.CAP_DSHOW``,
        ``cv2.CAP_FFMPEG``), forwarded to ``cv2.VideoCapture`` if given.
    reconnect_delay : float
        Seconds to wait between reconnect attempts after the source
        drops (default ``1.0``).
    max_reconnect_attempts : int, optional
        Cap on consecutive reconnect attempts before the reader thread
        gives up and stops (``None`` = retry forever, the default).
    loop_video_file : bool
        When *source* is a local video file and playback reaches EOF,
        rewind and keep playing from the start instead of stopping
        (default ``True``). Ignored for webcams/IP streams.
    """

    def __init__(
        self,
        source: CameraSource,
        api_preference: Optional[int] = None,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: Optional[int] = None,
        loop_video_file: bool = True,
    ) -> None:
        self.source = source
        self.api_preference = api_preference
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.loop_video_file = loop_video_file
        self._is_file = _looks_like_file(source)

        self._cap: Optional[cv2.VideoCapture] = None

        # Guards ONLY the shared frame slot below — never held across a
        # cap.read() call, so acquisition and consumption never block
        # each other for longer than a reference copy/swap.
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_count: int = 0

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False

        # Approximate source FPS, used only to pace local video file
        # playback so "latest frame" has real-time meaning. Webcams and
        # IP streams are read as fast as they can supply frames — there
        # is no separate pacing need there.
        self._source_fps: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "CameraStream":
        """Open the camera/stream and start the background reader thread.

        Safe to call once; calling again while already running is a
        no-op. Raises no exception on failure to open — check
        :meth:`is_opened` afterwards, matching the existing
        ``cap.isOpened()`` check pattern used by callers.
        """
        if self._running:
            return self

        self._cap = self._open_capture()
        self._connected = bool(self._cap and self._cap.isOpened())

        if self._connected:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._source_fps = fps if fps and fps > 0 else None

        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="CameraStreamReader",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the reader thread and release the camera resource.

        Idempotent — safe to call multiple times, and safe to call even
        if :meth:`start` was never called or failed.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Error releasing capture device.", exc_info=True)
            self._cap = None

        self._connected = False
        with self._lock:
            self._frame = None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recently captured frame, or ``None``.

        ``None`` is returned only before the very first frame has
        arrived (e.g. immediately after :meth:`start`) or once the
        stream has permanently given up after exhausting
        ``max_reconnect_attempts``. A copy is returned so callers may
        freely mutate it (existing code draws bounding boxes / overlays
        directly on the frame) without racing the reader thread's next
        write.
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def is_opened(self) -> bool:
        """True if the underlying capture is currently connected.

        Mirrors ``cv2.VideoCapture.isOpened()`` so call sites can keep
        the same "could not open camera" check they used before.
        """
        return self._connected

    def is_running(self) -> bool:
        """True while the background reader thread is active."""
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def frame_count(self) -> int:
        """Total number of frames successfully read since start()."""
        return self._frame_count

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        if self.api_preference is not None:
            cap = cv2.VideoCapture(self.source, self.api_preference)
        else:
            cap = cv2.VideoCapture(self.source)

        # Request the smallest internal buffer OpenCV/the backend will
        # honour. Not all backends respect this, which is precisely why
        # the "only keep the newest frame" thread below is still
        # necessary even after setting it — this is a best-effort
        # extra, not the fix itself.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:  # noqa: BLE001 — property not supported everywhere
            pass

        return cap

    def _reader_loop(self) -> None:
        """Background thread body: read continuously, keep only the newest.

        Runs until :meth:`stop` clears ``self._running``. On a read
        failure (disconnect, dropped Wi-Fi frame, camera unplugged)
        attempts to reconnect after ``reconnect_delay`` seconds, up to
        ``max_reconnect_attempts`` times (or forever if ``None``).
        Local video files reaching EOF are rewound if
        ``loop_video_file`` is True; otherwise the loop exits cleanly.
        """
        reconnect_attempts = 0
        frame_interval = (1.0 / self._source_fps) if self._source_fps else None
        last_read_time = 0.0

        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._try_reconnect(reconnect_attempts):
                    break
                reconnect_attempts = 0  # reset after a successful reconnect
                continue

            # Pace local file playback to roughly the source's own frame
            # rate so "latest frame" corresponds to real elapsed time
            # rather than racing through the file as fast as disk I/O
            # allows. Webcams/IP streams are never paced — they're read
            # as fast as the hardware/network supplies frames, which is
            # exactly the "always newest" behaviour we want from them.
            if frame_interval is not None:
                elapsed = time.monotonic() - last_read_time
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_read_time = time.monotonic()

            ok, frame = self._cap.read()

            if not ok or frame is None:
                if self._is_file:
                    if self.loop_video_file:
                        logger.info(
                            "Video file '%s' reached EOF; rewinding.",
                            self.source,
                        )
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.info(
                        "Video file '%s' reached EOF; stopping.", self.source
                    )
                    break

                logger.warning(
                    "Frame read failed for source %r; attempting reconnect.",
                    self.source,
                )
                self._connected = False
                try:
                    self._cap.release()
                except Exception:  # noqa: BLE001
                    pass
                self._cap = None
                if not self._try_reconnect(reconnect_attempts):
                    break
                reconnect_attempts += 1
                continue

            reconnect_attempts = 0
            with self._lock:
                self._frame = frame
                self._frame_count += 1

        self._running = False

    def _try_reconnect(self, attempts_so_far: int) -> bool:
        """Attempt one reconnect after ``reconnect_delay`` seconds.

        Returns ``True`` if a new capture was opened successfully (or
        if the loop should simply keep retrying), ``False`` if the
        retry budget is exhausted and the reader loop should give up.
        """
        if (
            self.max_reconnect_attempts is not None
            and attempts_so_far >= self.max_reconnect_attempts
        ):
            logger.error(
                "Giving up on source %r after %d reconnect attempts.",
                self.source,
                attempts_so_far,
            )
            self._connected = False
            return False

        time.sleep(self.reconnect_delay)
        if not self._running:
            return False

        logger.info("Reconnecting to source %r ...", self.source)
        self._cap = self._open_capture()
        self._connected = bool(self._cap and self._cap.isOpened())
        if self._connected:
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._source_fps = fps if fps and fps > 0 else None
            logger.info("Reconnected to source %r.", self.source)
        return True