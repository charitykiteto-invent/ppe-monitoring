import time

from ppe_monitoring.camera_stream import LatestFrameCapture


class FakeCapture:
    def __init__(self):
        self.frames = iter(range(8))
        self.open = True

    def read(self):
        try:
            value = next(self.frames)
        except StopIteration:
            self.open = False
            return False, None
        time.sleep(.005)
        return True, value

    def get(self, _prop):
        return 0

    def isOpened(self):
        return self.open

    def release(self):
        self.open = False


def test_latest_frame_capture_drops_stale_frames():
    stream = LatestFrameCapture(FakeCapture(), read_timeout=.2)
    ok, first = stream.read()
    assert ok
    time.sleep(.025)
    ok, newest = stream.read()
    stream.release()
    assert ok
    assert newest > first + 1
