"""Host-side tests for the sensorfwd frame parser and timestamp mapping."""

import struct

import collect_sensors as cs


def frame(*records):
    return struct.pack("<I", len(records)) + b"".join(cs.XYZ_RECORD.pack(*r) for r in records)


def test_record_size_matches_sensorfwd():
    # Measured on the watch: 28-byte frame for one sample.
    assert len(frame((1, 0.0, 0.0, 0.0))) == 28


def test_single_and_batched_frames():
    parser = cs.FrameParser()
    out = parser.feed(frame((10, 1.0, 2.0, 3.0)) + frame((20, 4.0, 5.0, 6.0), (30, 7.0, 8.0, 9.0)))
    assert [r[0] for r in out] == [10, 20, 30]
    assert out[2][1:] == (7.0, 8.0, 9.0)


def test_frame_split_across_reads():
    data = frame((10, 1.0, 2.0, 3.0), (20, 4.0, 5.0, 6.0))
    parser = cs.FrameParser()
    assert parser.feed(data[:3]) == []
    assert parser.feed(data[3:17]) == []
    assert [r[0] for r in parser.feed(data[17:])] == [10, 20]
    assert parser.buffer == b""


def test_real_watch_bytes():
    raw = bytes.fromhex("0100000020b70a5600000000506d7444508d5dc22db2d7424ce18200")
    ((ts, x, _y, _z),) = cs.FrameParser().feed(raw)
    assert ts == 0x560AB720
    assert round(x, 1) == 977.7


def test_sample_wall_time_goes_back_by_boottime_delay():
    # sample stamped 2.5 s (boottime) before "now"
    assert cs.sample_wall_time(97_500_000, 1000.0, 100.0) == 997.5


def test_real_watch_heart_rate_bytes():
    raw = bytes.fromhex("01000000f4988b6f00000000010000005e000000")
    ((ts, status, bpm),) = cs.FrameParser(cs.HEART_RATE_RECORD).feed(raw)
    assert (ts, status, bpm) == (0x6F8B98F4, 1, 94)
