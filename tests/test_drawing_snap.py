"""Magnet snapping of drawing points to candle open/high/low/close (#197)."""

from tradebot.gui.drawings import snap_point, snap_time

XS = [0.0, 10.0, 20.0]
OHLC = ([10, 20, 30], [15, 25, 35], [5, 15, 25], [12, 22, 32])


def test_nearest_value_wins():
    # candle 1 (x=10): O20 H25 L15 C22; y=21.4 is 1.4 from O, 0.6 from C
    x, y, which = snap_point(10, 21.4, XS, OHLC, 1, 1, radius=5)
    assert (x, y, which) == (10.0, 22.0, "close")


def test_radius_edge_and_outside():
    assert snap_point(10, 27, XS, OHLC, 1, 1, radius=2) == (10.0, 25.0, "high")  # exactly on the edge
    assert snap_point(10, 27.01, XS, OHLC, 1, 1, radius=2) is None


def test_pixel_scale_and_x_distance():
    # 4 view units off in x but 0.1 px per unit: still inside
    assert snap_point(14, 25, XS, OHLC, 0.1, 1, radius=2)[:2] == (10.0, 25.0)
    assert snap_point(14, 25, XS, OHLC, 1, 1, radius=2) is None


def test_picks_nearest_candle_in_x():
    assert snap_point(18, 31, XS, OHLC, 0.1, 1, radius=3)[0] == 20.0


def test_off_and_empty():
    assert snap_point(10, 22, XS, OHLC, 1, 1, enabled=False) is None
    assert snap_point(10, 22, [], ([], [], [], []), 1, 1) is None
    assert snap_time(10, [], 1) is None


def test_snap_time():
    assert snap_time(11, XS, 1, radius=2) == 10.0
    assert snap_time(13, XS, 1, radius=2) is None
    assert snap_time(11, XS, 1, radius=2, enabled=False) is None
