"""Media admission checks without FFmpeg, GPU, or user media fixtures."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from services.media.probe import MediaError, probe


class ProbeTests(unittest.TestCase):
    def inspect(self, timestamps, **stream_changes):
        stream = {"codec_type": "video", "width": 320, "height": 180,
                  "r_frame_rate": "4/1", "avg_frame_rate": "4/1", "duration": "1",
                  "time_base": "1/100000", "sample_aspect_ratio": "1:1", "start_time": "0"}
        stream.update(stream_changes)
        info = {"streams": [stream], "format": {"duration": "1"}}
        frames = {"frames": [{"best_effort_timestamp_time": str(t)} for t in timestamps]}
        with patch("services.media.probe.ffprobe", side_effect=[deepcopy(info), frames]):
            return probe("fixture.mp4")

    def test_variable_timestamps_are_accepted_with_average_frame_clock(self):
        meta = self.inspect([0, .03, .55, .8], r_frame_rate="120/1")
        self.assertEqual((meta["fps_num"], meta["fps_den"]), (4, 1))
        self.assertEqual(meta["frame_count"], 4)
        self.assertEqual(meta["duration_seconds"], 1)
        self.assertEqual(meta["validation_profile"], "cpu-vfr-v1")

    def test_small_timestamp_jitter_does_not_reject_decodable_video(self):
        meta = self.inspect([0, .24997, .50001, .75003])
        self.assertEqual(meta["frame_count"], 4)

    def test_missing_average_uses_decoded_count_and_duration(self):
        meta = self.inspect([0, .03, .55, .8], avg_frame_rate="0/0", r_frame_rate="120/1")
        self.assertEqual((meta["fps_num"], meta["fps_den"]), (4, 1))

    def test_cfr_metadata_remains_compatible(self):
        meta = self.inspect([0, .25, .5, .75])
        self.assertEqual(meta["validation_profile"], "cpu-cfr-v1")
        self.assertEqual(meta["duration_seconds"], 1)

    def test_rotation_and_missing_frames_remain_explicit_errors(self):
        with self.assertRaisesRegex(MediaError, "rotation"):
            self.inspect([0, .25], tags={"rotate": "90"})
        with self.assertRaisesRegex(MediaError, "no decoded frames"):
            self.inspect([])


if __name__ == "__main__":
    unittest.main()
