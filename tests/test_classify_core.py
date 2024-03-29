# -*- encoding: utf-8 -*-
"""Tests for the Nendo classify core plugin."""
import unittest

from nendo import Nendo, NendoConfig, NendoTrack

nd = Nendo(
    config=NendoConfig(
        log_level="DEBUG",
        max_threads=1,
        plugins=["nendo_plugin_classify_core"],
    ),
)


class ClassifyPluginTests(unittest.TestCase):
    def test_run_classify_plugin(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")
        nd.plugins.classify_core(track=track)
        self._assert(track)

    def test_run_process_classify_plugin(self):
        nd.library.reset(force=True)
        track = nd.library.add_track(file_path="tests/assets/test.wav")
        track.process("nendo_plugin_classify_core")
        self._assert(track)

    def _assert(self, track: NendoTrack):
        self.assertEqual(
            len(track.get_plugin_data(plugin_name="nendo_plugin_classify_core")),
            13,
        )

        key_data = track.get_plugin_data(
            plugin_name="nendo_plugin_classify_core",
            key="key",
        )
        self.assertEqual(len(key_data), 1)
        self.assertEqual(key_data[0].value, "B")

        scale_data = track.get_plugin_data(
            plugin_name="nendo_plugin_classify_core",
            key="scale",
        )
        self.assertEqual(len(scale_data), 1)
        self.assertEqual(scale_data[0].value, "minor")

        tempo_data = nd.library.filter_tracks(
            filters={"tempo": (80, 100)},
            plugin_names=["nendo_plugin_classify_core"],
        )
        self.assertEqual(len(tempo_data), 1)

        key_data = nd.library.filter_tracks(
            filters={"key": "B"},
            plugin_names=["nendo_plugin_classify_core"],
        )
        self.assertEqual(len(key_data), 1)

        instrument_data = track.get_plugin_data(
            plugin_name="nendo_plugin_classify_core",
            key="instruments",
        )
        self.assertIn("synthesizer", instrument_data[0].value)

        genre_data = track.get_plugin_data(
            plugin_name="nendo_plugin_classify_core",
            key="genres",
        )
        self.assertIn("Electronic", genre_data[0].value)

        mood_data = track.get_plugin_data(
            plugin_name="nendo_plugin_classify_core",
            key="moods",
        )
        self.assertIn("slow", mood_data[0].value)

        sfx_data = track.get_plugin_data(
            plugin_name="nendo_plugin_classify_core",
            key="sfx",
        )
        self.assertIn("Choir", sfx_data[0].value)


if __name__ == "__main__":
    unittest.main()
