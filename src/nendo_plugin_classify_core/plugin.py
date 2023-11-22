# -*- encoding: utf-8 -*-
"""Music Information Retrieval Classification Plugin for the Nendo framework."""
from functools import lru_cache

import essentia.standard as es
import numpy as np
from nendo import Nendo, NendoAnalysisPlugin, NendoConfig, NendoTrack


@lru_cache
def signal(file_path: str) -> dict:
    """Load the track using essentia and save signal and sr.

    Caches the result to avoid reloading the same file multiple times.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        dict: Dictionary containing the signal and sample rate.

    """
    signal = es.MonoLoader(filename=file_path)()
    sr = 44100  # Essentia works with this sample rate
    return {"signal": signal, "sr": sr}


class NendoClassifyCore(NendoAnalysisPlugin):
    """A nendo plugin for music information retrieval and classification.

    Based on the essentia library. https://essentia.upf.edu/

    Examples:
        ```python
        from nendo import Nendo, NendoConfig

        nendo = Nendo(config=NendoConfig(plugins=["nendo_plugin_classify_core"]))
        track = nendo.library.add_track_from_file(
            file_path="path/to/file.wav",
        )
        track = nendo.plugins.classify_core(track=track)

        data = track.get_plugin_data(plugin_name="nendo_plugin_classify_core")
        print(data)

        tracks_with_filtered_tempo = nd.library.filter_tracks(
            filters={"tempo": (170, 180)},
            plugin_names=["nendo_plugin_classify_core"],
        )

        assert len(tracks_with_filtered_tempo) == 1
        ```
    """

    nendo_instance: Nendo
    config: NendoConfig = None

    @NendoAnalysisPlugin.plugin_data
    def loudness(self, track: NendoTrack) -> dict:
        """Compute the loudness of the given track."""
        loudness = es.Loudness()(signal(track.resource.src)["signal"])
        return {"loudness": loudness}

    @NendoAnalysisPlugin.plugin_data
    def duration(self, track: NendoTrack) -> dict:
        """Compute the duration of the given track."""
        duration = es.Duration()(signal(track.resource.src)["signal"])
        return {"duration": duration}

    @NendoAnalysisPlugin.plugin_data
    def frequency(self, track: NendoTrack) -> dict:
        """Compute average frequency of the given track.

        Uses the YIN algorithm to compute the pitch.
        If essentia runs into an error, returns 0.
        """
        try:
            audio_signal = signal(track.resource.src)["signal"]

            # Check if the length of the audio signal is even, if not, trim it
            # This is required for the YIN algorithm to work
            if len(audio_signal) % 2 != 0:
                audio_signal = audio_signal[:-1]  # Excluding the last sample

            pitch, _ = es.PitchYinFFT()(es.Spectrum()(audio_signal))
        except RuntimeError:
            pitch = 0
        return {"frequency": pitch}

    @NendoAnalysisPlugin.plugin_data
    def tempo(self, track: NendoTrack) -> dict:
        """Compute tempo of the given track."""
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(signal(track.resource.src)["signal"])
        return {"tempo": bpm}

    @NendoAnalysisPlugin.plugin_data
    def key(self, track: NendoTrack) -> dict:
        """Compute the musical key of the given track."""
        key, scale, strength = es.KeyExtractor()(signal(track.resource.src)["signal"])
        return {"key": key, "scale": scale, "strength": strength}

    @NendoAnalysisPlugin.plugin_data
    def avg_volume(self, track: NendoTrack) -> dict:
        """Compute the average volume (interpreted as loudness) of the given track."""
        avg_volume = np.mean(signal(track.resource.src)["signal"])
        return {"avg_volume": avg_volume}

    @NendoAnalysisPlugin.plugin_data
    def intensity(self, track: NendoTrack) -> dict:
        """Compute the intensity (interpreted as energy) of the given track."""
        intensity = es.Energy()(signal(track.resource.src)["signal"])
        return {"intensity": intensity}

    @NendoAnalysisPlugin.run_track
    def classify(self, track: NendoTrack) -> None:
        """Run the core classification plugin on the given track and extract all possible features.

        Args:
            track (NendoTrack): The track to run the plugin on.
        """
        self.loudness(track)
        self.avg_volume(track)
        self.duration(track)
        self.frequency(track)
        self.tempo(track)
        self.intensity(track)
        self.key(track)
