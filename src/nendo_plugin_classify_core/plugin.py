# -*- encoding: utf-8 -*-
"""Music Information Retrieval Classification Plugin for the Nendo framework."""
from functools import lru_cache
from typing import Any

import essentia.standard as es
import essentia

# Suppress Essentia warnings
essentia.log.infoActive = False
essentia.log.warningActive = False
import numpy as np
import os
from nendo import Nendo, NendoAnalysisPlugin, NendoConfig, NendoTrack

from nendo_plugin_classify_core.config import ClassifyCoreConfig
from nendo_plugin_classify_core.utils import (
    download_model,
    filter_predictions,
    make_comma_separated_unique,
)


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



settings = ClassifyCoreConfig()


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
    embedding_model: es.TensorflowPredictEffnetDiscogs = None
    mood_model: es.TensorflowPredict2D = None
    genre_model: es.TensorflowPredict2D = None
    instrument_model: es.TensorflowPredict2D = None

    def __init__(self, **data: Any):
        """Initialize plugin."""
        super().__init__(**data)

        if not os.path.isfile("models"):
            os.makedirs("models", exist_ok=True)

        for model in ["embedding", "mood", "genre", "instrument"]:
            model_path = f"models/{model}.pb"
            if not os.path.isfile(model_path):
                download_model(getattr(settings, f"{model}_model"), model_path)

        self.embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename="models/embedding.pb", output="PartitionedCall:1"
        )
        self.mood_model = es.TensorflowPredict2D(graphFilename="models/mood.pb")
        self.genre_model = es.TensorflowPredict2D(
            graphFilename="models/genre.pb",
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        self.instrument_model = es.TensorflowPredict2D(
            graphFilename="models/instrument.pb"
        )

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

    @NendoAnalysisPlugin.plugin_data
    def moods(self, track: NendoTrack) -> dict:
        """Compute the moods of the given track."""
        emb = self.embedding_model(signal(track.resource.src)["signal"])
        predictions = self.mood_model(emb)
        filtered_labels, _ = filter_predictions(
            predictions, settings.mood_theme_classes, threshold=0.05
        )
        moods = make_comma_separated_unique(filtered_labels)
        return {"moods": moods}

    @NendoAnalysisPlugin.plugin_data
    def genres(self, track: NendoTrack) -> dict:
        """Compute the genres of the given track."""
        emb = self.embedding_model(signal(track.resource.src)["signal"])
        predictions = self.genre_model(emb)
        filtered_labels, _ = filter_predictions(
            predictions, settings.genre_labels, threshold=0.05
        )
        filtered_labels = ", ".join(filtered_labels).replace("---", ", ").split(", ")
        genres = make_comma_separated_unique(filtered_labels)
        return {"genres": genres}

    @NendoAnalysisPlugin.plugin_data
    def instruments(self, track: NendoTrack) -> dict:
        """Compute the instruments of the given track."""
        emb = self.embedding_model(signal(track.resource.src)["signal"])
        predictions = self.instrument_model(emb)
        filtered_labels, _ = filter_predictions(
            predictions, settings.instrument_classes, threshold=0.05
        )
        return {"instruments": filtered_labels[0]}

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
        self.moods(track)
        self.genres(track)
        self.instruments(track)
