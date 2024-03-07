# -*- encoding: utf-8 -*-
"""Music Information Retrieval Classification Plugin for the Nendo framework."""
import os
from pathlib import Path
from typing import Any

import essentia
import essentia.standard as es
import numpy as np

from nendo import Nendo, NendoAnalysisPlugin, NendoConfig, NendoTrack
from nendo_plugin_classify_core.config import ClassifyCoreConfig
from nendo_plugin_classify_core.utils import (
    download_model,
    filter_predictions,
    make_comma_separated_unique,
)

# Suppress Essentia warnings
essentia.log.infoActive = False
essentia.log.warningActive = False

settings = ClassifyCoreConfig()

def get_signal(track: NendoTrack) -> np.ndarray:
    """Return the signal of the given track."""
    return track.signal[0] if track.signal.ndim == 2 else track.signal

def get_trimmed_signal(track: NendoTrack) -> np.ndarray:
    """Return the signal of the given track, trimmed to 10 minutes."""
    signal = get_signal(track)
    duration = round(es.Duration()(signal), 3)
    if duration > settings.max_trim_duration:
        return signal[:int(settings.max_trim_duration * track.sr)]
    else:
        return signal

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
    sfx_model: es.TensorflowPredict2D = None

    def __init__(self, **data: Any):
        """Initialize plugin."""
        super().__init__(**data)

        model_dir = os.path.join(Path.home(), ".cache", "nendo", "models")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        for model in ["embedding", "mood", "genre", "instrument", "sfx"]:
            model_path = os.path.join(model_dir, f"{model}.pb")
            if not os.path.isfile(model_path):
                download_model(getattr(settings, f"{model}_model"), model_path)

        self.embedding_model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=os.path.join(model_dir, "embedding.pb"), output="PartitionedCall:1",
        )
        self.mood_model = es.TensorflowPredict2D(
            graphFilename=os.path.join(model_dir, "mood.pb")
        )
        self.genre_model = es.TensorflowPredict2D(
            graphFilename=os.path.join(model_dir, "genre.pb"),
            input="serving_default_model_Placeholder",
            output="PartitionedCall:0",
        )
        self.instrument_model = es.TensorflowPredict2D(
            graphFilename=os.path.join(model_dir, "instrument.pb"),
        )
        self.sfx_model = es.TensorflowPredictVGGish(
            graphFilename=os.path.join(model_dir, "sfx.pb"),
            input="melspectrogram",
            output="activations",
        )

    @NendoAnalysisPlugin.plugin_data("loudness")
    def loudness(self, track: NendoTrack) -> dict:
        """Compute the loudness of the given track."""
        loudness = es.Loudness()(get_signal(track))
        return {"loudness": loudness}

    @NendoAnalysisPlugin.plugin_data("duration")
    def duration(self, track: NendoTrack) -> dict:
        """Compute the duration of the given track."""
        duration = round(es.Duration()(get_signal(track)), 3)
        return {"duration": duration}

    @NendoAnalysisPlugin.plugin_data("frequency")
    def frequency(self, track: NendoTrack) -> dict:
        """Compute average frequency of the given track.

        Uses the YIN algorithm to compute the pitch.
        If essentia runs into an error, returns 0.
        """
        try:
            audio_signal = get_trimmed_signal(track)
            # Check if the length of the audio signal is even, if not, trim it
            # This is required for the YIN algorithm to work
            if len(audio_signal) % 2 != 0:
                audio_signal = audio_signal[:-1]  # Excluding the last sample

            pitch, _ = es.PitchYinFFT()(es.Spectrum()(audio_signal))
        except RuntimeError:
            pitch = 0
        return {"frequency": pitch}

    @NendoAnalysisPlugin.plugin_data("tempo")
    def tempo(self, track: NendoTrack) -> dict:
        """Compute tempo of the given track."""
        rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        bpm, _, _, _, _ = rhythm_extractor(get_signal(track))
        return {"tempo": bpm}

    @NendoAnalysisPlugin.plugin_data("key", "scale", "strength")
    def key(self, track: NendoTrack) -> dict:
        """Compute the musical key of the given track."""
        key, scale, strength = es.KeyExtractor()(get_signal(track))
        return {"key": key, "scale": scale, "strength": strength}

    @NendoAnalysisPlugin.plugin_data("avg_volume")
    def avg_volume(self, track: NendoTrack) -> dict:
        """Compute the average volume (interpreted as loudness) of the given track."""
        avg_volume = np.mean(get_signal(track))
        return {"avg_volume": avg_volume}

    @NendoAnalysisPlugin.plugin_data("intensity")
    def intensity(self, track: NendoTrack) -> dict:
        """Compute the intensity (interpreted as energy) of the given track."""
        intensity = es.Energy()(get_signal(track))
        return {"intensity": intensity}

    @NendoAnalysisPlugin.plugin_data("mooods")
    def moods(self, track: NendoTrack) -> dict:
        """Compute the moods of the given track."""
        emb = self.embedding_model(get_trimmed_signal(track))
        predictions = self.mood_model(emb)
        filtered_labels, _ = filter_predictions(
            predictions, settings.mood_theme_classes, threshold=0.05,
        )
        moods = make_comma_separated_unique(filtered_labels)
        return {"moods": moods}

    @NendoAnalysisPlugin.plugin_data("genres")
    def genres(self, track: NendoTrack) -> dict:
        """Compute the genres of the given track."""
        emb = self.embedding_model(get_trimmed_signal(track))
        predictions = self.genre_model(emb)
        filtered_labels, _ = filter_predictions(
            predictions, settings.genre_labels, threshold=0.05,
        )
        filtered_labels = ", ".join(filtered_labels).replace("---", ", ").split(", ")
        genres = make_comma_separated_unique(filtered_labels)
        return {"genres": genres}

    @NendoAnalysisPlugin.plugin_data("instruments")
    def instruments(self, track: NendoTrack) -> dict:
        """Compute the instruments of the given track."""
        emb = self.embedding_model(get_trimmed_signal(track))
        predictions = self.instrument_model(emb)
        filtered_labels, _ = filter_predictions(
            predictions, settings.instrument_classes, threshold=0.05,
        )
        instruments = make_comma_separated_unique(filtered_labels)
        return {"instruments": instruments}

    @NendoAnalysisPlugin.plugin_data("sfx")
    def sfx(self, track: NendoTrack) -> dict:
        """Compute the sfx of the given track."""
        # SR=16000 as recommended in the essentia documentation
        # https://essentia.upf.edu/reference/std_TensorflowPredictVGGish.html
        track_copy = track.copy()
        signal = track_copy.resample(16000)
        signal = signal[0] if signal.ndim == 2 else signal
        duration = round(es.Duration()(signal), 3)
        if duration > settings.max_trim_duration:
            signal = signal[:int(settings.max_trim_duration * track.sr)]
        predictions = self.sfx_model(signal)
        filtered_labels, _ = filter_predictions(
            predictions, settings.sfx_classes, threshold=0.01
        )
        filtered_labels = [label for label in filtered_labels if label not in ["Music", "Electronic music"]]
        sfx = make_comma_separated_unique(filtered_labels)
        return {"sfx": sfx}

    @NendoAnalysisPlugin.run_track
    def classify(self, track: NendoTrack) -> None:
        """Run the core classification plugin on the given track and extract all possible features.

        Args:
            track (NendoTrack): The track to run the plugin on.
        """
        self.duration(track)
        self.loudness(track)
        self.avg_volume(track)
        self.frequency(track)
        self.tempo(track)
        self.intensity(track)
        self.key(track)
        self.moods(track)
        self.genres(track)
        self.instruments(track)
        self.sfx(track)
