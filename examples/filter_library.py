from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_classify_core"]))

track = nd.library.add_tracks(file_path="/path/to/track.mp3")
track = nd.plugins.classify_core(track)

data = track.get_plugin_data(plugin_name="nendo_plugin_classify_core")

tracks_with_filtered_tempo = nd.library.filter_tracks(
    filters={"tempo": (170, 180)},
    plugin_names=["nendo_plugin_classify_core"],
)

assert len(tracks_with_filtered_tempo) == 1
