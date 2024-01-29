# Nendo Plugin Classify Core

<br>
<p align="left">
    <img src="https://okio.ai/docs/assets/nendo_core_logo.png" width="350" alt="nendo core">
</p>
<br>

<p align="left">
<a href="https://okio.ai" target="_blank">
    <img src="https://img.shields.io/website/https/okio.ai" alt="Website">
</a>
<a href="https://twitter.com/okio_ai" target="_blank">
    <img src="https://img.shields.io/twitter/url/https/twitter.com/okio_ai.svg?style=social&label=Follow%20%40okio_ai" alt="Twitter">
</a>
<a href="https://discord.gg/gaZMZKzScj" target="_blank">
    <img src="https://dcbadge.vercel.app/api/server/XpkUsjwXTp?compact=true&style=flat" alt="Discord">
</a>
</p>

---

Automatic music information retrieval (based on [essentia](https://essentia.upf.edu/)).

## Features

- Extract musical features from a `NendoTrack` or a `NendoCollection`
- Use descriptive features to filter, search and sort your library
- Extract rich features to annotate datasets for training custom models

## Installation

1. [Install nendo](https://github.com/okio-ai/nendo#installation)
2. `pip install nendo-plugin-classify-core`

    > **Note**: Installing this plugin together with another nendo plugin that uses `essentia`, you are likely to run into the runtime error `Error: module 'essentia.standard' has no attribute 'TensorflowPredictEffnetDiscogs'`. See the corresponding entry in the [troubleshooting guide](#essentia-cant-find-the-embedding-model) below for instructions on how to fix this issue.

## Usage

Take a look at a basic usage example below.
For more detailed information, please refer to the [documentation](https://okio.ai/docs/classify-core/advanced).

For more advanced examples, check out the examples folder.
or try it in colab:

<a target="_blank" href="https://colab.research.google.com/drive/1mmbjf0NfsF596p2zDWBDryuxvFYMGAzO?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


```python
from nendo import Nendo, NendoConfig

nd = Nendo(config=NendoConfig(plugins=["nendo_plugin_classify_core"]))

track = nd.library.add_track(file_path='/path/to/track.mp3')

track = nd.plugins.classify_core(track=track)

data = track.get_plugin_data(plugin_name="nendo_plugin_classify_core")
print(data)

tracks_with_filtered_tempo = nd.library.filter_tracks(
    filters={"tempo": (170, 180)},
    plugin_names=["nendo_plugin_classify_core"],
)

assert len(tracks_with_filtered_tempo) == 1
```

## Troubleshooting

### Essentia can't find the embedding model

When I try to run the plugin, I get the following error:

```bash
Failed to import plugin 'nendo_plugin_classify_core'. Error: module 'essentia.standard' has no attribute 'TensorflowPredictEffnetDiscogs'
```

This is due to the fact that you have `essentia` and `essentia-tensorflow` installed at the same time, possibly because you installed another nendo plugin that uses `essentia`, like e.g. [nendo_plugin_quantize_core](https://github.com/okio-ai/nendo_plugin_quantize_core/). The fix in this case is to reinstall essentia, this time only installing the `essentia-tensorflow` package:

```bash
pip uninstall -y essentia essentia-tensorflow
pip install essentia-tensorflow
```

## Contributing

Visit our docs to learn all about how to contribute to Nendo: [Contributing](https://okio.ai/docs/contributing/)

## License

Nendo: MIT License

Essentia: Affero GPLv3 license