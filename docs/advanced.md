# Advanced Usage

The classify core plugin provides different functionality to analyze and extract features from your nendo library.
Concretely the following features will be extracted:

| Feature      | Description                                                                                                                                                          |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `loudness`   | This algorithm computes the loudness of an audio signal defined by Steven's power law. It computes loudness as the energy of the signal raised to the power of 0.67. |    
| `duration`   | The tracks duration in seconds.                                                                                                                                      |
| `key`        | The key of the track.                                                                                                                                                |
| `scale`      | The scale of the track.                                                                                                                                              |
| `strength`   | The strength of the estimated key.                                                                                                                                   |                                                                                                                                                                     |
| `tempo`      | The tempo of the track in BPM.                                                                                                                                       |
| `avg_volume` | The average volume of the track.                                                                                                                                     |
| `frequency`  | This algorithm estimates the fundamental frequency given the spectrum of a monophonic music signal.                                                                  |
| `intensity`  | This algorithm computes the energy of an array.                                                                                                                      |

For more details also refer to the [essentia documentation](https://essentia.upf.edu/documentation.html)
on which most of the feature extraction is based on.


[//]: # (## Filtering your library based on MIR features)
[//]: # (## Accessing plugin data)
