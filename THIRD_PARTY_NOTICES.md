# Third-party notices

The Video Editor source code is licensed under the MIT License (see
[LICENSE](LICENSE)). It bundles or depends on the third-party components below.

## Bundled in this repository

### Silero VAD model (`silence_remover/models/silero_vad_v6.onnx`)

The voice-activity-detection model was created by the Silero Team
(<https://github.com/snakers4/silero-vad>). This ONNX export is the one
distributed with faster-whisper (<https://github.com/SYSTRAN/faster-whisper>).
Both projects are MIT licensed.

```
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

```
MIT License

Copyright (c) 2023 SYSTRAN

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Installed separately (not included in this repository)

These are installed by `pip install -r requirements.txt` or by the user, and
keep their own licenses:

| Component | License | Notes |
|---|---|---|
| [FFmpeg](https://ffmpeg.org/) | LGPL 2.1+ or GPL 2+, depending on the build | Run as a separate program. Not bundled. |
| [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) | GPL v3 or commercial | Desktop GUI. |
| [pyqtgraph](https://www.pyqtgraph.org/) | MIT | Waveform timeline. |
| [NumPy](https://numpy.org/) | BSD 3-Clause | Audio analysis. |
| [ONNX Runtime](https://onnxruntime.ai/) | MIT | Runs the Silero VAD model. |

> **Note on distributing binaries.** Because PyQt6 is GPL v3, a packaged
> application that bundles it (for example the standalone Windows build
> planned in ROADMAP item 6.1) must be distributed under terms compatible with
> the GPL v3, including making the source available. The same applies if a
> GPL build of FFmpeg is bundled. The MIT-licensed source in this repository
> is GPL-compatible. Switching the GUI to PySide6 (LGPL) would avoid this
> requirement.
