# English-to-Spanish dubbing timing and mixing: executable preview

Original, openly AI-assisted work from PaperPilot Tools for one human owner. This repository is a **pre-award technical sample**, not a hosted agent, accepted contract, revenue, or a claim that the full Video Dubbing Agent specification is complete.

## Actual outputs

- [English source video](demo/en.mp4): 14.12 seconds of original synthetic narration and a generated background stem.
- [Spanish dubbed video](demo/es.mp4): the same visual timeline with a licensed stock Spanish voice and sentence subtitles.
- [Spanish SRT](demo/dubbed-es.srt), [English SRT](demo/original-en.srt), and readable [input manifest](demo/manifest.json).
- [Measured render evidence](demo/render-evidence.json): PCM-frame placement, measured utterance ends, dimensions, duration and hashes.
- [Actual time-compression check](demo/tempo-stress-evidence.json): the same real synthesized line fitted into a deliberately shorter synthetic window using bounded FFmpeg tempo adjustment.

The transcript timestamps are known by construction; the three Spanish translations were authored for this fixture. The music is a supplied clean procedural stem. These are **not** automatic speech recognition, speaker diarization, autonomous translation, or source separation results. The visible video labels identify the footage as synthetic.

## What this implementation demonstrates

`dubbing.py` validates source-rights confirmation, explicit voice-mode/consent gating, utterance order, nonoverlapping source windows, exact protected-term counts, audio coverage and finite durations. Missing or invented segments are not silently discarded. The planner speeds an overlong line up only within a declared ceiling of 1.25x. Beyond it, the request fails with a shorter-faithful-translation requirement; no truncation is used to force a pass.

The renderer measures the actual output waveform after FFmpeg instead of assuming an ideal time-scale factor. An overrun remains an error. PCM mixing adds each voice to the original clean background stem and rejects clipping. SRT cues use measured audio boundaries. All three demo utterances start within one sample of their assigned start. Their waveforms end before the next utterance, and every background sample outside speech remains identical before AAC encoding. This proves preservation for a supplied stem, not recovery of music from a mixed customer video.

Protected spelling in source/translation and subtitles is checked. **Correct audible pronunciation of a brand or acronym is not proven by a string check** and remains a listening/lexicon acceptance test. No fluent human has reviewed this synthetic translation or voice quality.

## Reproduce

Pure engine tests require Python 3.10+ only:

```sh
python3 -m unittest discover -s tests -v
```

The exact recorded run passed 34 tests; `test-results.txt` is included. The separate render test uses Python 3.12, FFmpeg/ffprobe with libx264, libass, Flite, and a local Piper voice:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m piper.download_voices es_ES-davefx-medium --data-dir models
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python demo/build_demo.py
```

`voice-provenance.json` pins the downloaded model/config checksums. Downloads must match those checksums; do not silently replace a changed upstream model. Model weights and installed dependencies are intentionally not committed. ONNX uses two intra-operation threads and one inter-operation thread; FFmpeg encoding uses two threads. No paid model API, account, wallet or network request is used during the render. Regeneration can vary across numerical-runtime versions; the evidence hashes identify the actual included artifacts, not a universal bit-for-bit guarantee.

## License and provenance

This original repository is GPL-3.0-or-later, as stated in `LICENSE`; the demonstration imports Piper's GPLv3 implementation rather than presenting it as proprietary code. Piper project: https://github.com/OHF-Voice/piper1-gpl . Spanish stock voice: `es_ES-davefx-medium`, distributed in https://huggingface.co/rhasspy/piper-voices . Its model card lists a CC0 dataset and notes fine-tuning from the lessac English voice. Exact source and hashes are in `voice-provenance.json`; review upstream license files before redistribution of weights. FFmpeg, Flite and ONNX Runtime are separate dependencies with their own licenses. No customer video, real speaker sample or private information is included.

## Remaining work before the full paid-agent acceptance test

The requested complete service still needs lawful upload/link intake, ASR, speaker diarization, translation/glossary handling, a tested separation backend for mixed music/speech, multilingual stock voices, a consented own-voice backend, deletion/retention controls, hosting, buyer dialogue, marketplace lifecycle integration and owner-side adoption/listing. The preview does not implement lip-sync or promise fluent-language human review.

The buyer's 10-minute consented voice, 5-minute two-speaker German and 3-minute Japanese product-demo cases have **not** been run or passed. Publishing this sample does not satisfy those acceptance tests. Its value is inspectable evidence of the timing, no-overrun, supplied-stem preservation and subtitle path for a real 125-USDC build proposal.
