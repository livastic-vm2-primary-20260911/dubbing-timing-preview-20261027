"""Build a real EN-to-ES stock-voice dub from original synthetic inputs.

The transcript/translation and clean music stem are supplied fixtures, not ASR,
AI translation, voice cloning or separation results. Uses Piper (GPLv3).
"""
from pathlib import Path
from array import array
from dataclasses import asdict
import hashlib
import json
import math
import os
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo"
sys.path.insert(0, str(ROOT))
from dubbing import TimingError, plan, mix_stem, pcm_read, pcm_write, subtitles

RATE = 22050
TEXT = [
    ("DemoBrand keeps the original background music.", "DemoBrand conserva la música de fondo original."),
    ("Each line starts at its original time.", "Cada frase empieza en su momento original."),
    ("The price stays in USDC, and no voice is cloned.", "El precio sigue en USDC y no se clona ninguna voz."),
]

def run(args, timeout=90):
    result = subprocess.run(args, cwd=OUT, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Command failed: {args[0]}\n{result.stderr[-3000:]}")
    return result.stdout

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ffmpeg(args):
    return run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-filter_threads", "1", "-filter_complex_threads", "1", *args])

def scaled(samples, peak=10000):
    highest = max(abs(x) for x in samples)
    factor = min(1.0, peak / highest) if highest else 1.0
    return array("h", (round(x*factor) for x in samples))

def main():
    import onnxruntime as ort
    from piper import PiperVoice
    from piper.config import PiperConfig, SynthesisConfig
    model = ROOT / "models/es_ES-davefx-medium.onnx"
    configuration = ROOT / "models/es_ES-davefx-medium.onnx.json"
    expected = json.loads((ROOT / "voice-provenance.json").read_text())["files"]
    for path in (model, configuration):
        if digest(path) != expected[path.name]["sha256"]:
            raise RuntimeError("Stock voice checksum differs from reviewed provenance")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    voice = PiperVoice(config=PiperConfig.from_dict(json.loads(configuration.read_text())),
                       session=ort.InferenceSession(str(model), sess_options=options,
                                                    providers=["CPUExecutionProvider"]))
    english, spanish, lengths, rows = {}, {}, {}, []
    cursor = 1.0
    for i,(en,es) in enumerate(TEXT,1):
        identifier = f"line-{i:02}"
        (OUT / (identifier+"-en.txt")).write_text(en,encoding="utf-8")
        ffmpeg(["-f", "lavfi", "-i", f"flite=textfile={identifier}-en.txt:voice=slt",
                "-ar", str(RATE), "-ac", "1", identifier+"-en.wav"])
        with wave.open(str(OUT/(identifier+"-es-raw.wav")),"wb") as w:
            voice.synthesize_wav(es,w,SynthesisConfig(noise_scale=0.0,noise_w_scale=0.0))
        english[identifier] = scaled(pcm_read(OUT/(identifier+"-en.wav"),RATE))
        raw = pcm_read(OUT/(identifier+"-es-raw.wav"),RATE)
        lengths[identifier] = len(raw)/RATE
        # Deliberately constructed timing fixture: English fits, Spanish may
        # need a bounded speed-up. This is not a real-video alignment claim.
        window = round(max(len(english[identifier])/RATE+0.15, len(raw)/RATE/1.15+0.08),3)
        rows.append(dict(id=identifier,speaker="stock-narrator",start=round(cursor,3),
                         end=round(cursor+window,3),original=en,translation=es))
        cursor += window+0.75
    duration = math.ceil((cursor+0.5)*25)/25
    segments = plan(duration,rows,lengths,rights_confirmed=True,
                    voice_mode="stock",protected_terms=["DemoBrand","USDC"])
    tempo_receipts=[]
    for s in segments:
        fitted = OUT/(s.id+"-es-fit.wav")
        # Small safety margin handles FFmpeg's windowed time-domain filter.
        # If the measured result still does not fit, fail rather than truncate.
        tempo = min(1.25,s.tempo*1.015) if s.tempo>1 else 1.0
        ffmpeg(["-i",s.id+"-es-raw.wav","-af",f"atempo={tempo:.9f}",
                "-ar",str(RATE),"-ac","1",fitted.name])
        spanish[s.id]=scaled(pcm_read(fitted,RATE))
        tempo_receipts.append(dict(id=s.id,planned_tempo=s.tempo,rendered_tempo=tempo,
                                   raw_seconds=s.audio_duration,fitted_seconds=len(spanish[s.id])/RATE))
    total_frames=round(duration*RATE)
    background=array("h",(round(650*(math.sin(2*math.pi*220*i/RATE)+.35*math.sin(2*math.pi*330*i/RATE)))
                           for i in range(total_frames)))
    pcm_write(OUT/"music-original.wav",background,RATE)
    en_segments=plan(duration,rows,{k:len(v)/RATE for k,v in english.items()},
                     rights_confirmed=True,protected_terms=["DemoBrand","USDC"])
    en_mix,en_receipt=mix_stem(background,english,en_segments,RATE)
    es_mix,es_receipt=mix_stem(background,spanish,segments,RATE)
    # Silent regions must preserve every original background PCM sample.
    occupied=[False]*total_frames
    for receipt in es_receipt:
        occupied[receipt["start_frame"]:receipt["end_frame"]]=[True]*(receipt["end_frame"]-receipt["start_frame"])
    quiet_samples=sum(not x for x in occupied)
    assert quiet_samples>RATE
    assert all(es_mix[i]==background[i] for i,used in enumerate(occupied) if not used)
    assert all(r["end_frame"]<=r["window_end_frame"] for r in es_receipt)
    assert all(abs(r["actual_start"]-s.start)<1/RATE for r,s in zip(es_receipt,segments))
    pcm_write(OUT/"original-en.wav",en_mix,RATE)
    pcm_write(OUT/"dubbed-es.wav",es_mix,RATE)
    for i,row in enumerate(en_receipt):row["translation"]=TEXT[i][0]
    (OUT/"original-en.srt").write_text(subtitles(en_receipt),encoding="utf-8")
    (OUT/"dubbed-es.srt").write_text(subtitles(es_receipt),encoding="utf-8")
    for lang,audio,title in [("en","original-en.wav","ORIGINAL EN - SYNTHETIC QA FIXTURE"),
                             ("es","dubbed-es.wav","SPANISH STOCK DUB - NOT VOICE CLONING")]:
        sub="original-en.srt" if lang=="en" else "dubbed-es.srt"
        vf=(f"drawtext=text='{title}':fontsize=18:fontcolor=white:box=1:boxcolor=black@0.8:x=10:y=10,"
            f"subtitles={sub}:force_style='FontSize=18,Outline=2,MarginV=20'")
        ffmpeg(["-f","lavfi","-i",f"testsrc2=size=640x360:rate=25:duration={duration}",
                "-i",audio,"-vf",vf,"-c:v","libx264","-threads","2","-preset","veryfast",
                "-crf","27","-c:a","aac","-b:a","128k","-t",str(duration),"-movflags","+faststart",f"{lang}.mp4"])
    long_lengths=dict(lengths);long_lengths["line-01"]=(rows[0]["end"]-rows[0]["start"])*1.30
    try:
        plan(duration,rows,long_lengths,rights_confirmed=True)
    except TimingError as exc:
        long_line_error=str(exc)
    else:
        raise AssertionError("Overlong line must not be accepted")
    video={}
    for name in ["en.mp4","es.mp4"]:
        probe=json.loads(run(["ffprobe","-v","error","-show_streams","-show_format","-of","json",name]))
        stream=next(x for x in probe["streams"] if x["codec_type"]=="video")
        assert (stream["width"],stream["height"])==(640,360)
        assert abs(float(probe["format"]["duration"])-duration)<0.1
        video[name]={"width":640,"height":360,"duration":probe["format"]["duration"],"sha256":digest(OUT/name)}
    # Real waveform stress case: same stock utterance, shorter synthetic slot.
    raw_stress=pcm_read(OUT/"line-03-es-raw.wav",RATE)
    stress_duration=len(raw_stress)/RATE
    stress_window=round(stress_duration/1.15,3)
    stress_rows=[dict(id="stress",speaker="stock-narrator",start=1,end=1+stress_window,
                      original=TEXT[2][0],translation=TEXT[2][1])]
    stress_plan=plan(6,stress_rows,{"stress":stress_duration},rights_confirmed=True,
                     protected_terms=["USDC"])
    stress_tempo=stress_plan[0].tempo*1.015
    ffmpeg(["-i","line-03-es-raw.wav","-af",f"atempo={stress_tempo:.9f}",
            "-ar",str(RATE),"-ac","1","long-line-fitted.wav"])
    fitted_stress=pcm_read(OUT/"long-line-fitted.wav",RATE)
    _,stress_receipt=mix_stem(array("h",[0]*(RATE*6)),{"stress":fitted_stress},stress_plan,RATE)
    stress_evidence={"fixture":"Same real stock speech in a deliberately shorter synthetic window; not new source footage",
                     "raw_seconds":stress_duration,"window_seconds":stress_window,
                     "tempo":stress_tempo,"actual_seconds":len(fitted_stress)/RATE,
                     "audio_fully_fits":len(fitted_stress)<=round(stress_window*RATE),
                     "receipt":stress_receipt}
    (OUT/"tempo-stress-evidence.json").write_text(json.dumps(stress_evidence,ensure_ascii=False,indent=2)+"\n")
    manifest={"rights_confirmed":True,"synthetic":True,"duration":duration,
              "transcript_provenance":"authored script; not ASR", "translation_provenance":"authored English-Spanish fixture; not an automated translation system",
              "background_provenance":"original generated clean stem, not separation", "protected_terms":["DemoBrand","USDC"],"rows":rows}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    evidence={"synthetic_only":True,"source_separation_implemented":False,"asr_implemented":False,
              "voice_cloning_implemented":False,"fluent_human_review":False,"stock_voice":"es_ES-davefx-medium",
              "source_duration":duration,"planned_segments":[asdict(s) for s in segments],
              "rendered_tempo":tempo_receipts,"measured_utterances":es_receipt,
              "background_silent_region_exact_match":True,"matched_background_samples":quiet_samples,
              "background_stem_sha256":digest(OUT/"music-original.wav"),"overlong_line_rejected":long_line_error,
              "video":video,"ffmpeg":run(["ffmpeg","-version"]).splitlines()[0]}
    (OUT/"render-evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(evidence,ensure_ascii=False,indent=2))

if __name__=="__main__":main()
