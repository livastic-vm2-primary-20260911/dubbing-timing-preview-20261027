import copy
import math
import tempfile
import unittest
from array import array
from pathlib import Path
from dubbing import TimingError, plan, mix_stem, subtitles, srt_timestamp, pcm_read, pcm_write

class DubbingTests(unittest.TestCase):
    def setUp(self):
        self.rows=[dict(id="a",speaker="stock-1",start=1,end=5,original="DemoBrand uses USDC.",translation="DemoBrand usa USDC."),
                   dict(id="b",speaker="stock-2",start=6,end=10,original="Keep the music.",translation="Conserva la música.")]
        self.lengths={"a":3,"b":4}
    def make(self,**kw):
        return plan(12,self.rows,self.lengths,rights_confirmed=True,protected_terms=["DemoBrand","USDC"],**kw)
    def test_short_line_is_not_sped_up(self):self.assertEqual(self.make()[0].tempo,1)
    def test_equal_window_is_exact(self):self.assertEqual(self.make()[1].planned_end,10)
    def test_bounded_speedup(self):
        self.lengths["b"]=4.8;self.assertAlmostEqual(self.make()[1].tempo,1.2)
    def test_exact_limit_is_allowed(self):
        self.lengths["b"]=5;self.assertEqual(self.make()[1].tempo,1.25)
    def test_long_line_is_not_truncated(self):
        self.lengths["b"]=5.01
        with self.assertRaisesRegex(TimingError,"shorter faithful"):self.make()
    def test_rights_required(self):
        with self.assertRaises(TimingError):plan(12,self.rows,self.lengths)
    def test_unconsented_clone_rejected(self):
        with self.assertRaisesRegex(TimingError,"consent"):self.make(voice_mode="cloned")
    def test_stock_needs_no_clone_consent(self):self.make(voice_mode="stock",speaker_consent=False)
    def test_explicit_consent_is_only_a_gate_not_clone_implementation(self):self.make(voice_mode="cloned",speaker_consent=True)
    def test_unknown_voice_mode(self):
        with self.assertRaises(TimingError):self.make(voice_mode="impersonate")
    def test_protected_term_case_change(self):
        self.rows[0]["translation"]="Tapeout usa USDC."
        with self.assertRaisesRegex(TimingError,"Protected term"):self.make()
    def test_extra_protected_term_not_invented(self):
        self.rows[1]["translation"]="Conserva USDC."
        with self.assertRaises(TimingError):self.make()
    def test_protected_term_is_not_substring(self):
        self.rows[0]["original"]="DemoBrand uses USDCx.";self.rows[0]["translation"]="DemoBrand usa USDCx.";self.make()
    def test_missing_translation(self):
        self.rows[1]["translation"]=" "
        with self.assertRaises(TimingError):self.make()
    def test_subtitle_injection_rejected(self):
        self.rows[1]["translation"]="line\n\n5"
        with self.assertRaises(TimingError):self.make()
    def test_missing_audio(self):
        del self.lengths["b"]
        with self.assertRaises(TimingError):self.make()
    def test_extra_audio(self):
        self.lengths["c"]=1
        with self.assertRaises(TimingError):self.make()
    def test_zero_audio(self):
        self.lengths["a"]=0
        with self.assertRaises(TimingError):self.make()
    def test_duplicate_id(self):
        self.rows[1]["id"]="a"
        with self.assertRaises(TimingError):self.make()
    def test_path_shaped_id(self):
        self.rows[0]["id"]="../a"
        with self.assertRaises(TimingError):self.make()
    def test_overlapping_windows(self):
        self.rows[1]["start"]=4
        with self.assertRaises(TimingError):self.make()
    def test_unsorted_windows(self):
        self.rows.reverse()
        with self.assertRaises(TimingError):self.make()
    def test_outside_source(self):
        self.rows[1]["end"]=13
        with self.assertRaises(TimingError):self.make()
    def test_zero_window(self):
        self.rows[0]["end"]=1
        with self.assertRaises(TimingError):self.make()
    def test_nonfinite_numbers(self):
        for value in [float("nan"),float("inf"),True,None,"bad"]:
            with self.subTest(value=value):
                with self.assertRaises(TimingError):plan(value,self.rows,self.lengths,rights_confirmed=True)
    def test_excessive_speed_policy(self):
        with self.assertRaises(TimingError):self.make(max_tempo=2)
    def test_mixing_preserves_background_in_all_silent_regions(self):
        bg=array("h",[50]*120);voices={"a":[100]*30,"b":[200]*40}
        out,receipt=mix_stem(bg,voices,self.make(),10)
        self.assertEqual(list(bg),[50]*120)
        self.assertEqual(list(out[:10]),[50]*10);self.assertEqual(list(out[40:60]),[50]*20);self.assertEqual(list(out[100:]),[50]*20)
        self.assertEqual(out[10],150);self.assertEqual(out[60],250)
        self.assertEqual(receipt[1]["actual_start"],6);self.assertEqual(receipt[1]["actual_end"],10)
    def test_measured_tempo_overrun_is_rejected(self):
        with self.assertRaisesRegex(TimingError,"no truncation"):mix_stem([0]*120,{"a":[1]*41,"b":[1]*40},self.make(),10)
    def test_clipping_is_not_silently_saturated(self):
        with self.assertRaisesRegex(TimingError,"clip"):mix_stem([32000]*120,{"a":[1000]*30,"b":[1]*40},self.make(),10)
    def test_background_length(self):
        with self.assertRaises(TimingError):mix_stem([0]*90,{"a":[1]*30,"b":[1]*40},self.make(),10)
    def test_srt_matches_measured_audio(self):
        _,r=mix_stem([0]*120,{"a":[1]*30,"b":[1]*40},self.make(),10)
        t=subtitles(r);self.assertIn("00:00:01,000 --> 00:00:04,000",t);self.assertIn("Conserva la música.",t)
    def test_timestamp_carry(self):self.assertEqual(srt_timestamp(59.9996),"00:01:00,000")
    def test_timestamp_hour(self):self.assertEqual(srt_timestamp(3601.012),"01:00:01,012")
    def test_pcm_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.wav";pcm_write(p,[-32768,-1,0,1,32767],22050);self.assertEqual(list(pcm_read(p,22050)),[-32768,-1,0,1,32767])
            with self.assertRaises(TimingError):pcm_read(p,16000)

if __name__=="__main__":unittest.main()
