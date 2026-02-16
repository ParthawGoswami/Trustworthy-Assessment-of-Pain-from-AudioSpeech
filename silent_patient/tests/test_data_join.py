from pathlib import Path

import pandas as pd

from silent_patient.data import resolve_wav_path


def test_resolve_wav_path():
    audio_root = Path("/tmp/audio")
    row = pd.Series({"PID": "p123", "COND": "LC", "UTTNUM": 7, "UTTID": 99999})
    p = resolve_wav_path(audio_root, row)
    assert str(p).endswith("/p123/p123.LC.7.99999.wav")
