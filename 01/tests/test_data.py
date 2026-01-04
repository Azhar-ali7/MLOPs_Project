from src.data import load_and_process
import pandas as pd
from io import StringIO


def test_load_and_process_basic():
    csv = """
63,1,3,145,233,1,0,150,0,2.3,0,0,1,0
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
"""
    df = pd.read_csv(StringIO(csv), header=None)
    # write to temp file
    tmp = 'tests/tmp_heart.csv'
    df.to_csv(tmp, index=False, header=False)
    processed = load_and_process(tmp)
    assert 'target' in processed.columns
    assert processed.shape[0] == 2
