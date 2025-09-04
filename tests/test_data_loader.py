from pathlib import Path
import pandas as pd
from src.data.data_loader import SensorDataLoader


def test_get_subject_and_load(tmp_path: Path):
    src = Path(__file__).parent / "fixtures" / "tiny.csv"
    dst = tmp_path / "tiny.csv"
    dst.write_text(src.read_text())

    loader = SensorDataLoader(str(dst), chunk_size=2)
    subjects = loader.get_subject_list()
    assert subjects == ["1"], subjects

    df = loader.load_subject_data("1")
    assert not df.empty
    assert set(["X","Y","Z","EDA","HR","TEMP","id","datetime","label"]).issubset(df.columns)