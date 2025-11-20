import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import pytest

HEADER_NET_ROOT = Path(__file__).resolve().parents[1]
if str(HEADER_NET_ROOT) not in sys.path:
    sys.path.append(str(HEADER_NET_ROOT))

from utils.detections import make_video_key
from utils.labels import canonical_match_name, load_header_labels
from utils.videos import infer_half_from_stem

pytestmark = pytest.mark.filterwarnings(r"ignore:datetime\.datetime.utcnow")


MODULE_PATH = Path(__file__).resolve().parents[1] / "cache" / "build_labelled-only_ball_det_dict.py"


def load_labelled_module():
    spec = importlib.util.spec_from_file_location("labelled_only_builder", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@dataclass
class DummyVideo:
    video_id: str
    path: Path
    half: int
    rel_dir: Path


def build_dummy_videos(dataset_root: Path):
    videos = []
    soccer_root = dataset_root / "SoccerNet"
    for match_path in soccer_root.glob("*/*/*"):
        if not match_path.is_dir():
            continue
        for video_file in match_path.glob("*.*"):
            if video_file.suffix.lower() not in {".mp4", ".mkv"}:
                continue
            stem = video_file.stem
            half = infer_half_from_stem(stem)
            canonical_name = canonical_match_name(match_path.name)
            video_id = make_video_key(canonical_name, half)
            rel_dir = match_path.relative_to(soccer_root)
            videos.append(DummyVideo(video_id=video_id, path=video_file, half=half, rel_dir=rel_dir))
    return videos


def _prepare_datasets(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    example_match_dir = dataset_root / "SoccerNet" / "league" / "season" / "Example Match"
    example_match_dir.mkdir(parents=True)
    (example_match_dir / "1_224p.mkv").touch()

    unlabelled_match_dir = dataset_root / "SoccerNet" / "league" / "season" / "Another Match"
    unlabelled_match_dir.mkdir(parents=True)
    (unlabelled_match_dir / "1_224p.mkv").touch()

    header_root = tmp_path / "header_dataset"
    labelled_dir = header_root / "SoccerNetV2" / "ExampleMatch"
    labelled_dir.mkdir(parents=True)
    (labelled_dir / "annotations_half1.csv").write_text("frame\n42\n")
    (labelled_dir / "annotations_half2.csv").write_text("frame\n84\n")

    orphan_dir = header_root / "SoccerNetV2" / "MissingMatch"
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "orphan.csv").write_text("frame\n10\n")

    return dataset_root, header_root, labelled_dir


def test_collect_labelled_videos_filters_to_labelled_matches(tmp_path):
    dataset_root, header_root, _ = _prepare_datasets(tmp_path)

    module = load_labelled_module()
    all_videos = build_dummy_videos(dataset_root)
    labels_df = load_header_labels(header_root)

    filtered, missing, name_map, skipped = module.collect_labelled_videos(all_videos, labels_df)

    canonical = canonical_match_name("Example Match")
    expected_key = make_video_key(canonical, 1)
    filtered_keys = {video.video_id for video in filtered}

    assert filtered_keys == {expected_key}
    assert canonical in missing
    assert missing[canonical] == {2}
    assert canonical in name_map
    assert "ExampleMatch" in name_map[canonical]
    assert canonical_match_name("MissingMatch") in skipped
    assert canonical not in skipped


def test_load_labels_dataframe_accepts_subdirectories(tmp_path):
    _, header_root, labelled_dir = _prepare_datasets(tmp_path)

    module = load_labelled_module()

    df_root, resolved_root = module.load_labels_dataframe(header_root)
    df_sub, resolved_sub = module.load_labels_dataframe(labelled_dir)

    assert resolved_root == header_root
    assert resolved_sub == header_root
    assert not df_root.empty
    pd.testing.assert_frame_equal(df_root, df_sub)


def test_infer_half_from_stem_handles_resolution_suffixes():
    assert infer_half_from_stem("1_224p") == 1
    assert infer_half_from_stem("2_224p") == 2
    assert infer_half_from_stem("FirstHalf") == 1
    assert infer_half_from_stem("SecondHalf_720p") == 2


def test_collect_labelled_videos_matches_real_dataset():
    dataset_path = Path(__file__).resolve().parents[2] / "DeepImpact"
    soccer_root = dataset_path / "SoccerNet"
    header_dataset = dataset_path / "header_dataset"

    if not soccer_root.exists() or not header_dataset.exists():
        pytest.skip("Full DeepImpact dataset not available")

    target_matches_raw = [
        "2015-04-11 - 16-30 Bayern Munich 3 - 0 Eintracht Frankfurt",
        "2015-04-19-21-45Inter0-0ACMilan",
        "2015-04-22-21-45Monaco0-0Juventus",
        "2015-04-25-21-45Inter2-1ASRoma",
        "2015-05-06-21-45Barcelona3-0BayernMunich",
        "2015-09-12-17-00Espanyol0-6RealMadrid",
        "2015-09-26-19-15RealMadrid0-0Malaga",
        "2015-09-29-21-45Lyon0-1Valencia",
        "2015-11-07-19-00ParisSG5-0Toulouse",
        "2015-11-25-22-45BMonchengladbach4-2Sevilla",
        "2016-01-13-22-45Chelsea2-2WestBrom",
        "2016-01-24-22-30Betis1-1RealMadrid",
        "2016-02-14-19-15ManchesterCity1-2Tottenham",
        "2016-04-09-16-30VfBStuttgart1-3BayernMunich",
        "2016-09-17-16-30Dortmund6-0Darmstadt",
        "2016-09-20 - 21-00 Wolfsburg 1 - 5 Dortmund",
        "2016-09-21-21-45ASRoma4-0Crotone",
        "2016-09-28-21-45Ludogorets1-3ParisSG",
        "2016-10-15-18-00Nancy1-2ParisSG",
        "2016-10-17-22-00Liverpool0-0ManchesterUnited",
        "2016-10-19 - 21-45 Barcelona 4 - 0 Manchester City",
        "2016-10-25-21-45Genoa3-0ACMilan",
        "2016-11-26-17-30EintrachtFrankfurt2-1Dortmund",
        "2016-12-11-22-45ParisSG2-2Nice",
        "2016-12-19-23-00Everton0-1Liverpool",
        "2017-02-15-22-45RealMadrid3-1Napoli",
        "2017-04-09-21-45Lazio0-3Napoli",
        "2017-04-18-21-45RealMadrid4-2BayernMunich",
        "2017-05-20-22-00ParisSG1-1Caen",
    ]
    target_matches = {canonical_match_name(name) for name in target_matches_raw}

    module = load_labelled_module()
    labels_df, _ = module.load_labels_dataframe(header_dataset)
    labels_df = labels_df[
        labels_df["video_id"].map(lambda x: canonical_match_name(str(x)) in target_matches)
    ].copy()

    videos = []
    for match_path in soccer_root.glob("*/*/*"):
        if not match_path.is_dir():
            continue
        canonical = canonical_match_name(match_path.name)
        if canonical not in target_matches:
            continue
        for video_file in match_path.glob("*.*"):
            if video_file.suffix.lower() not in {".mp4", ".mkv"}:
                continue
            half = infer_half_from_stem(video_file.stem)
            rel_dir = match_path.relative_to(soccer_root)
            video_id = make_video_key(canonical, half)
            videos.append(
                DummyVideo(video_id=video_id, path=video_file, half=half, rel_dir=rel_dir)
            )

    available_matches = {video.video_id.rsplit("_half", 1)[0] for video in videos}
    missing_dirs = target_matches - available_matches
    if missing_dirs:
        pytest.skip(f"SoccerNet videos missing for expected matches: {sorted(missing_dirs)}")

    filtered, missing, _, skipped = module.collect_labelled_videos(videos, labels_df)
    found_matches = {video.video_id.rsplit("_half", 1)[0] for video in filtered}

    assert target_matches.issubset(found_matches)
    assert not any(missing.get(match) for match in target_matches)
    assert target_matches.isdisjoint(skipped)


def test_load_header_labels_infers_half_from_filename(tmp_path):
    header_root = tmp_path / "header_dataset"
    match_dir = header_root / "SoccerNetV2" / "TestMatch"
    match_dir.mkdir(parents=True)

    (match_dir / "TestMatch_1_labels.csv").write_text("frame\n10\n")
    (match_dir / "TestMatch_2_labels.csv").write_text("frame\n20\n")

    df = load_header_labels(header_root)
    subset = df[df["video_id"] == "TestMatch"]
    assert sorted(subset["half"].unique().tolist()) == [1, 2]
