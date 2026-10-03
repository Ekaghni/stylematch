from pathlib import Path

import pytest

import stylematch
from stylematch import compare, interpret, similarity
from stylematch.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def read(name):
    return (EXAMPLES / name).read_text(encoding="utf-8")


def test_identical_texts_score_one():
    text = "The quick brown fox jumps over the lazy dog, again and again."
    assert similarity(text, text) == pytest.approx(1.0)


def test_score_is_symmetric_and_bounded():
    a, b = read("casual_a.txt"), read("formal.txt")
    assert similarity(a, b) == pytest.approx(similarity(b, a))
    assert 0.0 <= similarity(a, b) <= 1.0


def test_same_voice_beats_different_voice():
    same = similarity(read("casual_a.txt"), read("casual_b.txt"))
    different = similarity(read("casual_a.txt"), read("formal.txt"))
    assert same > different


def test_matches_scikit_learn():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    pairs = [
        (read("casual_a.txt"), read("casual_b.txt")),
        (read("casual_a.txt"), read("formal.txt")),
        ("Short  text\twith   odd\n\nspacing here", "Another SHORT text, also odd!"),
    ]
    for a, b in pairs:
        matrix = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True).fit_transform([a, b])
        expected = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        assert similarity(a, b) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("a,b", [("", "hello there"), ("hello there", "   "), ("", "")])
def test_empty_input_raises(a, b):
    with pytest.raises(ValueError):
        similarity(a, b)


def test_very_short_text_does_not_crash():
    assert similarity("a", "b") == 0.0


def test_short_text_gets_a_warning():
    result = compare("too short", "also short")
    assert result.warnings


def test_long_enough_text_has_no_warning():
    assert not compare(read("casual_a.txt"), read("casual_b.txt")).warnings


def test_interpret_thresholds():
    assert interpret(0.70).startswith("Very similar")
    assert interpret(0.69).startswith("Somewhat similar")
    assert interpret(0.50).startswith("Somewhat similar")
    assert interpret(0.49).startswith("Different")


def test_version_exposed():
    assert stylematch.__version__


def test_cli_files(capsys):
    code = main([str(EXAMPLES / "casual_a.txt"), str(EXAMPLES / "casual_b.txt")])
    assert code == 0
    assert "Similarity score" in capsys.readouterr().out


def test_cli_json(capsys):
    import json

    assert main(["--json", "--text", "one two three", "one two four"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert set(data) == {"score", "verdict", "words_1", "words_2", "warnings"}


def test_cli_missing_file(capsys):
    assert main(["nope.txt", "nada.txt"]) == 1
    assert "stylematch:" in capsys.readouterr().err


def test_cli_no_args(capsys):
    assert main([]) == 2


# ---- AI detection (skipped when torch/transformers are not installed) ----

def test_detect_missing_deps_message(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    from stylematch import detect as d

    d._cache.clear()
    with pytest.raises(d.MissingDependencyError):
        d.detect("some text")


def test_detect_empty_raises():
    from stylematch.detect import detect

    with pytest.raises(ValueError):
        detect("   ")


def test_chunking_folds_small_tail():
    from stylematch.detect import CHUNK_WORDS, _chunks

    text = " ".join(["word"] * (CHUNK_WORDS + 10))
    assert len(_chunks(text)) == 1


@pytest.mark.slow
def test_detect_separates_ai_from_human():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from stylematch import detect

    ai = detect.detect(read("ai_style.txt"), device="cpu")
    human = detect.detect(read("human_twain.txt"), device="cpu")
    assert ai.score >= detect.LIKELY_AI and human.score < detect.UNSURE


def test_cli_flags_after_positionals(capsys):
    assert main([str(EXAMPLES / "casual_a.txt"), str(EXAMPLES / "casual_b.txt"), "--json"]) == 0
    assert '"score"' in capsys.readouterr().out
    assert main(["--json", str(EXAMPLES / "casual_a.txt"), str(EXAMPLES / "casual_b.txt")]) == 0


def test_detect_verdict_bands():
    from stylematch.detect import LIKELY_AI, UNSURE, _verdict

    assert _verdict(LIKELY_AI).startswith("Likely AI")
    assert _verdict((LIKELY_AI + UNSURE) / 2).startswith("Unclear")
    assert _verdict(UNSURE - 0.01).startswith("Likely human")
