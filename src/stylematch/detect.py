"""Neural AI-text detection.

Uses the pretrained desklib/ai-text-detector-v1.01 classifier (DeBERTa-v3-large,
MIT licence, about 1.7 GB, downloaded from the Hugging Face Hub on first use).
It runs on a CUDA GPU when PyTorch can see one and falls back to the CPU
otherwise (slower, but it works). Install the extras with ``pip install "stylematch[ai]"``.

No detector is fully reliable. Formal human writing (encyclopedias, old
novels, reports) is the usual source of false positives, and lightly edited
AI text often slips through. Treat the score as a hint, not evidence.
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MIN_WORDS = 50
CHUNK_WORDS = 250

# Cutoffs picked on a development split of a 2,600-text benchmark and checked on
# a held-out split: 0.977 gives about 2% false flags on human text (about 81% of
# AI text caught), 0.907 gives about 5% (about 90% caught). See the README.
LIKELY_AI = 0.977
UNSURE = 0.907

MODEL_ID = "desklib/ai-text-detector-v1.01"

_INSTALL_HINT = (
    'AI detection needs PyTorch and transformers. Install them with:\n'
    '    pip install "stylematch[ai]"\n'
    "For GPU speed on Windows, install the CUDA build of PyTorch first "
    "(https://pytorch.org/get-started/locally/)."
)

_cache: Dict[tuple, tuple] = {}


class MissingDependencyError(RuntimeError):
    """Raised when torch or transformers is not installed."""


def _verdict(score: float) -> str:
    if score >= LIKELY_AI:
        return "Likely AI-generated"
    if score >= UNSURE:
        return "Unclear - possibly AI-assisted"
    return "Likely human-written"


def _pick_device(torch, device: Optional[str]) -> str:
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load(device: Optional[str]):
    try:
        import torch
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel
        from transformers.utils import logging as hf_logging
    except ImportError as err:
        raise MissingDependencyError(_INSTALL_HINT) from err

    dev = _pick_device(torch, device)
    key = dev
    if key in _cache:
        return _cache[key]

    hf_logging.set_verbosity_error()
    repo = MODEL_ID
    print(f"stylematch: loading {repo} on {dev} (first run downloads the model)...",
          file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(repo)

    class _Desklib(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = AutoModel.from_config(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.post_init()

        def forward(self, input_ids, attention_mask):
            hidden = self.model(input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.classifier(pooled)

    model = _Desklib.from_pretrained(repo)
    model = model.eval().to(dev)
    _cache[key] = (tokenizer, model, dev, torch)
    return _cache[key]


def _chunks(text: str) -> List[str]:
    words = text.split()
    pieces = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)]
    # A tiny tail chunk is noisy, so fold it into the previous one.
    if len(pieces) > 1 and len(pieces[-1].split()) < CHUNK_WORDS // 3:
        tail = pieces.pop()
        pieces[-1] += " " + tail
    return pieces


@dataclass
class Detection:
    """What `detect` found out about a single text."""

    score: float
    verdict: str
    words: int
    model: str
    device: str
    chunk_scores: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ai_score": round(self.score, 4),
            "verdict": self.verdict,
            "words": self.words,
            "model": self.model,
            "device": self.device,
            "chunk_scores": [round(s, 4) for s in self.chunk_scores],
            "warnings": list(self.warnings),
        }


def detect(text: str, device: Optional[str] = None) -> Detection:
    """Score how AI-like a text is, from 0.0 to 1.0 (higher means more AI-like).

    The score is a raw model output, not a calibrated probability. Use the
    verdict, which applies tuned cutoffs.

    Long texts are split into chunks and the chunk scores are averaged.
    ``device`` may be ``"auto"`` (default), ``"cuda"`` or ``"cpu"``.
    """
    if not text or not text.strip():
        raise ValueError("There is no text to check.")

    tokenizer, net, dev, torch = _load(device)
    pieces = _chunks(text)
    scores: List[float] = []

    with torch.no_grad():
        for start in range(0, len(pieces), 8):
            batch = tokenizer(pieces[start:start + 8], return_tensors="pt", padding=True,
                              truncation=True, max_length=512).to(dev)
            logits = net(batch["input_ids"], batch["attention_mask"])
            scores += torch.sigmoid(logits).reshape(-1).tolist()

    # Weight each chunk by its length so a short tail cannot swing the result.
    weights = [len(p.split()) for p in pieces]
    score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    words = len(text.split())
    warnings = ["This is a statistical guess, not proof. Formal human writing can score "
                "high and edited AI text can score low."]
    if words < MIN_WORDS:
        warnings.insert(0, f"Only {words} words. Use at least {MIN_WORDS} for a meaningful reading.")
    return Detection(score, _verdict(score), words, MODEL_ID, dev, scores, warnings)
