# stylematch

[![tests](https://github.com/Ekaghni/stylematch/actions/workflows/ci.yml/badge.svg)](https://github.com/Ekaghni/stylematch/actions)
[![PyPI](https://img.shields.io/pypi/v/stylematch)](https://pypi.org/project/stylematch/)
[![Python](https://img.shields.io/pypi/pyversions/stylematch)](https://pypi.org/project/stylematch/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Find out whether a text looks AI-written.** stylematch runs a neural classifier on your text, on your GPU if you have one or on the CPU if you don't, and gives you a score plus a plain-English reading. It works from the terminal, from Python, or from a small desktop window.

It also has a second tool that compares the writing style of two texts, for the "do these sound like the same person?" question. More on that further down.

![The desktop app checking an AI-written essay and a passage by Mark Twain](https://raw.githubusercontent.com/Ekaghni/stylematch/main/assets/detector.png)

## Quick start

```bash
pip install "stylematch[ai]"
stylematch detect essay.txt
```

Two of the samples that ship in `examples/`:

```text
$ stylematch detect examples/ai_style.txt
AI score       : 1.000
Reading        : Likely AI-generated
Model / device : desklib/ai-text-detector-v1.01 on cuda
Words          : 116

$ stylematch detect examples/human_twain.txt
AI score       : 0.070
Reading        : Likely human-written
Model / device : desklib/ai-text-detector-v1.01 on cuda
Words          : 75
```

The first is a typical "AI essay" (furthermore, moreover, in conclusion). The second is the opening of *Huckleberry Finn*.

The first time you run it, the detection model (about 1.7 GB) gets downloaded and cached. After that it starts from disk.

## Read this first

The AI check is a statistical guess from a pretrained model, not proof. It is good at raw output from modern chatbots. Formal human writing, like encyclopedia text or old novels, can come out looking "AI", and AI text that someone has reworded often slips through. Please don't use a result on its own to accuse anyone of anything. The numbers are further down so you can judge for yourself.

## Install

You need Python 3.8 or newer.

```bash
pip install "stylematch[ai]"
```

That pulls in PyTorch and transformers, which the detector needs.

**GPU or no GPU:** the model uses your NVIDIA GPU if PyTorch can see one, and runs on the CPU if not. The scores are the same either way, the CPU is just slower. On Windows, `pip install torch` gives you the CPU build by default, so if you want the GPU, install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install stylematch. You can force a device with `--device cpu` or `--device cuda`.

If you only want the style comparison, plain `pip install stylematch` is enough. It has no dependencies and installs in a couple of seconds.

## Using the AI detector

### From the terminal

```bash
stylematch detect essay.txt
stylematch detect --string "paste some text here"
cat essay.txt | stylematch detect -
stylematch detect essay.txt --json
```

Texts under about 50 words get a warning. Long texts are split into chunks and the chunk scores are averaged.

### From Python

```python
from stylematch.detect import detect

result = detect(open("essay.txt").read())
print(result.score)     # 0.0 to 1.0, higher means more AI-like (not a true probability)
print(result.verdict)   # Likely human-written / Unclear / Likely AI-generated
print(result.device)    # cuda or cpu
```

### Desktop window

```bash
stylematch gui
```

Paste a text into one or both boxes and press **Detect AI**. The model loads in the background, so the window stays responsive. It uses tkinter, which comes with the standard Python installers on Windows and macOS. On Debian or Ubuntu you may need `sudo apt install python3-tk`.

### How the verdict is decided

The detector is [desklib/ai-text-detector-v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01), a DeBERTa-v3-large model fine-tuned to tell human text from LLM text (MIT licence, made by its authors, not me). I wrap it so it downloads itself, picks the GPU when there is one, handles long input, and applies cutoffs I tuned.

The model's own default cutoff of 0.5 turned out to flag far too much human writing, so the verdict uses these instead:

| AI score | Reading |
| --- | --- |
| 0.977 and above | Likely AI-generated |
| 0.907 to 0.976 | Unclear, possibly AI-assisted |
| below 0.907 | Likely human-written |

### How accurate is it?

I tested it on about 2,600 labelled texts: ChatGPT and human answers (HC3), a mix of many generators and domains (MAGE), GPT-4 text, GPT-4 text paraphrased to dodge detectors, and formal human writing from Wikipedia and Project Gutenberg. Half the data was used to pick the cutoffs. The numbers below come from the other half, which was never used for tuning. The scripts are in [`benchmarks/`](benchmarks/).

| Cutoff | Human text wrongly flagged | AI text caught |
| --- | --- | --- |
| 0.5 (the model's default) | roughly 6% to 30%, depending on the kind of text | about 95% to 100% |
| 0.907 (start of "unclear") | 5.2% | 89.8% |
| 0.977 (my "likely AI" line) | 2.0% | 80.7% |

By kind of text at the 0.977 cutoff: ChatGPT answers caught 97%, raw GPT-4 text 90%, mixed-generator MAGE text 72%, GPT-4 text that was paraphrased 67%. Human text wrongly flagged: Wikipedia 4%, MAGE human text 3%, Gutenberg 0%.

In plain terms: it is good at raw output from modern chatbots, decent on mixed generators, and weaker on AI text that someone has reworded. Roughly 1 in 50 human texts will still get flagged. These figures come from news, reviews, stories, Q&A and Wikipedia. Student essays, emails or technical writing may behave differently, so test on your own kind of text before relying on it.

## The second tool: compare writing style

Give it two texts and it tells you how alike they sound, as a score from 0 to 1.

```bash
stylematch first.txt second.txt
```

```text
$ stylematch examples/casual_a.txt examples/casual_b.txt
Similarity score : 0.6143
Reading          : Somewhat similar - possibly the same author, or related topics
Words            : 86 vs 91

$ stylematch examples/casual_a.txt examples/formal.txt
Similarity score : 0.4007
Reading          : Different - likely different authors or topics
Words            : 86 vs 68
```

The two casual texts are about different things (a cafe and a broken bike), yet they still score higher together than either does against the formal committee report. That is the idea working as intended.

You can also pass text straight in, or pipe from stdin with `-`, and add `--json` for scripts:

```bash
stylematch --text "first piece of writing here" "second piece here"
cat essay.txt | stylematch - other.txt
stylematch --json first.txt second.txt
```

In Python:

```python
from stylematch import compare, similarity

result = compare(open("a.txt").read(), open("b.txt").read())
print(result.score, result.verdict)
```

### How the comparison works

1. Lowercase both texts and collapse runs of whitespace.
2. Cut each one into overlapping chunks of 2, 3 and 4 characters.
3. Weight the chunks with TF-IDF, so chunks that only one text uses count for more.
4. Take the cosine similarity of the two weighted vectors.

Because the chunks are characters and not whole words, punctuation habits, favourite word endings and spacing all leave a mark on the score, while the topic matters less. This is a common starting point in stylometry.

The maths is plain Python, about 40 lines in [`src/stylematch/core.py`](src/stylematch/core.py). The test suite checks it against scikit-learn's `TfidfVectorizer` and the two agree to nine decimal places. I dropped scikit-learn as a dependency because it is a big download for a very small calculation.

| Score | Reading |
| --- | --- |
| 0.70 and above | Very similar, likely the same author |
| 0.50 to 0.69 | Somewhat similar, same author or related topics |
| below 0.50 | Different, likely different authors or topics |

These cutoffs are rules of thumb, not something I calibrated on a big dataset. Compare against a few texts you know are by the same person before trusting any single number.

## Limitations

- Short samples (a few sentences) give noisy results, for both tools.
- The AI check is trained mostly on English. Other languages are a gamble.
- Formal or very polished human writing is the classic false positive.
- Paraphrased or hand-edited AI text is often missed (about a third of paraphrased GPT-4 text slipped through in my test).
- It is a screening aid. Never use it alone to accuse someone of cheating or plagiarism.
- The style comparison is thrown off by shared topics or copied vocabulary, and someone imitating another person's style on purpose can fool it.

## Development

```bash
git clone https://github.com/Ekaghni/stylematch
cd stylematch
pip install -e ".[dev,ai]"
pytest -m "not slow"   # quick tests
pytest                 # also runs the model test (downloads the model)
```

This started as a small single-file tkinter project that only compared writing style. That first version is kept in [`legacy/`](legacy/) if you are curious how it began.

## License

MIT. See [LICENSE](LICENSE). The detection model has its own MIT licence from its authors.
