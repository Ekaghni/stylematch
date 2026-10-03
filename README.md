# stylematch

[![tests](https://github.com/Ekaghni/stylematch/actions/workflows/ci.yml/badge.svg)](https://github.com/Ekaghni/stylematch/actions)
[![PyPI](https://img.shields.io/pypi/v/stylematch)](https://pypi.org/project/stylematch/)
[![Python](https://img.shields.io/pypi/pyversions/stylematch)](https://pypi.org/project/stylematch/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

stylematch does two jobs:

1. **Style comparison.** Give it two pieces of writing and it tells you how alike they sound, as a score from 0 to 1.
2. **AI-text check.** Give it one piece of writing and a neural classifier estimates the chance that a machine wrote it.

I started this as a small side project to see whether character patterns could say something about who wrote a text. It grew into a proper package, so now you can install it with one command and use it from the terminal, from Python, or from a small desktop window.

![The desktop window comparing two texts](https://raw.githubusercontent.com/Ekaghni/stylematch/main/assets/gui.png)

## Read this first

Both features give you a hint, not a verdict.

- The **comparison** score is a rough measure of stylistic overlap. It can't prove two texts share an author.
- The **AI check** is a statistical guess from a pretrained model. In my own tests it handled a clearly machine-written essay and a passage of Mark Twain well, but formal human writing like encyclopedia text or old novels can come out looking "AI", and lightly edited AI text can come out looking human. Don't use either result to accuse someone of anything.

## Install

You need Python 3.8 or newer.

```bash
pip install stylematch
```

That gives you the style comparison. It is pure Python with no dependencies, so it installs in a couple of seconds and runs anywhere.

For the AI check, add the extras:

```bash
pip install "stylematch[ai]"
```

This pulls in PyTorch and transformers. The first time you run a check it downloads the detection model (about 1.7 GB) and caches it.

**GPU or no GPU:** the model uses your NVIDIA GPU if PyTorch can see one, and runs on the CPU if not. The scores are the same either way, the CPU is just slower. Speed depends on your hardware, and the very first run also downloads the model. On Windows, `pip install torch` gives you the CPU build by default, so if you want the GPU, install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install stylematch. You can force a device with `--device cpu` or `--device cuda`.

## Quick start

### From the terminal

```bash
stylematch first.txt second.txt
```

Using the two casual samples that ship in `examples/`:

```text
$ stylematch examples/casual_a.txt examples/casual_b.txt
Similarity score : 0.6143
Reading          : Somewhat similar - possibly the same author, or related topics
Words            : 86 vs 91
```

And a casual text against a stiff committee report:

```text
$ stylematch examples/casual_a.txt examples/formal.txt
Similarity score : 0.4007
Reading          : Different - likely different authors or topics
Words            : 86 vs 68
```

The two casual texts are about different things (a cafe and a broken bike), yet they still score higher together than either does against the formal one. That is the idea working as intended.

You can also pass text straight in, or pipe from stdin with `-`:

```bash
stylematch --text "first piece of writing here" "second piece here"
cat essay.txt | stylematch - other.txt
```

Add `--json` if you want to feed the result into another script:

```text
$ stylematch --json --text "I love long walks on the beach, honestly." "Honestly, I love a long walk by the sea."
{
  "score": 0.4701,
  "verdict": "Different - likely different authors or topics",
  "words_1": 8,
  "words_2": 9,
  "warnings": [
    "Texts should have at least 10 words each. Short samples give unreliable scores."
  ]
}
```

Notice the warning. Short samples give shaky scores, and the tool will say so.

### Checking for AI-written text

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

The first is a typical "AI essay" (furthermore, moreover, in conclusion) and the second is the opening of *Huckleberry Finn*. Texts under about 50 words get a warning, and long texts are split into chunks whose scores are averaged. Add `--json` for machine-readable output, or `--string "your text"` instead of a file.

### From Python

```python
from stylematch import compare, similarity

a = open("examples/casual_a.txt").read()
b = open("examples/formal.txt").read()

result = compare(a, b)
print(result.score)     # 0.4007
print(result.verdict)   # Different - likely different authors or topics

# or, if you only want the number
print(similarity(a, b))
```

And for the AI check (needs the `ai` extra):

```python
from stylematch.detect import detect

result = detect(open("essay.txt").read())
print(result.score)     # 0.0 to 1.0, higher means more AI-like (not a true probability)
print(result.verdict)   # Likely human-written / Unclear / Likely AI-generated
```

### Desktop window

```bash
stylematch gui
```

Paste a text in each box and press Compare for the style score, or Detect AI to check each box on its own. It uses tkinter, which comes with the standard Python installers on Windows and macOS. On Debian or Ubuntu you may need `sudo apt install python3-tk`. If tkinter is missing, the terminal version still works.

## How the style comparison works

1. Lowercase both texts and collapse runs of whitespace.
2. Cut each one into overlapping chunks of 2, 3 and 4 characters.
3. Weight the chunks with TF-IDF, so chunks that only one text uses count for more.
4. Take the cosine similarity of the two weighted vectors.

Because the chunks are characters and not whole words, things like punctuation habits, favourite word endings and spacing all leave a mark on the score, while the topic matters less. This is a very common starting point in stylometry.

The maths is written in plain Python, about 40 lines of it in [`src/stylematch/core.py`](src/stylematch/core.py). The test suite checks the output against scikit-learn's `TfidfVectorizer`, and the two agree to nine decimal places. I dropped scikit-learn as a dependency because it is a big download for a very small calculation.

### How the AI check works

It runs [desklib/ai-text-detector-v1.01](https://huggingface.co/desklib/ai-text-detector-v1.01), a DeBERTa-v3-large model fine-tuned to tell human text from LLM text (MIT licence, not my work). I wrap it so it downloads itself, picks the GPU when there is one, splits long input into chunks, and averages the results.

The raw score is not a probability, and the model's default 0.5 cutoff turned out to be far too jumpy (see below). So the verdict uses cutoffs I tuned myself:

| AI score | Reading |
| --- | --- |
| 0.977 and above | Likely AI-generated |
| 0.907 to 0.976 | Unclear, possibly AI-assisted |
| below 0.907 | Likely human-written |

### How accurate is it?

I tested it on about 2,600 labelled texts: ChatGPT and human answers (HC3), a mix of many generators and domains (MAGE), GPT-4 text, GPT-4 text paraphrased to dodge detectors, and formal human writing from Wikipedia and Gutenberg. Half the data was used to pick the cutoffs, and the numbers below are from the other half, which was never used for tuning. The scripts are in [`benchmarks/`](benchmarks/).

| Cutoff | Human text wrongly flagged | AI text caught |
| --- | --- | --- |
| 0.5 (the model's default) | roughly 6% to 30%, depending on the kind of text | about 95% to 100% |
| 0.907 (start of "unclear") | 5.2% | 89.8% |
| 0.977 (my "likely AI" line) | 2.0% | 80.7% |

By kind of text at the 0.977 cutoff: ChatGPT answers caught 97%, raw GPT-4 text 90%, mixed-generator MAGE text 72%, GPT-4 text that was paraphrased 67%. Human text wrongly flagged: Wikipedia 4%, MAGE human text 3%, Gutenberg 0%.

What this means in plain terms: it is good at raw output from modern chatbots, decent on mixed generators, and weaker on AI text that someone has reworded. Roughly 1 in 50 human texts will still get flagged. These figures come from news, reviews, stories, Q&A and Wikipedia. Student essays, emails or technical writing may behave differently, so test on your own kind of text before relying on it.

### Reading the style score

| Score | Reading |
| --- | --- |
| 0.70 and above | Very similar, likely the same author |
| 0.50 to 0.69 | Somewhat similar, same author or related topics |
| below 0.50 | Different, likely different authors or topics |

These cut-offs are rules of thumb, not something I calibrated on a big dataset. Your own texts may sit higher or lower. Compare against a few texts you know are by the same person before trusting any single number.

## Limitations

- Short samples (a few sentences) give noisy results, for both features.
- The AI check is trained mostly on English. Other languages are a gamble.
- Formal or very polished human writing is the classic false positive.
- Paraphrased or hand-edited AI text is often missed (about a third of paraphrased GPT-4 text slipped through in my test).
- Never use the result alone to accuse someone of cheating or plagiarism. It is a screening aid.
- The style comparison works on two texts at a time and is thrown off by shared topics or copied vocabulary.
- Someone imitating another person's style on purpose can fool it.

## Development

```bash
git clone https://github.com/Ekaghni/stylematch
cd stylematch
pip install -e ".[dev,ai]"
pytest -m "not slow"   # quick tests
pytest                 # also runs the model test (downloads the model)
```

The original single-file tkinter version I wrote while learning is kept in [`legacy/`](legacy/) if you are curious how it began.

## License

MIT. See [LICENSE](LICENSE). The detection model has its own MIT licence from its authors.
