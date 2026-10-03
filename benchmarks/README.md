# Benchmark

How the AI-text thresholds in `stylematch.detect` were chosen.

1. `build_bench.py` builds about 2,600 labelled texts (40 to 350 words each) from public sources: HC3 (ChatGPT vs human answers), MAGE (many generators and domains, plus a GPT-4 set and a GPT-4 paraphrased set), and formal human writing from Wikipedia and Project Gutenberg. Download the CSV/JSONL files named at the top of the script into the same folder first.
2. `score_desklib.py` scores every text with the detector.
3. Half of each group is the dev split (used to pick cutoffs), half is the test split (used for the numbers in the main README).

Cutoffs are the dev-split quantiles of human scores at 5% and 2% false-positive rate. Nothing on the test split was used to tune anything.
