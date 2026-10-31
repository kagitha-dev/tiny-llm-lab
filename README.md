# Tiny LLM Lab

A language model small enough to see every weight.

Real LLMs have billions of weights. This one has about 100, so you can watch it learn, open its weights file, train it again, and see exactly how it picks the next token. The steps are the same ones every LLM goes through:

1. **Train**: start from random weights and nudge them so the real next token gets a higher score
2. **Save**: write the learned weights to a file
3. **Freeze and predict**: load the file, turn scores into chances (softmax), pick a token, repeat

Everything runs in the browser. No server, no install, no AI library: the training math is about ten lines of plain code.

## Pages

| Page | What it shows |
|---|---|
| `index.html`: Tiny LLM Lab | Tokens and IDs, training with a live weight grid, saving and loading a weights file, prediction with temperature and top-k, and a panel that explains every step with the real numbers |
| `tokens.html`: Tokens and embeddings | A word tokenizer next to a subword (BPE) tokenizer, the embedding table lookup, and a 2D map where embeddings start random and similar tokens drift together as they train |

Open `index.html` in a browser to start. Both pages are self-contained HTML files.

## The Python version

`tiny_llm.py` is the same model in Python (numpy only). It writes a real weights file you can open.

```bash
pip install numpy                     # optional: pip install safetensors
python tiny_llm.py train data.txt --rounds 50
python tiny_llm.py show               # print the weight grid
python tiny_llm.py predict "the dog"  # dog -> sat 96%
python tiny_llm.py train data2.txt --rounds 50   # data2 adds "the dog ate fish ."
python tiny_llm.py predict "the dog"  # dog -> ate 55%, sat 42%
python tiny_llm.py generate "the" --words 5
python tiny_llm.py reset
```

Training saves `tiny_model.json`. The browser lab reads and writes the same format, so a model trained in Python can be loaded with **Load a model file** in Step 2. If `safetensors` is installed, the script also writes `tiny_model.safetensors`, the format Hugging Face uses for real models (this one is 608 bytes instead of gigabytes).

## How it works

**Tokenizer.** Text is split into tokens and each different token gets an ID. The lab uses whole words; real LLMs use subword pieces learned with byte-pair encoding (see `tokens.html`). The tokenizer has no weights.

**Model.** One grid of weights: a row for each token the model sees, a column for each token that could come next. To predict after "dog", it takes the "dog" row and applies softmax: `chance = e^(score / temperature) / sum of all e^(score / temperature)`.

**Training.** For every pair in the text (for example `dog → sat`), the model guesses with softmax and then moves each weight in that row by `learning_rate × (should_be − got)`. The right token goes up, the others go down. One pass over all pairs is one round.

**How a real LLM differs.** Real models use subword tokens, look at the whole input instead of only the last token, and have billions of weights: an embedding table plus many layers that compute the scores. The output step (scores, softmax, pick one token) and the training rule (guess, measure the error, nudge the weights) are the same.

## Files

| File | Purpose |
|---|---|
| `index.html` | Tiny LLM Lab |
| `tokens.html` | Tokens and embeddings |
| `tiny_llm.py` | Python version: train, save, predict, generate |
| `data.txt`, `data2.txt` | Sample training text (`data2.txt` adds one sentence) |

A simplified model for understanding, not a production language model.
