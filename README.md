# Tiny LLM Lab

A language model small enough to see every weight.

Real LLMs have billions of weights. This one has about 100, so you can watch it learn, open its weights file, train it again, and see exactly how it picks the next token. The steps are the same ones every LLM goes through:

1. **Train**: start from random weights and nudge them so the real next token gets a higher score
2. **Save**: write the learned weights to a file
3. **Freeze and predict**: load the file, turn scores into chances (softmax), pick a token, repeat

Everything runs in the browser. No server, no install, no AI library: the training math is about ten lines of plain code.
# Experience: https://kagitha-dev.github.io/tiny-llm-lab/index.html
## Pages

The five labs build on each other:

| Lab | What it shows |
|---|---|
| `index.html`: 1 · Tiny LLM Lab | What a model does. Train a model on any text and watch its weight grid change, save the weights to a file, then freeze them and predict the next token with temperature and top-k. A side panel explains every step with the real numbers. |
| `tokens.html`: 2 · Text to numbers | How text reaches the weights. Your sentence goes through six steps: characters, clean-up, subword tokens (byte-pair encoding), IDs, a row read from the embedding weights, and the input grid. Also shows GPT-4o's real tokenizer on the same text, and a map where embeddings are learned. |
| `transformer.html`: 3 · Transformer | How it all connects. A real one-block transformer trained in the browser. Follow a prompt through every block (embeddings, positions, attention, feed-forward, output, softmax), each tagged with the lab that covered it, and see where it beats the last-token-only model. |
| `weights.html`: 4 · How a weight is made | How one weight is made and used, from the sentence "My puppy name is River. River likes Ruby." The number for "my → puppy" starts random; each round the model checks the chance it gave "puppy", sees how much is missing to 100%, and adds half of that (like salting a curry: taste, see how far off, add a pinch). Then the number is saved, and a prompt reads it to predict the next word. |
| `ask.html`: 5 · Ask a question | Why an LLM can answer “What is my puppy name?” when Lab 1 can’t. Same weight grid and steps as Lab 1, but it reads **every word** of the input: the rows of all input tokens are added and the total goes through softmax. Trained on “My puppy name is River. River likes Ruby.”, it answers *river*. Works with any text (it starts from the text used in Lab 1 and suggests questions made from it). |

Open `index.html` in a browser to start. Each page is a single self-contained HTML file; Lab 2 loads the GPT-4o tokenizer from a CDN.

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
| `ask.html` | Ask a question (reads the whole input, not only the last token) |
| `tokens.html` | Tokens and embeddings |
| `tiny_llm.py` | Python version: train, save, predict, generate |
| `data.txt`, `data2.txt` | Sample training text (`data2.txt` adds one sentence) |

A simplified model for understanding, not a production language model.
