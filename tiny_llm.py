"""
Tiny LLM: the smallest possible "next word predictor" neural network.

Same idea as a real LLM, shrunk until you can read every weight:
  - train    : read sentences, nudge weights so the real next word scores higher
  - save     : weights are written to a file (tiny_model.json, plus
               tiny_model.safetensors if the `safetensors` package is installed,
               the same format Hugging Face uses for real models)
  - predict  : load the frozen weights, score every word, softmax -> %, pick one

Usage:
  python tiny_llm.py train data.txt --rounds 50     # train (or continue training) and save
  python tiny_llm.py show                           # print the weight grid
  python tiny_llm.py predict "the dog"              # next-word probabilities
  python tiny_llm.py generate "the" --words 6       # predict, append, repeat
  python tiny_llm.py reset                          # delete the saved model

Needs only numpy:  pip install numpy   (optional: pip install safetensors)
"""
import argparse
import json
import os
import re
import sys

import numpy as np

MODEL_FILE = "tiny_model.json"
SAFETENSORS_FILE = "tiny_model.safetensors"
LEARNING_RATE = 0.5


def tokenize(text):
    return re.sub(r"([.,!?])", r" \1 ", text.lower()).split()


def softmax(row):
    e = np.exp(row - row.max())
    return e / e.sum()


def load_model():
    if not os.path.exists(MODEL_FILE):
        return [], np.zeros((0, 0)), 0
    with open(MODEL_FILE) as f:
        m = json.load(f)
    return m["vocab"], np.array(m["weights"], dtype=np.float64), m["rounds"]


def save_model(vocab, W, rounds):
    with open(MODEL_FILE, "w") as f:
        rows = ",\n".join("    " + json.dumps([round(float(v), 4) for v in r]) for r in W)
        f.write('{\n  "vocab": %s,\n  "rounds": %d,\n  "weights": [\n%s\n  ]\n}\n'
                % (json.dumps(vocab), rounds, rows))
    saved = [MODEL_FILE]
    try:
        from safetensors.numpy import save_file
        save_file({"weights": W.astype(np.float32)}, SAFETENSORS_FILE,
                  metadata={"vocab": json.dumps(vocab), "rounds": str(rounds)})
        saved.append(SAFETENSORS_FILE)
    except ImportError:
        pass
    return saved


def grow_vocab(vocab, W, tokens, rng):
    """New words get new rows/columns of small random numbers; old weights are kept."""
    new_words = [w for w in dict.fromkeys(tokens) if w not in vocab]
    if not new_words:
        return vocab, W, []
    vocab2 = vocab + new_words
    W2 = rng.uniform(-0.5, 0.5, (len(vocab2), len(vocab2)))
    W2[: len(vocab), : len(vocab)] = W
    return vocab2, W2, new_words


def print_grid(vocab, W, highlight=None):
    width = max(6, max(len(w) for w in vocab) + 1)
    print(" " * width + "".join(f"{w[:6]:>7}" for w in vocab) + "   <- next word")
    for i, w in enumerate(vocab):
        mark = " <" if highlight == w else ""
        print(f"{w:>{width}}" + "".join(f"{v:7.2f}" for v in W[i]) + mark)
    print(f"(rows = current word, {W.size} weights in total)")


def cmd_train(args):
    with open(args.file) as f:
        tokens = tokenize(f.read())
    if len(tokens) < 2:
        sys.exit("Need at least two words of training text.")
    vocab, W, rounds = load_model()
    rng = np.random.default_rng(7)
    vocab, W, new_words = grow_vocab(vocab, W, tokens, rng)
    if new_words:
        print("New words added with random weights:", ", ".join(new_words))
    pairs = [(vocab.index(a), vocab.index(b)) for a, b in zip(tokens, tokens[1:])]
    before = W.copy()

    for r in range(1, args.rounds + 1):
        loss = 0.0
        for x, y in pairs:
            p = softmax(W[x])
            loss -= np.log(p[y])
            grad = p.copy()
            grad[y] -= 1.0              # push the true next word up, everything else down
            W[x] -= LEARNING_RATE * grad
        if r == 1 or r % 10 == 0 or r == args.rounds:
            print(f"round {rounds + r:4d}   error (loss) = {loss / len(pairs):.3f}")
    rounds += args.rounds

    diff = W - before
    top = np.dstack(np.unravel_index(np.argsort(-np.abs(diff), axis=None)[:5], W.shape))[0]
    print("\nBiggest weight changes:")
    for i, j in top:
        print(f"  {vocab[i]:>6} -> {vocab[j]:<6} {before[i, j]:6.2f} -> {W[i, j]:6.2f}")
    print("\nSaved:", ", ".join(save_model(vocab, W, rounds)))


def cmd_show(_):
    vocab, W, rounds = load_model()
    if not vocab:
        sys.exit("No model yet. Run: python tiny_llm.py train data.txt")
    print(f"Model after {rounds} training rounds\n")
    print_grid(vocab, W)


def predict_next(vocab, W, text):
    tokens = tokenize(text)
    if not tokens or tokens[-1] not in vocab:
        sys.exit(f'"{tokens[-1] if tokens else ""}" was never in the training text.')
    i = vocab.index(tokens[-1])
    return tokens[-1], W[i], softmax(W[i])


def cmd_predict(args):
    vocab, W, _ = load_model()
    last, row, p = predict_next(vocab, W, args.text)
    print(f'Looking at last word "{last}". Its weights (raw scores):')
    print("  " + "  ".join(f"{w}:{v:.2f}" for w, v in zip(vocab, row)))
    print("\nAfter softmax (chance of each next word):")
    for j in np.argsort(-p)[:6]:
        print(f"  {vocab[j]:>6}  {p[j] * 100:5.1f}%  " + "#" * int(p[j] * 40))


def cmd_generate(args):
    vocab, W, _ = load_model()
    text = args.text
    rng = np.random.default_rng()
    for _ in range(args.words):
        _, _, p = predict_next(vocab, W, text)
        j = int(rng.choice(len(vocab), p=p)) if args.random else int(np.argmax(p))
        text += " " + vocab[j]
        print(text)


def cmd_reset(_):
    for f in (MODEL_FILE, SAFETENSORS_FILE):
        if os.path.exists(f):
            os.remove(f)
    print("Model deleted. Next train starts from random weights.")


def main():
    ap = argparse.ArgumentParser(description="Tiny next-word predictor you can read.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("file"); t.add_argument("--rounds", type=int, default=50)
    sub.add_parser("show")
    p = sub.add_parser("predict"); p.add_argument("text")
    g = sub.add_parser("generate"); g.add_argument("text"); g.add_argument("--words", type=int, default=6)
    g.add_argument("--random", action="store_true", help="weighted random pick instead of top word")
    sub.add_parser("reset")
    args = ap.parse_args()
    {"train": cmd_train, "show": cmd_show, "predict": cmd_predict,
     "generate": cmd_generate, "reset": cmd_reset}[args.cmd](args)


if __name__ == "__main__":
    main()
