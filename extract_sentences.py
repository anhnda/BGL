"""
extract_sentences.py
=====================
Extract N sentences from a sentiment dataset (sst2 / imdb / rotten / vsfc /
vicomment), using the same HuggingFace loading and seeding as
eval_lime_sentiment.py, and write them out (one per line, or as JSONL with
labels). Drop-in source of sentences for lime_nlp.py --multi.

USAGE:
    python extract_sentences.py --dataset sst2 --n 30
    python extract_sentences.py --dataset imdb --n 100 --out imdb_100.txt
    python extract_sentences.py --dataset rotten --n 50 --jsonl --out rotten.jsonl
    python extract_sentences.py --dataset sst2 --n 30 --max-chars 300
    python extract_sentences.py --dataset vsfc --n 30        # Vietnamese (UIT-VSFC)
    python extract_sentences.py --dataset vicomment --n 50   # Vietnamese social comments

Without --out it just prints to stdout (numbered preview).

VIETNAMESE NOTE:
    `vsfc` -> uitnlp/vietnamese_students_feedback (sentiment label scheme:
              0=negative, 1=neutral, 2=positive). Pairs naturally with the
              ViSoBERT sentiment model in lime_nlp_k2.py (--model visobert),
              but BEWARE the label index differs from the model's head
              (model: 0=NEG, 1=POS, 2=NEU). Labels emitted here are the
              DATASET's scheme; remap if comparing against model argmax.
    `vicomment` -> minhtoan/vietnamese-comment-sentiment (social-media domain,
              closer to ViSoBERT's training distribution). String labels are
              mapped negative/neutral/positive -> 0/1/2.
"""
from __future__ import annotations
import argparse
import json
import random


# string-label -> int for datasets that ship textual sentiment labels
_VI_SENTIMENT_MAP = {
    "negative": 0, "neg": 0, "tiêu cực": 0,
    "neutral": 1, "neu": 1, "trung tính": 1, "trung lập": 1,
    "positive": 2, "pos": 2, "tích cực": 2,
}


def _coerce_label(lbl):
    """Datasets vary: some give ints, some strings. Normalise to int."""
    if isinstance(lbl, (int, bool)):
        return int(lbl)
    if isinstance(lbl, str):
        key = lbl.strip().lower()
        if key in _VI_SENTIMENT_MAP:
            return _VI_SENTIMENT_MAP[key]
        try:
            return int(key)
        except ValueError:
            raise ValueError(f"unrecognised sentiment label {lbl!r}")
    return int(lbl)


def _load_vsfc(load_dataset, split):
    """Load one split of UIT-VSFC without executing its loader script.

    The uitnlp/vietnamese_students_feedback repo ships a loading script, which
    `datasets` >= 4.0 refuses to run. We instead pull the Hub's auto-converted
    PARQUET mirror (published for every dataset under the refs/convert/parquet
    branch), which is plain data files -- no script, no trust_remote_code.

    Strategy, in order of preference:
      1. Resolve parquet URLs via the datasets-server /parquet API, then load
         them with the "parquet" builder.
      2. Hit the refs/convert/parquet branch directly over https.
      3. Fail with an actionable message (pin datasets<4.0, or load locally).

    `split` is one of train | validation | test.
    """
    repo = "uitnlp/vietnamese_students_feedback"

    # ---- 1. datasets-server parquet API (most reliable, script-free) --- #
    # Returns absolute resolve/refs%2Fconvert%2Fparquet/... URLs that the
    # "parquet" builder loads directly. This is the documented way to get the
    # Hub's auto-converted parquet for ANY dataset, no script involved.
    try:
        import requests
        api = ("https://datasets-server.huggingface.co/parquet"
               f"?dataset={repo}")
        info = requests.get(api, timeout=60).json()
        files = [f["url"] for f in info.get("parquet_files", [])
                 if f.get("split") == split]
        if files:
            return load_dataset("parquet", data_files=files, split="train")
    except Exception:
        pass

    # ---- 2. direct hf:// path to the convert branch -------------------- #
    # Fallback if the API is unreachable but the Hub is. The convert branch is
    # literally named "refs/convert/parquet"; url-encode the slash.
    rev = "refs%2Fconvert%2Fparquet"
    for fname in ("0000.parquet", "*.parquet"):
        url = (f"https://huggingface.co/datasets/{repo}/resolve/"
               f"{rev}/default/{split}/{fname}")
        try:
            return load_dataset("parquet", data_files=url, split="train")
        except Exception:
            continue

    # ---- 3. give up with an actionable message ------------------------- #
    raise RuntimeError(
        "Could not load UIT-VSFC without its loader script. The parquet "
        "mirror could not be reached (check network / proxy allow-list for "
        "huggingface.co and datasets-server.huggingface.co). Options: "
        "(a) `pip install \"datasets<4.0\"` to run the legacy script, or "
        "(b) download the parquet/CSV manually and point --dataset at a local "
        "file."
    )


def load_sentences(dataset, n, seed=42, shuffle=True, split=None,
                   max_chars=None, min_chars=1):
    """Return up to n (text, label) pairs from the chosen dataset.

    Mirrors eval_lime_sentiment.py: same dataset ids, same fields, seed=42,
    random.sample/shuffle for the same selection behaviour.
    """
    from datasets import load_dataset
    random.seed(seed)

    if dataset == "sst2":
        ds = load_dataset("glue", "sst2")[split or "test"]
        pairs = list(zip(ds["sentence"], ds["label"]))
    elif dataset == "imdb":
        ds = load_dataset("imdb")[split or "test"]
        pairs = list(zip(ds["text"], ds["label"]))
    elif dataset == "rotten":
        ds = load_dataset("rotten_tomatoes")[split or "test"]
        pairs = list(zip(ds["text"], ds["label"]))
    elif dataset == "vsfc":
        # UIT-VSFC Vietnamese Students' Feedback Corpus.
        # Fields: "sentence", "sentiment" (0=neg,1=neu,2=pos), "topic".
        #
        # NOTE: the uitnlp/vietnamese_students_feedback repo ships a loader
        # SCRIPT (vietnamese_students_feedback.py), which `datasets` >= 4.0
        # refuses to run ("Dataset scripts are no longer supported"). We avoid
        # the script entirely by loading the Hub's auto-converted PARQUET
        # mirror (the dataset-viewer publishes one for every dataset under the
        # refs/convert/parquet branch). This needs no trust_remote_code and
        # works on datasets 4.x / 5.x. If the parquet mirror is unavailable we
        # fall back to an older datasets that can still run scripts.
        ds = _load_vsfc(load_dataset, split or "test")
        labels = ds["sentiment"] if "sentiment" in ds.column_names else ds["label"]
        pairs = list(zip(ds["sentence"], labels))
    elif dataset == "vicomment":
        # minhtoan/vietnamese-comment-sentiment (social-media domain).
        # Only a "train" split ships; default to it. Text col: "Content".
        ds = load_dataset("minhtoan/vietnamese-comment-sentiment")[split or "train"]
        text_col = ("Content" if "Content" in ds.column_names
                    else ("content" if "content" in ds.column_names
                          else ds.column_names[0]))
        lab_col = ("Sentiment" if "Sentiment" in ds.column_names
                   else ("sentiment" if "sentiment" in ds.column_names
                         else "label"))
        pairs = list(zip(ds[text_col], ds[lab_col]))
    else:
        raise ValueError(f"unknown dataset {dataset!r} "
                         f"(choose sst2 | imdb | rotten | vsfc | vicomment)")

    # normalise labels (some Vietnamese sets use string sentiment labels)
    pairs = [(t, _coerce_label(lbl)) for t, lbl in pairs]

    # length filtering (e.g. drop empty rows or trim huge imdb reviews)
    def ok(t):
        t = t.strip()
        if len(t) < min_chars:
            return False
        if max_chars is not None and len(t) > max_chars:
            return False
        return True

    pairs = [(t.strip(), int(lbl)) for t, lbl in pairs if ok(t)]

    if shuffle:
        random.shuffle(pairs)
    return pairs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sst2",
                    choices=["sst2", "imdb", "rotten", "vsfc", "vicomment"])
    ap.add_argument("--n", type=int, default=30,
                    help="number of sentences to extract")
    ap.add_argument("--split", default=None,
                    help="dataset split (default: test; vicomment defaults to train)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true",
                    help="take the first n rows in order instead of shuffling")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="drop sentences longer than this many characters")
    ap.add_argument("--min-chars", type=int, default=1,
                    help="drop sentences shorter than this many characters")
    ap.add_argument("--out", default=None,
                    help="output path; if omitted, print a preview to stdout")
    ap.add_argument("--jsonl", action="store_true",
                    help="write JSONL {\"text\":..., \"label\":...} instead of "
                         "plain text (one sentence per line)")
    args = ap.parse_args()

    pairs = load_sentences(
        args.dataset, args.n, seed=args.seed,
        shuffle=not args.no_shuffle, split=args.split,
        max_chars=args.max_chars, min_chars=args.min_chars,
    )

    if not pairs:
        print("no sentences matched the filters.")
        return

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for text, label in pairs:
                if args.jsonl:
                    f.write(json.dumps({"text": text, "label": label},
                                       ensure_ascii=False) + "\n")
                else:
                    # collapse newlines so each sentence stays on one line
                    f.write(text.replace("\n", " ").replace("\r", " ") + "\n")
        print(f"wrote {len(pairs)} sentences from {args.dataset} -> {args.out}")
    else:
        for i, (text, label) in enumerate(pairs):
            preview = text if len(text) <= 120 else text[:117] + "..."
            print(f"[{i:>3d}] (label={label}) {preview}")
        print(f"\n{len(pairs)} sentences from {args.dataset} "
              f"(seed={args.seed}, shuffle={not args.no_shuffle})")


if __name__ == "__main__":
    main()