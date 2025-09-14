"""
Profile a JSONL documents file:
- Unique 'src' dataset names
- Counts per 'src'
- Field presence across docs
- Basic length stats (chars/words) for common text fields
- Per-src content length stats (chars/words)
- Optional token stats using the project tokenizer
- Exact duplicate detection by normalized content hash
- TF-IDF top keywords per src

Outputs are saved under outputs/reports/data_profile/<timestamp>/ for reuse,
and a concise summary is printed to stdout.

Usage (via Poetry):
    poetry run python scripts/data/profile_documents.py --file_path data/documents.jsonl

Optional args:
    --out_dir outputs/reports/data_profile
    --max_preview 30      # limit of unique src printed inline
    --save 1              # whether to save report files (default: 1)
    --token_stats 0       # compute token counts using project tokenizer
    --keywords_top_k 20   # number of top tf-idf terms per src
    --min_df 2            # min_df for TfidfVectorizer
    --max_features 20000  # max features for TfidfVectorizer
"""
from __future__ import annotations

import os
import sys
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse as sps
import hashlib
import fire


def _add_src_to_path() -> None:
    scripts_dir = Path(__file__).resolve().parent
    repo_dir = scripts_dir.parent.parent
    src_dir = repo_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Thin wrapper around project util with graceful fallback."""
    _add_src_to_path()
    try:
        from ir_core.utils.core import read_jsonl  # type: ignore

        return read_jsonl(path)  # type: ignore[return-value]
    except Exception:
        # Fallback minimal reader
        def _gen() -> Iterable[Dict[str, Any]]:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

        return _gen()


def _summarize(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    arr = np.array(values)
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def profile(
    file_path: str = "data/documents.jsonl",
    out_dir: str = "outputs/reports/data_profile",
    max_preview: int = 30,
    save: bool = True,
    token_stats: bool = False,
    keywords_top_k: int = 20,
    min_df: int = 2,
    max_features: int = 20000,
    stopwords_top_n: int = 200,
    per_src_stopwords_top_n: int = 50,
    near_dup: bool = True,
    near_dup_hamming: int = 3,
    embedding_health: bool = False,
    embedding_outlier_threshold: float = 3.0,
) -> Dict[str, Any]:
    start = time.time()
    file_path = str(file_path)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    out_time = time.strftime("%Y%m%d_%H%M%S")
    out_dir_ts = out_root / out_time
    if save:
        out_dir_ts.mkdir(parents=True, exist_ok=True)

    unique_src: set[str] = set()
    src_counts: Counter[str] = Counter()
    field_presence: Counter[str] = Counter()
    string_field_chars: defaultdict[str, List[int]] = defaultdict(list)
    string_field_words: defaultdict[str, List[int]] = defaultdict(list)
    # per-src content length stats
    per_src_chars: defaultdict[str, List[int]] = defaultdict(list)
    per_src_words: defaultdict[str, List[int]] = defaultdict(list)
    # duplicate detection
    duplicate_map: defaultdict[str, List[str]] = defaultdict(list)  # hash -> [docid]
    # collect for tf-idf keywords
    texts: List[str] = []
    text_srcs: List[str] = []
    # optional token stats
    token_counts_overall: List[int] = []
    token_counts_per_src: defaultdict[str, List[int]] = defaultdict(list)
    tokenizer = None
    # embedding health checks
    embedding_norms: List[float] = []
    embedding_outliers: List[Dict[str, Any]] = []
    embeddings_computed: List[np.ndarray] = []

    n_docs = 0
    for doc in _read_jsonl(file_path):
        n_docs += 1
        # Track field presence
        for k, v in doc.items():
            field_presence[k] += 1
            if isinstance(v, str):
                string_field_chars[k].append(len(v))
                string_field_words[k].append(len(v.split()))

        # Handle 'src' or 'source'
        src = doc.get("src")
        if not src:
            src = doc.get("source")
        if isinstance(src, str):
            unique_src.add(src)
            src_counts[src] += 1

        # Per-src content length and TF-IDF collection
        content = doc.get("content", "")
        if isinstance(content, str) and content:
            cchars = len(content)
            cwords = len(content.split())
            if isinstance(src, str):
                per_src_chars[src].append(cchars)
                per_src_words[src].append(cwords)
                text_srcs.append(src)
            texts.append(content)

            # Duplicate detection (normalize whitespace and lowercase)
            norm = " ".join(content.split()).lower()
            key = hashlib.sha1(norm.encode("utf-8")).hexdigest()
            docid = str(doc.get("docid", f"idx:{n_docs-1}"))
            duplicate_map[key].append(docid)

            # Optional token stats
            if token_stats:
                if tokenizer is None:
                    # lazy-load tokenizer
                    try:
                        from ir_core.embeddings.core import load_model  # type: ignore

                        tokenizer, _ = load_model()
                    except Exception:
                        tokenizer = False  # sentinel for failure
                if tokenizer and hasattr(tokenizer, "encode"):
                    try:
                        token_len = len(tokenizer.encode(content))
                        token_counts_overall.append(token_len)
                        if isinstance(src, str):
                            token_counts_per_src[src].append(token_len)
                    except Exception:
                        pass

            # Optional embedding health checks
            if embedding_health and len(content) > 10:  # Skip very short content
                try:
                    from ir_core.embeddings.core import encode_texts  # type: ignore
                    
                    emb = encode_texts([content])[0]
                    if emb is not None and len(emb) > 0:
                        norm = float(np.linalg.norm(emb))
                        embedding_norms.append(norm)
                        embeddings_computed.append(emb)
                        
                        # Check for outliers (very high or low norms)
                        if len(embedding_norms) > 10:  # Need some baseline
                            mean_norm = np.mean(embedding_norms)
                            std_norm = np.std(embedding_norms)
                            if abs(norm - mean_norm) > embedding_outlier_threshold * std_norm:
                                embedding_outliers.append({
                                    "docid": docid,
                                    "src": src,
                                    "norm": norm,
                                    "z_score": (norm - mean_norm) / std_norm if std_norm > 0 else 0,
                                    "content_preview": content[:100] + "..." if len(content) > 100 else content
                                })
                except Exception:
                    pass  # Graceful failure for embedding health checks

    # Summaries
    field_length_stats: Dict[str, Dict[str, Any]] = {}
    for f in sorted(string_field_chars.keys() | string_field_words.keys()):
        field_length_stats[f] = {
            "chars": _summarize(string_field_chars.get(f, [])),
            "words": _summarize(string_field_words.get(f, [])),
        }

    per_src_length_stats: Dict[str, Dict[str, Any]] = {}
    for s in sorted(unique_src):
        per_src_length_stats[s] = {
            "content_chars": _summarize(per_src_chars.get(s, [])),
            "content_words": _summarize(per_src_words.get(s, [])),
        }

    # Duplicate groups (>1 only)
    duplicate_groups = [
        {"hash": h, "count": len(ids), "docids": ids[:50]}  # cap sample
        for h, ids in duplicate_map.items()
        if len(ids) > 1
    ]

    # TF-IDF keywords per src and global/per-src stopwords
    keywords_per_src: Dict[str, List[Tuple[str, float]]] = {}
    stopwords_global: List[str] = []
    per_src_stopwords: Dict[str, List[str]] = {}
    if texts:
        try:
            vectorizer = TfidfVectorizer(
                min_df=min_df, max_features=max_features, ngram_range=(1, 2)
            )
            X = vectorizer.fit_transform(texts)
            X_csr: sps.csr_matrix = sps.csr_matrix(X)
            n_features_tfidf = int(getattr(X, "shape")[1])
            vocab = np.array(vectorizer.get_feature_names_out())
            # Global stopwords from lowest IDF (most common)
            try:
                idf = vectorizer.idf_
                low_idx = np.argsort(idf)[:stopwords_top_n]
                stopwords_global = [str(vocab[i]) for i in low_idx]
            except Exception:
                stopwords_global = []
            # aggregate per src by summing rows belonging to that src
            from collections import defaultdict as _dd

            rows_by_src: Dict[str, List[int]] = _dd(list)
            for i, s in enumerate(text_srcs):
                rows_by_src[s].append(i)

            for s, idxs in rows_by_src.items():
                # sum sparse rows via CSR row slicing
                if not idxs:
                    sub = sps.csr_matrix((0, n_features_tfidf))
                else:
                    import numpy as _np
                    sub = X_csr[_np.array(idxs), :]
                weights = np.asarray(sub.sum(axis=0)).ravel()
                if weights.size:
                    top_idx = np.argsort(weights)[::-1][:keywords_top_k]
                    terms = [(str(vocab[j]), float(weights[j])) for j in top_idx if weights[j] > 0]
                    keywords_per_src[s] = terms
        except Exception:
            pass

    # Per-src stopwords using raw counts (CountVectorizer)
    if texts:
        try:
            cvec = CountVectorizer(min_df=min_df, max_features=max_features, ngram_range=(1, 2))
            CX = cvec.fit_transform(texts)
            CX_csr: sps.csr_matrix = sps.csr_matrix(CX)
            n_features_count = int(getattr(CX, "shape")[1])
            cvocab = np.array(cvec.get_feature_names_out())
            from collections import defaultdict as _dd
            rows_by_src2: Dict[str, List[int]] = _dd(list)
            for i, s in enumerate(text_srcs):
                rows_by_src2[s].append(i)
            for s, idxs in rows_by_src2.items():
                if not idxs:
                    sub = sps.csr_matrix((0, n_features_count))
                else:
                    import numpy as _np
                    sub = CX_csr[_np.array(idxs), :]
                counts = np.asarray(sub.sum(axis=0)).ravel()
                if counts.size:
                    top_idx = np.argsort(counts)[::-1][:per_src_stopwords_top_n]
                    terms = [str(cvocab[j]) for j in top_idx if counts[j] > 0]
                    per_src_stopwords[s] = terms
        except Exception:
            pass

    # Near-duplicate detection via SimHash + banding
    near_duplicates = []
    if texts and near_dup:
        try:
            def simhash(text: str, bits: int = 64) -> int:
                tokens = text.split()
                v = [0] * bits
                for tok in tokens:
                    h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                    for b in range(bits):
                        v[b] += 1 if (h >> b) & 1 else -1
                fingerprint = 0
                for b in range(bits):
                    if v[b] >= 0:
                        fingerprint |= (1 << b)
                return fingerprint

            def hamming(a: int, b: int) -> int:
                return (a ^ b).bit_count()

            bits = 64
            bands = 4
            band_size = bits // bands
            sigs: List[int] = []
            for content in texts:
                norm = " ".join(content.split()).lower()
                sigs.append(simhash(norm, bits=bits))
            # LSH buckets
            from collections import defaultdict as _dd
            buckets = [_dd(list) for _ in range(bands)]
            for i, s in enumerate(sigs):
                for b in range(bands):
                    key = (s >> (b * band_size)) & ((1 << band_size) - 1)
                    buckets[b][key].append(i)
            # candidate pairs within buckets
            parent = list(range(len(sigs)))
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
            for b in range(bands):
                for _, idxs in buckets[b].items():
                    if len(idxs) < 2:
                        continue
                    base = idxs[0]
                    for j in idxs[1:]:
                        if hamming(sigs[base], sigs[j]) <= near_dup_hamming:
                            union(base, j)
            # collect clusters
            clusters = {}
            for i in range(len(sigs)):
                r = find(i)
                clusters.setdefault(r, []).append(i)
            for root, idxs in clusters.items():
                if len(idxs) > 1:
                    docids = []
                    for i in idxs[:100]:
                        # map from i back to docid order: texts appended once per doc
                        docids.append(str(i))
                    near_duplicates.append({"size": len(idxs), "doc_indices_sample": docids})
        except Exception:
            pass

    summary: Dict[str, Any] = {
        "file_path": file_path,
        "n_docs": n_docs,
        "n_unique_src": len(unique_src),
        "top_fields": [
            {"field": k, "count": int(c)} for k, c in field_presence.most_common(20)
        ],
        "top_src": [
            {"src": s, "count": int(c)} for s, c in src_counts.most_common(50)
        ],
        "elapsed_sec": round(time.time() - start, 3),
    }

    # Save artifacts
    if save:
        with (out_dir_ts / "unique_src.json").open("w", encoding="utf-8") as f:
            json.dump(sorted(unique_src), f, ensure_ascii=False, indent=2)
        with (out_dir_ts / "src_counts.json").open("w", encoding="utf-8") as f:
            json.dump(
                [{"src": s, "count": int(c)} for s, c in src_counts.most_common()],
                f,
                ensure_ascii=False,
                indent=2,
            )
        with (out_dir_ts / "field_presence.json").open("w", encoding="utf-8") as f:
            json.dump(dict(field_presence), f, ensure_ascii=False, indent=2)
        with (out_dir_ts / "field_length_stats.json").open("w", encoding="utf-8") as f:
            json.dump(field_length_stats, f, ensure_ascii=False, indent=2)
        with (out_dir_ts / "per_src_length_stats.json").open("w", encoding="utf-8") as f:
            json.dump(per_src_length_stats, f, ensure_ascii=False, indent=2)
        if duplicate_groups:
            with (out_dir_ts / "duplicates.json").open("w", encoding="utf-8") as f:
                json.dump(duplicate_groups, f, ensure_ascii=False, indent=2)
        if token_stats and token_counts_overall:
            token_summary = {
                "overall": _summarize(token_counts_overall),
                "per_src": {s: _summarize(v) for s, v in token_counts_per_src.items()},
            }
            with (out_dir_ts / "token_stats.json").open("w", encoding="utf-8") as f:
                json.dump(token_summary, f, ensure_ascii=False, indent=2)
        if keywords_per_src:
            with (out_dir_ts / "keywords_per_src.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {s: [{"term": t, "weight": w} for t, w in terms] for s, terms in keywords_per_src.items()},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        if stopwords_global:
            with (out_dir_ts / "stopwords_global.json").open("w", encoding="utf-8") as f:
                json.dump(stopwords_global, f, ensure_ascii=False, indent=2)
        if per_src_stopwords:
            with (out_dir_ts / "per_src_stopwords.json").open("w", encoding="utf-8") as f:
                json.dump(per_src_stopwords, f, ensure_ascii=False, indent=2)
        if near_duplicates:
            with (out_dir_ts / "near_duplicates.json").open("w", encoding="utf-8") as f:
                json.dump(near_duplicates, f, ensure_ascii=False, indent=2)
        with (out_dir_ts / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        # Maintain a 'latest' symlink to this report dir
        try:
            latest = out_root / "latest"
            if latest.exists() or latest.is_symlink():
                if latest.is_symlink() or latest.is_file():
                    latest.unlink()
                else:
                    # if existing directory, remove it only if empty
                    try:
                        latest.rmdir()
                    except OSError:
                        pass
            latest.symlink_to(out_dir_ts.name)
        except Exception:
            pass

    # Print concise report
    print("--- Dataset Profile ---")
    print(f"Path: {file_path}")
    print(f"Docs: {n_docs}")
    print(f"Unique src: {len(unique_src)}")
    preview = sorted(unique_src)[:max_preview]
    print(f"src preview ({len(preview)} of {len(unique_src)}): {preview}")
    if save:
        print(f"Saved artifacts to: {out_dir_ts}")

    return {
        "unique_src": sorted(unique_src),
        "src_counts": src_counts,
        "summary": summary,
        "report_dir": str(out_dir_ts) if save else None,
    }


if __name__ == "__main__":
    fire.Fire(profile)
