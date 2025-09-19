### l Naming Convention for Embedding Indexes

Your current index names like `..._new` and `..._fixed` are prone to confusion. A structured naming convention will make your indexes self-documenting and easier to manage.

I recommend a pattern that includes the corpus, content, model, dimension, and date.

**Proposed Structure:** `[corpus]-[content]-[model_family]-[variant]-[dims]-[date]`

| Component | Description | Example |
| :--- | :--- | :--- |
| `[corpus]` | The name of your project or dataset. | `docs` |
| `[content]` | The language or type of documents. | `ko`, `en`, `bilingual` |
| `[model_family]` | The general type of embedding model. | `sbert`, `polyglot`, `solar` |
| `[variant]` | A short, unique name for the specific model. | `kr-v40k`, `pko-1b` |
| `[dims]` | The vector dimension, prefixed with 'd'. **Crucial for avoiding errors.** | `d768`, `d2048`, `d4096` |
| `[date]` | The creation date in `YYYYMMDD` format for versioning. | `20250917` |

#### Examples

Here is how your current indexes would look with the new convention:

| Old Name | New Name |
| :--- | :--- |
| `documents_ko_with_embeddings_fixed` | `docs-ko-sbert-kr-v40k-d768-20250917` |
| `documents_polyglot_1b_with_embeddings_new` | `docs-ko-polyglot-1b-d2048-20250917` |
| `documents_en_with_embeddings_new` | `docs-en-sbert-minilm-d384-20250917` |

### Best Practices for Index Management

The most important practice is to use an **alias**. An alias is a pointer to one or more indexes. Your application should **always** query the alias, not the index name directly. This allows you to re-index in the background and swap the pointer atomically with zero downtime.

1.  **Use Descriptive Aliases:** Create aliases that describe the purpose, like `docs-ko-active` or `docs-bilingual-production`.
2.  **Swap Aliases Atomically:** When your new index is ready, use a single command to move the alias from the old index to the new one.

**Example `curl` command to swap an alias:**

```bash
curl -X POST "localhost:9200/_aliases" -H 'Content-Type: application/json' -d'
{
  "actions": [
    { "remove": { "index": "docs-ko-polyglot-1b-d2048-20250917", "alias": "docs-ko-active" }},
    { "add":    { "index": "docs-ko-polyglot-5b-d4096-20250918", "alias": "docs-ko-active" }}
  ]
}
'
```

This instantly switches your application's traffic to the new index without you needing to change any code.