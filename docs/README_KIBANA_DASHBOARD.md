Kibana dashboard for Reindex Orchestrator
======================================

This document explains how to create a minimal Kibana setup for monitoring reindex operations.

1) Create the `reindex_audit` index with recommended mapping

```bash
# from repo root
curl -X PUT "http://localhost:9200/reindex_audit" -H 'Content-Type: application/json' -d @docs/reindex_audit_mapping.json
```

2) In Kibana, create an index pattern for `reindex_audit*` and set `timestamp` as the time field.

3) Create Saved Search / Dashboard
- Saved Search: show `timestamp`, `action`, `alias`, `old_indices`, `new_index`, `counts.source`, `counts.target`, `model`.
- Visualization: simple time-series of `count` per `action` or `metrics` (if available).

4) Optional: export/import saved objects via Kibana's Saved Objects UI when you want to ship the dashboard.

Notes
- The orchestrator writes `pre_swap` and `post_swap` audit docs to `reindex_audit/_doc` when alias swaps occur.
- If you prefer automation, add a small Kibana saved object JSON here and I can generate it.
