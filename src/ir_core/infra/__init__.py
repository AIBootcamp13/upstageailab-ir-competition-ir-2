"""Infrastructure helpers (ES, cache, connectors).

This module provides Elasticsearch helper utilities. The implementation
is self-contained here to avoid relying on a top-level compatibility
module.
"""
from elasticsearch import Elasticsearch
from ..config import settings


def get_es():
	"""Return an Elasticsearch client configured from settings.

	Use conservative timeouts to avoid the client appearing to hang when
	Elasticsearch isn't available. These settings are safe defaults for
	local development.
	"""
	return Elasticsearch(
		settings.ES_HOST,
		request_timeout=10,
		max_retries=3,
		retry_on_timeout=True,
	)


def count_docs_with_embeddings(es=None, index=None):
	"""Count documents containing an `embeddings` field."""
	es = es or get_es()
	index = index or settings.INDEX_NAME
	resp = es.count(index=index, query={"exists": {"field": "embeddings"}})
	return resp.get("count", 0)


__all__ = ["get_es", "count_docs_with_embeddings"]
