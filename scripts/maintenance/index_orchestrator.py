#!/usr/bin/env python3
"""
Index orchestrator: reindex an Elasticsearch index and atomically swap an alias.

Usage examples:
  # Reindex alias 'papers' into a new index and swap alias atomically (dry-run shows actions):
  python scripts/maintenance/index_orchestrator.py --alias papers --dry-run

  # Create a timestamped target index, reindex, verify counts, then swap alias:
  python scripts/maintenance/index_orchestrator.py --alias papers --verify

This script is conservative: it will fetch the source index settings/mappings and recreate
them on the target index before reindexing. It supports dry-run, verification, timeout, and
basic rollback behavior.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from typing import Optional, Callable
from typing import List

try:
    import requests
except Exception:  # pragma: no cover - fallback
    requests = None


def fatal(msg: str, code: int = 1):
    print(f"ERROR: {msg}")
    sys.exit(code)


def with_retries(fn: Callable, retries: int = 3, backoff: float = 1.0, *args, **kwargs):
    """Call fn with retries and exponential backoff on exceptions.

    fn should raise on failure. Returns fn(*args, **kwargs) on success.
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > retries:
                raise
            sleep = backoff * (2 ** (attempt - 1))
            print(f"Operation failed (attempt {attempt}/{retries}). Retrying in {sleep}s... Error: {e}")
            time.sleep(sleep)


def _requests_get(url: str, **kwargs):
    if requests is None:
        raise RuntimeError('requests library is required')
    r = requests.get(url, timeout=kwargs.pop('timeout', 30), **kwargs)
    if not r.ok and r.status_code != 404:
        raise RuntimeError(f"GET {url} failed: {r.status_code} {r.text}")
    return r


def _requests_put(url: str, json_body=None, **kwargs):
    if requests is None:
        raise RuntimeError('requests library is required')
    r = requests.put(url, json=json_body, timeout=kwargs.pop('timeout', 300), **kwargs)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"PUT {url} failed: {r.status_code} {r.text}")
    return r


def _requests_post(url: str, json_body=None, params=None, **kwargs):
    if requests is None:
        raise RuntimeError('requests library is required')
    r = requests.post(url, json=json_body, params=params or {}, timeout=kwargs.pop('timeout', 300), **kwargs)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"POST {url} failed: {r.status_code} {r.text}")
    return r


def _requests_delete(url: str, **kwargs):
    if requests is None:
        raise RuntimeError('requests library is required')
    r = requests.delete(url, timeout=kwargs.pop('timeout', 300), **kwargs)
    if not (200 <= r.status_code < 300) and r.status_code != 404:
        raise RuntimeError(f"DELETE {url} failed: {r.status_code} {r.text}")
    return r


def index_exists(es: str, index: str) -> bool:
    r = with_retries(lambda: _requests_get(es.rstrip('/') + f'/{index}', timeout=10), retries=2)
    return r.status_code == 200


def fetch_source_index_info(es: str, index: str):
    info = with_retries(lambda: _requests_get(es.rstrip('/') + f'/{index}'), retries=3)
    if info.status_code != 200:
        fatal(f"Index {index} not found in GET /{index} (status {info.status_code})")
    data = info.json()
    idx = data.get(index, {})
    settings = idx.get('settings', {}).get('index', {})
    for k in ('creation_date', 'uuid', 'version', 'provided_name'):
        settings.pop(k, None)
    mappings = idx.get('mappings', {}) or {}
    return settings, mappings


def create_target_index(es: str, target: str, settings: dict, mappings: dict):
    body = {}
    if settings:
        body['settings'] = settings
    if mappings:
        body['mappings'] = mappings
    print(f"Creating target index {target} with source settings/mappings")
    return with_retries(lambda: _requests_put(es.rstrip('/') + f'/{target}', json_body=body), retries=3)


def start_reindex_task(es: str, source: str, target: str):
    body = {
        'source': {'index': source},
        'dest': {'index': target},
        'conflicts': 'proceed'
    }
    r = with_retries(lambda: _requests_post(es.rstrip('/') + '/_reindex?wait_for_completion=false', json_body=body), retries=3)
    return r.json()


def get_task(es: str, task_id: str):
    r = with_retries(lambda: _requests_get(es.rstrip('/') + f'/_tasks/{task_id}'), retries=3)
    return r.json()


def wait_for_task_completion(es: str, task_id: str, timeout: int = 600, poll_interval: float = 2.0):
    start = time.time()
    while True:
        data = get_task(es, task_id)
        if data.get('completed'):
            return data
        if time.time() - start > timeout:
            raise RuntimeError(f"Task {task_id} did not complete within {timeout}s")
        time.sleep(poll_interval)


def count(es: str, index: str) -> int:
    r = with_retries(lambda: _requests_get(es.rstrip('/') + f'/{index}/_count'), retries=3)
    return int(r.json().get('count', 0))


def validate_index(es: str, index: str, limit: int = 50) -> dict:
    """Run the repository validation script for a small sample against the given index.

    This function invokes the existing validation script as a subprocess. It returns
    a dict with {'map': float or None, 'total_queries': int, 'raw_output': str}.
    """
    import subprocess
    import shlex

    # Run the existing Hydra-based validation script but override es_host and index_name
    # so it targets the freshly created index. We also pass the sample limit.
    # Use Poetry to ensure environment consistency.
    cmd = (
        f"PYTHONPATH=src poetry run python scripts/evaluation/validate_retrieval.py "
        f"+es_host={es} +index_name={index} limit={limit}"
    )
    print(f"Running validation command: {cmd}")
    try:
        proc = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        out = proc.stdout
        map_val = None
        total_q = None
        for line in out.splitlines():
            if 'MAP Score' in line:
                try:
                    # parse trailing number
                    parts = line.strip().split()
                    map_val = float(parts[-1])
                except Exception:
                    continue
            if '검증된 쿼리 수' in line or 'Validated' in line:
                try:
                    total_q = int(line.strip().split()[-1])
                except Exception:
                    continue
        return {'map': map_val, 'total_queries': total_q or limit, 'raw_output': out}
    except subprocess.CalledProcessError as e:
        print(f"Validation script failed: {e}")
        return {'map': None, 'total_queries': 0, 'raw_output': getattr(e, 'output', '')}


def build_alias_actions(alias: str, add_index: str, remove_indices: Optional[list] = None, keep_old: bool = False):
    actions = []
    if remove_indices and not keep_old:
        for idx in remove_indices:
            actions.append({'remove': {'index': idx, 'alias': alias}})
    actions.append({'add': {'index': add_index, 'alias': alias}})
    return {'actions': actions}


def swap_alias(es: str, alias: str, add_index: str, remove_indices: Optional[list] = None, keep_old: bool = False):
    body = build_alias_actions(alias, add_index, remove_indices, keep_old)
    print(f"Swapping alias {alias} -> {add_index} (removing: {remove_indices}, keep_old={keep_old})")
    return with_retries(lambda: _requests_post(es.rstrip('/') + '/_aliases', json_body=body), retries=3)


def resolve_alias(es: str, alias: str) -> list:
    r = with_retries(lambda: _requests_get(es.rstrip('/') + f'/_alias/{alias}'), retries=2)
    if r.status_code == 404:
        return []
    data = r.json()
    return list(data.keys())


def main(argv=None):
    parser = argparse.ArgumentParser(description='Index orchestrator: reindex + atomic alias swap')
    parser.add_argument('--es', default='http://127.0.0.1:9200', help='Elasticsearch URL')
    parser.add_argument('--alias', help='Alias name to move to the new index (optional if --source provided)')
    parser.add_argument('--source', help='Source index name (optional if --alias provided)')
    parser.add_argument('--target', help='Target index name to create (default: autogenerated)')
    parser.add_argument('--dry-run', action='store_true', help='Show actions but do not modify cluster')
    parser.add_argument('--verify', action='store_true', help='Verify doc counts before swapping alias')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout seconds for reindexing (only used for task polling)')
    parser.add_argument('--force', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--keep-old', action='store_true', help='Do not remove old indices when swapping alias')
    parser.add_argument('--rollback-on-failure', action='store_true', help='Delete the created target index if verification fails')
    parser.add_argument('--recompute-embeddings', action='store_true', help='Recompute embeddings for documents and bulk index into the target')
    parser.add_argument('--documents-path', help='Path to documents.jsonl to index when --recompute-embeddings is set', default='data/documents.jsonl')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for recomputing embeddings')
    parser.add_argument('--embedding-batch-size', type=int, default=32, help='Per-call batch size passed to the embedding function (helps control GPU memory)')
    parser.add_argument('--recompute-only', action='store_true', help='When set with --recompute-embeddings, skip the _reindex step and only use recomputed documents in target')
    parser.add_argument('--ensure-audit-index', action='store_true', help='Create the reindex_audit index with mapping from docs/reindex_audit_mapping.json before writing audit docs')
    parser.add_argument('--import-kibana-saved-objects', help='Path to a Kibana saved objects JSON file to import after a successful swap')
    parser.add_argument('--kibana-url', help='Kibana base URL (e.g., http://localhost:5601) used when --import-kibana-saved-objects is set')
    parser.add_argument('--kibana-api-key', help='Kibana API key for authentication (optional). If not set and kibana-url is provided, import will try unauthenticated POST.')
    parser.add_argument('--validation', action='store_true', help='Run retrieval validation (MAP) against the target index before alias swap')
    parser.add_argument('--validation-limit', type=int, default=50, help='Number of validation queries to run when --validation is set')
    parser.add_argument('--min-map', type=float, default=0.0, help='Minimum MAP required to pass validation (default 0.0)')

    args = parser.parse_args(argv)
    es = args.es
    c_src = None
    c_tgt = None

    # If alias/source not provided, try to use central settings defaults
    if not args.alias and not args.source:
        try:
            # Lazy import of project settings
            from ir_core.config import settings as project_settings
            default_alias = getattr(project_settings, 'INDEX_ALIAS', None) or None
            default_index = getattr(project_settings, 'INDEX_NAME', None) or None
            if default_alias:
                args.alias = default_alias
                print(f"Using default alias from settings: {args.alias}")
            elif default_index:
                args.source = default_index
                print(f"Using default source index from settings: {args.source}")
        except Exception:
            pass

    if not args.alias and not args.source:
        fatal('Either --alias or --source must be provided (or set INDEX_ALIAS/INDEX_NAME in config)')

    if args.source:
        source = args.source
    else:
        aliased = resolve_alias(es, args.alias)
        if not aliased:
            fatal(f'Alias {args.alias} not found or points to no indices')
        source = aliased[0]

    if args.target:
        target = args.target
    else:
        # Use configured prefix if available
        try:
            from ir_core.config import settings as project_settings
            prefix = getattr(project_settings, 'INDEX_NAME_PREFIX', None) or f"{source}-reindexed-"
        except Exception:
            prefix = f"{source}-reindexed-"
        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        # if prefix looks like a full prefix (not containing source) just append timestamp
        if prefix.endswith('-') or prefix.endswith('_'):
            target = f"{prefix}{ts}"
        else:
            # make a sensible name: <prefix><timestamp>
            target = f"{prefix}{ts}"

    print(f"Source index: {source}")
    print(f"Target index: {target}")
    if args.alias:
        print(f"Alias to swap: {args.alias}")

    if args.dry_run:
        print("DRY RUN: no changes will be made. Showing planned actions:")
        print(f" - create index: {target} (copy settings/mappings from {source})")
        print(f" - reindex from {source} -> {target} (async task)")
        if args.alias:
            old = resolve_alias(es, args.alias)
            print(f" - update alias {args.alias}: remove {old} add {target} (keep_old={args.keep_old})")
        return 0

    if not index_exists(es, source):
        fatal(f"Source index {source} does not exist")

    settings, mappings = fetch_source_index_info(es, source)

    if not args.force:
        resp = input(f"Proceed to create target index '{target}' and reindex from '{source}'? [y/N]: ")
        if resp.lower() != 'y':
            print('Aborting')
            return 2

    create_target_index(es, target, settings, mappings)

    # Optionally recompute embeddings and bulk index into the newly created target
    if args.recompute_embeddings:
        recompute = None
        try:
            # Local import to keep optional dependency minimal
            import scripts.maintenance.recompute as recompute
        except Exception as e:
            fatal(f"Recompute requested but recompute module not available: {e}")
        # Detect device: prefer cuda when available
        device = None
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            device = 'cpu'

        print(f"Recomputing embeddings from {args.documents_path} into {target} (batch={args.batch_size}, embedding_batch_size={args.embedding_batch_size}, device={device})")
        # Call stream_and_index on the recompute module and pass embedding parameters
        getattr(recompute, 'stream_and_index')(
            es,
            args.documents_path,
            target,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            embedding_batch_size=args.embedding_batch_size,
            device=device,
        )
    # Unless recompute-only is set, start reindex as an asynchronous task and poll
    task_resp = {}
    if not (args.recompute_embeddings and args.recompute_only):
        task_resp = start_reindex_task(es, source, target)
        task_id = task_resp.get('task') or task_resp.get('task_id') or (task_resp.get('response') or {}).get('task')
        if not task_id:
            # Fallback: some ES versions return different shapes; try to detect
            # If wait_for_completion was used originally, it would return completed response.
            print('Reindex response did not include a task id; assuming synchronous completion')
        else:
            print(f'Reindex started as task {task_id}; polling for completion (timeout={args.timeout}s)')
            completed = wait_for_task_completion(es, task_id, timeout=args.timeout)
            if not completed.get('completed'):
                fatal(f'Reindex task {task_id} did not complete successfully')
    else:
        # recompute-only: assume recompute already wrote documents into target
        print('Recompute-only mode: skipping _reindex step')
    task_id = task_resp.get('task') or task_resp.get('task_id') or (task_resp.get('response') or {}).get('task')
    if not task_id:
        # Fallback: some ES versions return different shapes; try to detect
        # If wait_for_completion was used originally, it would return completed response.
        print('Reindex response did not include a task id; assuming synchronous completion')
    else:
        print(f'Reindex started as task {task_id}; polling for completion (timeout={args.timeout}s)')
        completed = wait_for_task_completion(es, task_id, timeout=args.timeout)
        if not completed.get('completed'):
            fatal(f'Reindex task {task_id} did not complete successfully')

    # Capture validation metrics if requested to include in audit entries
    validation_result = None
    if args.verify and args.alias:
        if args.validation:
            print('Running retrieval validation (MAP) against the target index...')
            val = validate_index(es, target, limit=args.validation_limit)
            validation_result = val
            print(f"Validation result: MAP={val.get('map')} total_queries={val.get('total_queries')}")
            if val.get('map') is None or val.get('map', 0.0) < args.min_map:
                if args.rollback_on_failure:
                    print(f"Validation failed and --rollback-on-failure set: deleting target index {target}")
                    try:
                        with_retries(lambda: _requests_delete(es.rstrip('/') + f'/{target}'), retries=2)
                    except Exception as e:
                        print(f"Rollback deletion failed: {e}")
                fatal('Validation failed (MAP below threshold). Aborting alias swap.')
        else:
            print('Verifying document counts...')
            c_src = count(es, source)
            c_tgt = count(es, target)
            print(f'counts: source={c_src} target={c_tgt}')
            if c_src != c_tgt:
                # Optionally rollback by deleting the created target index
                if args.rollback_on_failure:
                    print(f"Verification failed and --rollback-on-failure set: deleting target index {target}")
                    try:
                        with_retries(lambda: _requests_delete(es.rstrip('/') + f'/{target}'), retries=2)
                    except Exception as e:
                        print(f"Rollback deletion failed: {e}")
                fatal('Document counts differ between source and target. Aborting alias swap.')

    if args.alias:
        old_indices = resolve_alias(es, args.alias)
        # Write audit entry before swap
        try:
            if args.ensure_audit_index:
                try:
                    # Load mapping from repo and create index if missing
                    import json as _json
                    with open('docs/reindex_audit_mapping.json', 'r', encoding='utf-8') as _fh:
                        _mapping = _json.load(_fh)
                    print('Ensuring reindex_audit index exists with recommended mapping')
                    with_retries(lambda: _requests_put(es.rstrip('/') + '/reindex_audit', json_body=_mapping), retries=2)
                except Exception as _e:
                    print(f'Warning: failed to create reindex_audit index: {_e}')
            audit = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'pre_swap',
                'alias': args.alias,
                'old_indices': old_indices,
                'new_index': target,
                'counts': {'source': c_src if 'c_src' in locals() else None, 'target': c_tgt if 'c_tgt' in locals() else None},
                'validation': {
                    'map': (validation_result or {}).get('map') if validation_result else None,
                    'total_queries': (validation_result or {}).get('total_queries') if validation_result else None,
                },
            }
            with_retries(lambda: _requests_post(es.rstrip('/') + '/reindex_audit/_doc', json_body=audit), retries=2)
        except Exception:
            print('Warning: failed to write pre-swap audit entry')
        swap_alias(es, args.alias, target, remove_indices=old_indices, keep_old=args.keep_old)
        print(f'Alias {args.alias} now points to {target}')
        # Write audit entry after swap
        try:
            audit = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'post_swap',
                'alias': args.alias,
                'old_indices': old_indices,
                'new_index': target,
                'validation': {
                    'map': (validation_result or {}).get('map') if validation_result else None,
                    'total_queries': (validation_result or {}).get('total_queries') if validation_result else None,
                },
            }
            if args.ensure_audit_index:
                # best-effort: ensure index exists (ignore errors)
                try:
                    import json as _json
                    with open('docs/reindex_audit_mapping.json', 'r', encoding='utf-8') as _fh:
                        _mapping = _json.load(_fh)
                    with_retries(lambda: _requests_put(es.rstrip('/') + '/reindex_audit', json_body=_mapping), retries=1)
                except Exception:
                    pass
            with_retries(lambda: _requests_post(es.rstrip('/') + '/reindex_audit/_doc', json_body=audit), retries=2)
        except Exception:
            print('Warning: failed to write post-swap audit entry')

        # Optionally import Kibana saved objects (best-effort)
        if args.import_kibana_saved_objects and args.kibana_url:
            try:
                path = args.import_kibana_saved_objects
                with open(path, 'rb') as fh:
                    payload = fh.read()
                headers = {'kbn-xsrf': 'true', 'Content-Type': 'application/json'}
                if args.kibana_api_key:
                    headers['Authorization'] = f'ApiKey {args.kibana_api_key}'
                url = args.kibana_url.rstrip('/') + '/api/saved_objects/_import?overwrite=true'
                # Kibana import endpoint expects multipart/form-data file upload; fallback to bulk_create if import endpoint not available
                try:
                    # Try the import endpoint (file upload)
                    files = {'file': (path, payload, 'application/json')}
                    r = requests.post(url, headers={'kbn-xsrf': 'true'}, files=files, timeout=60)
                    if not (200 <= r.status_code < 300):
                        raise RuntimeError(f'Kibana import failed: {r.status_code} {r.text}')
                    print('Kibana saved objects imported via /api/saved_objects/_import')
                except Exception:
                    # Fallback: POST objects to _bulk_create
                    try:
                        import json as _json
                        objs = _json.loads(payload)
                        bulk_url = args.kibana_url.rstrip('/') + '/api/saved_objects/_bulk_create?overwrite=true'
                        r = requests.post(bulk_url, json=objs.get('objects', []), headers=headers, timeout=60)
                        if not (200 <= r.status_code < 300):
                            raise RuntimeError(f'Kibana bulk_create failed: {r.status_code} {r.text}')
                        print('Kibana saved objects bulk-created')
                    except Exception as e:
                        print(f'Warning: failed to import Kibana saved objects: {e}')
            except Exception as e:
                print(f'Warning: Kibana import step failed: {e}')

    print('Index orchestrator completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
