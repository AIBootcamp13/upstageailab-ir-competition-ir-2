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
    parser.add_argument('--recompute-only', action='store_true', help='When set with --recompute-embeddings, skip the _reindex step and only use recomputed documents in target')

    args = parser.parse_args(argv)
    es = args.es
    c_src = None
    c_tgt = None

    if not args.alias and not args.source:
        fatal('Either --alias or --source must be provided')

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
        ts = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        target = f"{source}-reindexed-{ts}"

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
        print(f"Recomputing embeddings from {args.documents_path} into {target} (batch={args.batch_size})")
        # Call stream_and_index on the recompute module
        getattr(recompute, 'stream_and_index')(es, args.documents_path, target, batch_size=args.batch_size, dry_run=args.dry_run)
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

    if args.verify and args.alias:
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
            audit = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'pre_swap',
                'alias': args.alias,
                'old_indices': old_indices,
                'new_index': target,
                'counts': {'source': c_src if 'c_src' in locals() else None, 'target': c_tgt if 'c_tgt' in locals() else None},
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
            }
            with_retries(lambda: _requests_post(es.rstrip('/') + '/reindex_audit/_doc', json_body=audit), retries=2)
        except Exception:
            print('Warning: failed to write post-swap audit entry')

    print('Index orchestrator completed successfully')
    return 0


if __name__ == '__main__':
    sys.exit(main())
