#!/usr/bin/env python3
"""
Swap an Elasticsearch alias to point to a new index atomically.

Usage:
    uv run python scripts/maintenance/swap_alias.py --alias <alias_name> --new <new_index> [--old <old_index>] [--delete-old]

If --old is not provided and the alias exists, the script will remove the alias from all indices and point it to the new index.
"""
import argparse
from ir_core.infra import get_es


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alias", required=True, help="Alias name to update")
    p.add_argument("--new", required=True, help="New index to point the alias to")
    p.add_argument("--old", help="Old index currently holding the alias (optional)")
    p.add_argument(
        "--delete-old",
        action="store_true",
        help="Delete the old index after swapping (use with caution)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    es = get_es()
    alias = args.alias
    new_index = args.new
    old_index = args.old

    actions = []

    # If old index provided, try to remove alias from it (if exists)
    if old_index:
        # Only remove if alias actually exists on that index
        try:
            aliases = es.indices.get_alias(name=alias, index=old_index)
            if aliases:
                actions.append({"remove": {"index": old_index, "alias": alias}})
        except Exception:
            # alias not present on old_index
            pass
    else:
        # No old specified: remove alias from all indices that currently have it
        try:
            current = es.indices.get_alias(name=alias)
            for idx in current.keys():
                actions.append({"remove": {"index": idx, "alias": alias}})
        except Exception:
            # alias not found anywhere, that's fine
            pass

    # Add alias to new index
    actions.append({"add": {"index": new_index, "alias": alias}})

    if not actions:
        print("No alias actions to perform. Exiting.")
        return

    body = {"actions": actions}
    print("Executing alias update actions:", body)
    resp = es.indices.update_aliases(body=body)
    print("Alias update response:", resp)

    if args.delete_old and old_index:
        # Double-check alias no longer points to old_index
        try:
            has_alias = False
            current = es.indices.get_alias(name=alias)
            if old_index in current:
                has_alias = True
        except Exception:
            has_alias = False

        if not has_alias:
            print(f"Deleting old index: {old_index}")
            es.indices.delete(index=old_index, ignore=[400, 404])
            print("Old index deleted.")
        else:
            print("Alias still points to old index; not deleting.")

    # Print current alias assignments
    try:
        cur = es.indices.get_alias(name=alias)
        print(f"Alias '{alias}' now points to indices: {list(cur.keys())}")
    except Exception:
        print(f"Alias '{alias}' does not exist after update (unexpected).")


if __name__ == "__main__":
    main()
