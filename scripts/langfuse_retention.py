"""Delete Langfuse traces older than the retention period (default LANGFUSE_RETENTION_DAYS = 30).

Langfuse's built-in Data Retention setting needs a paid plan (Pro / Enterprise). On the free
Hobby plan data older than 30 days is only HIDDEN, not deleted, so this script deletes it.
Deleting a trace removes it and its observations. It cannot be undone.

Usage (run by hand for now):

    python -m scripts.langfuse_retention                    # dry run: only counts
    python -m scripts.langfuse_retention --yes              # delete traces older than 30 days
    python -m scripts.langfuse_retention --days 14 --yes

    # In Docker:
    docker compose run --rm --entrypoint python api -m scripts.langfuse_retention --yes

Prints counts only, never trace content.
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone

from app.config import get_settings

PAGE_SIZE = 1000  # API maximum
DELETE_BATCH = 100


def find_old_trace_ids(client, cutoff: datetime) -> list[str]:
    """Trace ids whose ROOT observation started before `cutoff` (one root per chat request)."""
    trace_ids: list[str] = []
    seen: set[str] = set()
    cursor = None
    while True:
        page = client.api.observations.get_many(
            is_root_observation=True,
            to_start_time=cutoff,
            fields="core",
            limit=PAGE_SIZE,
            cursor=cursor,
        )
        for observation in page.data:
            if observation.trace_id and observation.trace_id not in seen:
                seen.add(observation.trace_id)
                trace_ids.append(observation.trace_id)
        cursor = getattr(page.meta, "cursor", None)
        if not cursor or not page.data:
            return trace_ids


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--days", type=int, default=settings.langfuse_retention_days,
                        help=f"delete traces older than this many days (default {settings.langfuse_retention_days})")
    parser.add_argument("--yes", action="store_true", help="actually delete (without it: dry run)")
    args = parser.parse_args(argv)

    if args.days < 0:
        parser.error("--days must be >= 0")
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        print("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set (.env).", file=sys.stderr)
        return 2

    from langfuse import Langfuse

    client = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
        tracing_enabled=False,  # this script only uses the REST API
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    print(f"Langfuse project at {settings.langfuse_base_url}")
    print(f"Retention: {args.days} days -> traces that started before {cutoff:%Y-%m-%d %H:%M} UTC")

    trace_ids = find_old_trace_ids(client, cutoff)
    print(f"Found {len(trace_ids)} trace(s) older than the retention period.")
    if not trace_ids:
        return 0
    if not args.yes:
        print("Dry run: nothing deleted. Re-run with --yes to delete them.")
        return 0

    deleted = 0
    for start in range(0, len(trace_ids), DELETE_BATCH):
        batch = trace_ids[start : start + DELETE_BATCH]
        client.api.trace.delete_multiple(trace_ids=batch)
        deleted += len(batch)
        print(f"Deleted {deleted}/{len(trace_ids)}")
    print("Done. Langfuse removes the data in the background; it can take a few minutes to disappear.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
