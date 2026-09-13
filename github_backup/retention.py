#!/usr/bin/env python
"""Retention tidy-up for an existing github-backup output directory.

This is an *offline* maintenance pass, separate from the normal backup flow:

1. :func:`build_retention_plan` scans a backup directory read-only and produces
   a deterministic plan describing what would be removed, how large it is and
   why. Nothing is deleted while planning.
2. :func:`apply_retention_plan` executes an approved plan. Execution is
   journaled and two-phase: every victim is first moved into a quarantine
   directory (an atomic rename on the same filesystem), and bytes are only
   actually unlinked once every planned entry has been staged. An interrupted
   run is resumed from the journal on the next invocation, so re-running the
   policy or continuing after a crash converges to the same, reviewable
   result.

Safety rules baked into the scan:

* The most recent complete backup data is never touched: repository/wiki/gist
  clones, ``*.json`` dumps, incremental checkpoints (``last_update``,
  ``reviews_last_update``) and release assets are all kept.
* An attachment is only removable when its ``manifest.json`` is present and
  readable and the file is not recorded there as a successful download. File
  modification times are never used as a deletion criterion. When a manifest
  is missing or corrupt, the whole attachment directory is kept.
* Directories from older tool versions and anything the scanner does not
  recognize are kept by default and only reported.
* The backup process and a retention pass mutually exclude each other via an
  ``flock``-based lock file, so cleanup never interleaves with a backup
  writing into the same tree.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix platforms
    fcntl = None

from .github_backup import logger

RETENTION_FORMAT_VERSION = 1

#: Directory (inside the backup root) holding plans, journals, reports and
#: quarantine areas. Dot-prefixed so it never collides with backup data.
METADATA_DIRNAME = ".github-backup-retention"

#: Mutual-exclusion lock shared by backups and retention passes.
LOCK_FILENAME = ".github-backup.lock"

# Resource directories created per repository. Anything else next to them is
# an old-version/unknown directory and is kept by default.
REPO_CHILDREN = frozenset(
    {
        "repository",
        "wiki",
        "issues",
        "pulls",
        "discussions",
        "milestones",
        "security-advisories",
        "labels",
        "hooks",
        "releases",
    }
)

#: Resource directories that can hold per-item attachment downloads.
ATTACHMENT_RESOURCE_DIRS = frozenset({"issues", "pulls", "discussions"})

#: Incremental checkpoint files relied upon by future backups.
CHECKPOINT_FILENAMES = frozenset({"last_update", "reviews_last_update"})

# Top-level entries produced by github-backup.
KNOWN_TOP_LEVEL_DIRS = frozenset({"repositories", "starred", "gists", "account"})
KNOWN_TOP_LEVEL_FILES = frozenset({"last_update"})

# Removal categories (values are stable identifiers used in plans/journals).
REMOVE_TEMP = "temp_file"
REMOVE_ORPHAN_ATTACHMENT = "orphan_attachment"
REMOVE_EMPTY_DIRECTORY = "empty_directory"

REMOVABLE_CATEGORIES = frozenset(
    {REMOVE_TEMP, REMOVE_ORPHAN_ATTACHMENT, REMOVE_EMPTY_DIRECTORY}
)

# Keep categories.
KEEP_BACKUP_DATA = "backup_data"
KEEP_CHECKPOINT = "checkpoint"
KEEP_REFERENCED_ATTACHMENT = "referenced_attachment"
KEEP_UNVERIFIED_ATTACHMENT = "unverified_attachment"
KEEP_UNRECOGNIZED = "unrecognized"
KEEP_METADATA = "retention_metadata"

REASONS = {
    REMOVE_TEMP: "atomic-write leftover (.temp) from an interrupted backup run",
    REMOVE_ORPHAN_ATTACHMENT: "not recorded as a successful download in the "
    "directory's manifest.json",
    REMOVE_EMPTY_DIRECTORY: "empty attachments container once orphan files "
    "are removed",
    KEEP_BACKUP_DATA: "backup data (repository clones, JSON dumps or release assets)",
    KEEP_CHECKPOINT: "incremental backup checkpoint required by future backup runs",
    KEEP_REFERENCED_ATTACHMENT: "recorded as a successful download in manifest.json",
    KEEP_UNVERIFIED_ATTACHMENT: "manifest.json missing or unreadable, kept to be safe",
    KEEP_UNRECOGNIZED: "unrecognized or old-version resource, kept by policy",
    KEEP_METADATA: "backup lock/retention metadata",
}

# Cap how many unrecognized paths are embedded verbatim in a plan.
UNRECOGNIZED_SAMPLE_LIMIT = 200


class RetentionError(Exception):
    """Base class for retention failures."""


class BackupLockedError(RetentionError):
    """Raised when the backup directory lock is already held."""


class PlanMismatchError(RetentionError):
    """Raised when an approved plan no longer matches the directory."""


class InvalidPlanError(RetentionError):
    """Raised when a plan file cannot be authenticated."""


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def backup_lock(root, purpose):
    """Exclusive lock over a backup output directory.

    Both regular backups and retention passes take the same lock, so cleanup
    can never interleave with a backup writing into ``root`` (and vice versa).

    The lock is a non-blocking :func:`fcntl.flock`, so it is released
    automatically by the kernel if the holder crashes; there is no stale-lock
    cleanup problem. The lock file itself is intentionally left on disk.
    """
    if fcntl is None:  # pragma: no cover - platform guard
        raise RetentionError(
            "Retention/backup locking requires the fcntl module (Unix only)"
        )

    lock_path = os.path.join(root, LOCK_FILENAME)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    acquired = False
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError:
            holder = _read_lock_holder(lock_fd)
            raise BackupLockedError(
                "Cannot {0} '{1}': another github-backup process holds the lock"
                " ({2}). Re-run once it finishes.".format(
                    purpose, root, holder or "owner unknown"
                )
            )

        os.ftruncate(lock_fd, 0)
        os.write(
            lock_fd,
            (
                json.dumps(
                    {
                        "purpose": purpose,
                        "pid": os.getpid(),
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            ).encode("utf-8"),
        )
        os.fsync(lock_fd)
        yield
    finally:
        if acquired:
            with contextlib.suppress(OSError):
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def _read_lock_holder(lock_fd):
    try:
        os.lseek(lock_fd, 0, os.SEEK_SET)
        raw = os.read(lock_fd, 4096).decode("utf-8", errors="replace").strip()
        if not raw:
            return None
        payload = json.loads(raw)
        return "purpose={0}, pid={1}, started_at={2}".format(
            payload.get("purpose", "?"),
            payload.get("pid", "?"),
            payload.get("started_at", "?"),
        )
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Read-only scanning and planning
# ---------------------------------------------------------------------------


def _attachment_item_parts(parts):
    """Return the relative parts of an attachment item directory, or None.

    Matches files laid out as::

        repositories/<repo>/<issues|pulls|discussions>/attachments/<item>/<file>
        starred/<owner>/<repo>/<resource>/attachments/<item>/<file>
    """
    if len(parts) < 4 or parts[-3] != "attachments":
        return None
    if parts[-4] not in ATTACHMENT_RESOURCE_DIRS:
        return None
    if parts[0] == "repositories" and len(parts) >= 6:
        return parts[:-1]
    if parts[0] == "starred" and len(parts) >= 7:
        return parts[:-1]
    return None


def _attachment_container_parts(parts):
    """Whether ``parts`` (a directory) is an attachments/ or attachments/<item>
    container that is eligible for empty-directory removal."""
    if parts[0] == "repositories" and len(parts) >= 4:
        if parts[-1] == "attachments" and parts[-2] in ATTACHMENT_RESOURCE_DIRS:
            return True
        if (
            len(parts) >= 5
            and parts[-2] == "attachments"
            and parts[-3] in ATTACHMENT_RESOURCE_DIRS
        ):
            return True
    if parts[0] == "starred" and len(parts) >= 5:
        if parts[-1] == "attachments" and parts[-2] in ATTACHMENT_RESOURCE_DIRS:
            return True
        if (
            len(parts) >= 6
            and parts[-2] == "attachments"
            and parts[-3] in ATTACHMENT_RESOURCE_DIRS
        ):
            return True
    return False


def _load_attachment_manifest(manifest_path, manifest_cache):
    """Return (ok, expected_filenames) for an attachment manifest.

    ``ok`` is False when the manifest is missing or cannot be parsed, which
    forces every file in the directory to be kept.
    """
    if manifest_path in manifest_cache:
        return manifest_cache[manifest_path]

    result = (False, frozenset())
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        attachments = manifest.get("attachments")
        if isinstance(manifest, dict) and isinstance(attachments, list):
            expected = frozenset(
                entry.get("saved_as")
                for entry in attachments
                if isinstance(entry, dict)
                and entry.get("success")
                and entry.get("saved_as")
            )
            result = (True, expected)
    except (OSError, ValueError):
        result = (False, frozenset())

    manifest_cache[manifest_path] = result
    return result


def _classify_structural(parts, is_dir_context=False):
    """Keep-category for a path inside the recognized layout, or None if the
    structural location itself is unknown (kept as unrecognized)."""
    top = parts[0]

    if top not in KNOWN_TOP_LEVEL_DIRS and top not in KNOWN_TOP_LEVEL_FILES:
        return KEEP_UNRECOGNIZED

    if top == "repositories":
        if len(parts) < 3:
            # File directly under repositories/ (no such thing today).
            return KEEP_UNRECOGNIZED
        if parts[2] not in REPO_CHILDREN:
            return KEEP_UNRECOGNIZED
        return KEEP_BACKUP_DATA

    if top == "starred":
        if len(parts) < 4:
            return KEEP_UNRECOGNIZED
        if parts[3] not in REPO_CHILDREN:
            return KEEP_UNRECOGNIZED
        return KEEP_BACKUP_DATA

    if top == "gists":
        if len(parts) < 3:
            return KEEP_UNRECOGNIZED
        if parts[2] == "gist.json" or parts[2] == "repository":
            return KEEP_BACKUP_DATA
        return KEEP_UNRECOGNIZED

    if top == "account":
        if len(parts) <= 2:
            return KEEP_BACKUP_DATA
        return KEEP_UNRECOGNIZED

    if top == "last_update":
        return KEEP_CHECKPOINT

    return KEEP_UNRECOGNIZED


def _classify_file(root, parts, name, is_symlink, manifest_cache):
    """Return (decision, category, reason) for one file.

    ``parts`` is the file's path relative to ``root`` as a tuple including the
    file ``name`` as its last element.
    """
    # Our own metadata (plans, journals, quarantine) is always retained.
    if parts[0] == METADATA_DIRNAME:
        return ("keep", KEEP_METADATA, REASONS[KEEP_METADATA])
    if len(parts) == 1 and name == LOCK_FILENAME:
        return ("keep", KEEP_METADATA, REASONS[KEEP_METADATA])

    # Atomic-write leftovers are the one kind of garbage recognizable
    # anywhere in the tree (the backup itself ignores *.temp files).
    if name.endswith(".temp"):
        return ("remove", REMOVE_TEMP, REASONS[REMOVE_TEMP])

    if is_symlink:
        # Never follow or remove symlinks: treat as unknown.
        return ("keep", KEEP_UNRECOGNIZED, REASONS[KEEP_UNRECOGNIZED])

    # Attachment rule: manifest-driven, never mtime-driven.
    item_parts = _attachment_item_parts(parts)
    if item_parts is not None:
        item_dir = os.path.join(root, *item_parts)
        manifest_path = os.path.join(item_dir, "manifest.json")
        if name == "manifest.json":
            return (
                "keep",
                KEEP_REFERENCED_ATTACHMENT,
                REASONS[KEEP_REFERENCED_ATTACHMENT],
            )
        ok, expected = _load_attachment_manifest(manifest_path, manifest_cache)
        if not ok:
            return (
                "keep",
                KEEP_UNVERIFIED_ATTACHMENT,
                REASONS[KEEP_UNVERIFIED_ATTACHMENT],
            )
        if name in expected:
            return (
                "keep",
                KEEP_REFERENCED_ATTACHMENT,
                REASONS[KEEP_REFERENCED_ATTACHMENT],
            )
        return (
            "remove",
            REMOVE_ORPHAN_ATTACHMENT,
            REASONS[REMOVE_ORPHAN_ATTACHMENT],
        )

    if name in CHECKPOINT_FILENAMES and len(parts) >= 2:
        parent = parts[-2]
        if parent in ATTACHMENT_RESOURCE_DIRS:
            return ("keep", KEEP_CHECKPOINT, REASONS[KEEP_CHECKPOINT])

    if len(parts) == 1 and name == "last_update":
        return ("keep", KEEP_CHECKPOINT, REASONS[KEEP_CHECKPOINT])

    category = _classify_structural(parts)
    if category is None:
        category = KEEP_BACKUP_DATA
    return ("keep", category, REASONS[category])


def build_retention_plan(root):
    """Scan ``root`` read-only and build a deterministic retention plan.

    The plan never depends on wall-clock time or file mtimes: scanning the
    same tree twice yields the same ``plan_id``.
    """
    root = os.path.realpath(root)
    manifest_cache = {}

    # dir_rel -> {"dirs": [names], "files": [(name, decision, category, size)]}
    dirs = {(): {"dirs": [], "files": []}}
    symlink_dirs = set()
    kept_summary = {}
    unrecognized_paths = []

    def record_keep(category, size):
        bucket = kept_summary.setdefault(category, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += size

    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        dirnames.sort()
        filenames.sort()
        rel_current = os.path.relpath(current, root)
        current_parts = () if rel_current == "." else tuple(rel_current.split(os.sep))
        record = dirs.setdefault(current_parts, {"dirs": [], "files": []})
        record["dirs"] = list(dirnames)

        for dirname in dirnames:
            child_parts = current_parts + (dirname,)
            dirs.setdefault(child_parts, {"dirs": [], "files": []})
            # A symlinked directory is an unknown entry that must keep its
            # parent; os.walk with followlinks=False still lists it.
            if os.path.islink(os.path.join(current, dirname)):
                symlink_dirs.add(child_parts)
                record_keep(KEEP_UNRECOGNIZED, 0)
                if len(unrecognized_paths) < UNRECOGNIZED_SAMPLE_LIMIT:
                    unrecognized_paths.append("/".join(child_parts) + "/")

        for filename in filenames:
            child_parts = current_parts + (filename,)
            full_path = os.path.join(current, filename)
            is_symlink = os.path.islink(full_path)
            size = 0
            if not is_symlink:
                try:
                    size = os.lstat(full_path).st_size
                except OSError:
                    # Disappeared mid-scan (e.g. a running backup despite the
                    # lock contract): be conservative and skip classification.
                    continue

            decision, category, reason = _classify_file(
                root, child_parts, filename, is_symlink, manifest_cache
            )
            record["files"].append((filename, decision, category, size))
            if decision == "keep":
                record_keep(category, size)
                if category == KEEP_UNRECOGNIZED:
                    if len(unrecognized_paths) < UNRECOGNIZED_SAMPLE_LIMIT:
                        unrecognized_paths.append("/".join(child_parts))

    # Post-order pass: find attachments/ and attachments/<item>/ containers
    # that contain nothing worth keeping.
    removable_dirs = []
    dir_states = {}  # parts -> "kept" | "removable" | "empty"
    for parts in sorted(dirs, key=lambda p: (len(p), p), reverse=True):
        record = dirs[parts]
        has_removable = False
        if any(decision == "remove" for _, decision, _, _ in record["files"]):
            has_removable = True
        if any(decision == "keep" for _, decision, _, _ in record["files"]):
            state = "kept"
        else:
            child_states = [
                (
                    "kept"
                    if (parts + (d,)) in symlink_dirs
                    else dir_states.get(parts + (d,), "kept")
                )
                for d in record["dirs"]
            ]
            if "kept" in child_states:
                state = "kept"
            elif has_removable or "removable" in child_states:
                state = "removable"
            else:
                state = "empty"
        dir_states[parts] = state
        if (
            parts
            and parts not in symlink_dirs
            and _attachment_container_parts(parts)
            and state != "kept"
        ):
            removable_dirs.append(parts)

    entries = []
    for parts, record in dirs.items():
        for filename, decision, category, size in record["files"]:
            if decision != "remove":
                continue
            entries.append(
                {
                    "path": "/".join(parts + (filename,)),
                    "size_bytes": size,
                    "category": category,
                    "reason": REASONS[category],
                    "type": "file",
                }
            )

    for parts in removable_dirs:
        entries.append(
            {
                "path": "/".join(parts) + "/",
                "size_bytes": 0,
                "category": REMOVE_EMPTY_DIRECTORY,
                "reason": REASONS[REMOVE_EMPTY_DIRECTORY],
                "type": "directory",
            }
        )

    # Files first (sorted by path), then directories deepest-first so children
    # are removed before their parents.
    entries.sort(
        key=lambda e: (
            0 if e["type"] == "file" else 1,
            0 if e["type"] == "file" else -e["path"].count("/"),
            e["path"],
        )
    )

    total_bytes = sum(
        entry["size_bytes"] for entry in entries if entry["type"] == "file"
    )
    plan = {
        "format_version": RETENTION_FORMAT_VERSION,
        "root": root,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
        "total_remove_count": len(entries),
        "total_remove_bytes": total_bytes,
        "kept_summary": kept_summary,
        "unrecognized_sample": unrecognized_paths,
        "unrecognized_truncated": len(unrecognized_paths) >= UNRECOGNIZED_SAMPLE_LIMIT,
    }
    plan["plan_id"] = compute_plan_id(plan)
    return plan


def _plan_fingerprint_material(plan):
    """Stable content covered by the plan id. Timestamps and summaries are
    deliberately excluded so identical trees hash identically."""
    entries = [
        (entry["path"], entry["size_bytes"], entry["category"], entry["type"])
        for entry in plan["entries"]
    ]
    entries.sort()
    return [RETENTION_FORMAT_VERSION, plan["root"], entries]


def compute_plan_id(plan):
    material = json.dumps(
        _plan_fingerprint_material(plan), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def format_bytes(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024.0 or unit == "TB":
            if unit == "B":
                return "{0} B".format(int(size))
            return "{0:.1f} {1}".format(size, unit)
        size /= 1024.0


def summarize_plan(plan):
    """Human-readable preview, grouped by category with sizes and reasons."""
    lines = []
    lines.append(
        "Retention plan {0} for {1}".format(plan["plan_id"][:12], plan["root"])
    )
    lines.append(
        "Would remove: {0} entr{1}, {2}".format(
            plan["total_remove_count"],
            "y" if plan["total_remove_count"] == 1 else "ies",
            format_bytes(plan["total_remove_bytes"]),
        )
    )

    groups = {}
    for entry in plan["entries"]:
        groups.setdefault(entry["category"], []).append(entry)
    if groups:
        lines.append("")
        lines.append("Removable:")
        for category in sorted(groups):
            items = groups[category]
            files = [i for i in items if i["type"] == "file"]
            dirs = [i for i in items if i["type"] == "directory"]
            bytes_total = sum(i["size_bytes"] for i in files)
            lines.append(
                "  {0:20s} {1:4d} file(s), {2:4d} dir(s)  {3:>10s}  {4}".format(
                    category,
                    len(files),
                    len(dirs),
                    format_bytes(bytes_total),
                    REASONS.get(category, ""),
                )
            )
        limit = 20
        shown = 0
        for entry in plan["entries"]:
            if shown >= limit:
                lines.append(
                    "  ... and {0} more (see the plan file for the full list)".format(
                        len(plan["entries"]) - limit
                    )
                )
                break
            lines.append(
                "    - [{0}] {1} ({2})".format(
                    entry["category"],
                    entry["path"],
                    format_bytes(entry["size_bytes"]),
                )
            )
            shown += 1

    if plan["kept_summary"]:
        lines.append("")
        lines.append("Kept by policy:")
        for category in sorted(plan["kept_summary"]):
            bucket = plan["kept_summary"][category]
            lines.append(
                "  {0:24s} {1:6d} file(s)  {2:>10s}  {3}".format(
                    category,
                    bucket["count"],
                    format_bytes(bucket["bytes"]),
                    REASONS.get(category, ""),
                )
            )

    sample = plan.get("unrecognized_sample") or []
    if sample:
        lines.append("")
        lines.append(
            "Unrecognized/old-version resources are kept by default ({0}shown):".format(
                "partial list, " if plan.get("unrecognized_truncated") else ""
            )
        )
        for path in sample[:20]:
            lines.append("    ? {0}".format(path))
        if len(sample) > 20:
            lines.append("    ? ... and {0} more".format(len(sample) - 20))

    if plan["total_remove_count"] == 0:
        lines.append("")
        lines.append(
            "Nothing to remove: every file is still referenced or "
            "deliberately retained."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plan files (approval artifact)
# ---------------------------------------------------------------------------


def metadata_dir(root):
    return os.path.join(root, METADATA_DIRNAME)


def write_plan_file(root, plan):
    directory = metadata_dir(root)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "plan-{0}.json".format(plan["plan_id"][:12]))
    _atomic_write_json(path, plan)
    return path


def load_approved_plan(path, expected_root):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            plan = json.load(handle)
    except (OSError, ValueError) as exc:
        raise InvalidPlanError("Cannot read plan file {0!r}: {1}".format(path, exc))

    for key in ("format_version", "root", "entries", "plan_id"):
        if key not in plan:
            raise InvalidPlanError("Plan file {0!r} is missing {1!r}".format(path, key))

    if os.path.realpath(plan["root"]) != os.path.realpath(expected_root):
        raise InvalidPlanError(
            "Plan {0!r} was written for {1!r}, not for {2!r}".format(
                path, plan["root"], expected_root
            )
        )

    if compute_plan_id(plan) != plan["plan_id"]:
        raise InvalidPlanError(
            "Plan {0!r} has been modified since it was generated; "
            "re-run --tidy to produce a current plan.".format(path)
        )

    return plan


def _atomic_write_json(path, payload):
    temporary = path + ".temp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# Journaled, two-phase execution
# ---------------------------------------------------------------------------


def _journal_path(root, plan_id):
    return os.path.join(metadata_dir(root), "journal-{0}.json".format(plan_id[:12]))


def _quarantine_dir(root, plan_id):
    return os.path.join(metadata_dir(root), "quarantine-{0}".format(plan_id[:12]))


def _report_path(root, plan_id):
    return os.path.join(metadata_dir(root), "report-{0}.json".format(plan_id[:12]))


def _find_journals(root):
    directory = metadata_dir(root)
    if not os.path.isdir(directory):
        return []
    paths = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("journal-") and name.endswith(".json")
    ]
    journals = []
    for path in sorted(paths):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                journals.append((path, json.load(handle)))
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable retention journal %s", path)
    return journals


def _new_journal(plan):
    return {
        "format_version": RETENTION_FORMAT_VERSION,
        "plan_id": plan["plan_id"],
        "root": plan["root"],
        "state": "moving",
        "quarantine": os.path.relpath(
            _quarantine_dir(plan["root"], plan["plan_id"]), plan["root"]
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "path": entry["path"],
                "size_bytes": entry["size_bytes"],
                "category": entry["category"],
                "reason": entry["reason"],
                "type": entry["type"],
                "status": "pending",
            }
            for entry in plan["entries"]
        ],
    }


def _save_journal(root, journal):
    journal["updated_at"] = datetime.now(timezone.utc).isoformat()
    os.makedirs(metadata_dir(root), exist_ok=True)
    _atomic_write_json(_journal_path(root, journal["plan_id"]), journal)


def _stage_entries(root, journal):
    """Move every pending file entry into quarantine; remove empty dirs.

    File moves use os.rename, which is atomic within the same filesystem.
    Nothing is unlinked in this phase, so a crash leaves all data restorable
    in the quarantine directory.
    """
    quarantine_root = _quarantine_dir(root, journal["plan_id"])
    for entry in journal["entries"]:
        if entry["status"] == "moved":
            continue

        rel_path = entry["path"].rstrip("/")
        source = os.path.join(root, rel_path)

        if entry["type"] == "directory":
            # Empty containers are removed in place only after their files
            # (handled by entry ordering). They carry no data.
            try:
                os.rmdir(source)
            except FileNotFoundError:
                pass
            entry["status"] = "moved"
            _save_journal(root, journal)
            continue

        destination = os.path.join(quarantine_root, rel_path)
        destination_rel = os.path.relpath(destination, root)

        if os.path.exists(destination) and not os.path.exists(source):
            # Already staged during a previous attempt.
            entry["status"] = "moved"
            entry["quarantine_path"] = destination_rel
            _save_journal(root, journal)
            continue

        try:
            stat_result = os.lstat(source)
        except FileNotFoundError:
            raise RetentionError(
                "Planned file vanished before it could be staged: {0}".format(
                    entry["path"]
                )
            )

        if not stat.S_ISREG(stat_result.st_mode):
            raise RetentionError(
                "Refusing to remove non-regular file: {0}".format(entry["path"])
            )
        if stat_result.st_size != entry["size_bytes"]:
            raise RetentionError(
                "File changed since the plan was approved: {0} "
                "(planned {1} bytes, now {2}); re-run --tidy".format(
                    entry["path"], entry["size_bytes"], stat_result.st_size
                )
            )

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        os.replace(source, destination)
        entry["status"] = "moved"
        entry["quarantine_path"] = destination_rel
        _save_journal(root, journal)


def _purge_quarantine(root, journal):
    quarantine_root = _quarantine_dir(root, journal["plan_id"])
    shutil.rmtree(quarantine_root, ignore_errors=True)
    if os.path.exists(quarantine_root):
        raise RetentionError(
            "Unable to fully remove quarantine directory {0}; "
            "no data was lost, re-run to retry.".format(quarantine_root)
        )


def _write_report(root, journal, recovered=False):
    removed = [entry for entry in journal["entries"] if entry["status"] == "moved"]
    report = {
        "plan_id": journal["plan_id"],
        "root": root,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "recovered_after_interruption": recovered,
        "removed_count": len(removed),
        "freed_bytes": sum(
            entry["size_bytes"] for entry in removed if entry["type"] == "file"
        ),
        "entries": [
            {
                "path": entry["path"],
                "size_bytes": entry["size_bytes"],
                "category": entry["category"],
                "reason": entry["reason"],
                "type": entry["type"],
            }
            for entry in removed
        ],
    }
    path = _report_path(root, journal["plan_id"])
    _atomic_write_json(path, report)
    return report, path


def _run_journal_to_completion(root, journal, recovered):
    """Drive an existing journal through moving -> committed -> purged."""
    if journal["state"] == "moving":
        moved = sum(1 for e in journal["entries"] if e["status"] == "moved")
        if moved:
            logger.info(
                "Resuming retention plan %s (%d/%d entr%s already staged)",
                journal["plan_id"][:12],
                moved,
                len(journal["entries"]),
                "y" if len(journal["entries"]) == 1 else "ies",
            )
        else:
            logger.info(
                "Applying retention plan %s (%d entr%s)",
                journal["plan_id"][:12],
                len(journal["entries"]),
                "y" if len(journal["entries"]) == 1 else "ies",
            )
        _stage_entries(root, journal)
        journal["state"] = "committed"
        _save_journal(root, journal)

    if journal["state"] == "committed":
        # Every planned file is staged: only now is it safe to actually
        # release the bytes.
        _purge_quarantine(root, journal)
        journal["state"] = "purged"
        _save_journal(root, journal)

    report, report_path = _write_report(root, journal, recovered=recovered)
    with contextlib.suppress(FileNotFoundError):
        os.remove(_journal_path(root, journal["plan_id"]))
    return report, report_path


def _confirm(plan, assume_yes, prompt_func):
    if assume_yes:
        return True
    question = "Apply retention plan {0}: remove {1} entr{2} ({3})? [y/N]: ".format(
        plan["plan_id"][:12],
        plan["total_remove_count"],
        "y" if plan["total_remove_count"] == 1 else "ies",
        format_bytes(plan["total_remove_bytes"]),
    )
    try:
        answer = prompt_func(question)
    except EOFError:
        answer = ""
    return str(answer).strip().lower() in ("y", "yes")


def apply_retention_plan(root, plan, assume_yes=False, prompt_func=input):
    """Execute an approved plan under the exclusive directory lock.

    Returns the report dict. Raises on failure; in every failure mode data is
    either fully present or staged in quarantine (never partially unlinked).
    """
    root = os.path.realpath(root)

    with backup_lock(root, "tidy"):
        # First finish anything a previous, interrupted run left behind so
        # repeated executions converge to one stable result.
        for journal_path, existing in _find_journals(root):
            if existing.get("plan_id") == plan["plan_id"]:
                continue
            logger.info(
                "Completing previously interrupted retention journal %s", journal_path
            )
            _run_journal_to_completion(root, existing, recovered=True)

        journals = dict(
            (journal.get("plan_id"), (path, journal))
            for path, journal in _find_journals(root)
        )

        if plan["plan_id"] in journals:
            # Resumption of *this* plan: the journal is the durable record of
            # the approval, so no fresh confirmation is needed.
            _, journal = journals[plan["plan_id"]]
            return _run_journal_to_completion(root, journal, recovered=True)[0]

        # Fresh execution: the approved plan must still describe the tree
        # exactly. A concurrent change (possible only without the lock
        # contract) invalidates it.
        fresh_plan = build_retention_plan(root)
        if fresh_plan["plan_id"] != plan["plan_id"]:
            raise PlanMismatchError(
                "The backup directory has changed since this plan was "
                "generated ({0} vs {1}). Re-run --tidy and review the new "
                "plan before applying.".format(
                    plan["plan_id"][:12], fresh_plan["plan_id"][:12]
                )
            )

        if not plan["entries"]:
            logger.info(
                "Retention plan %s has no removable entries, nothing to do",
                plan["plan_id"][:12],
            )
            return {
                "plan_id": plan["plan_id"],
                "root": root,
                "removed_count": 0,
                "freed_bytes": 0,
                "entries": [],
            }

        if not _confirm(plan, assume_yes, prompt_func):
            raise RetentionError("Retention plan not confirmed, no changes made")

        os.makedirs(metadata_dir(root), exist_ok=True)
        journal = _new_journal(plan)
        _save_journal(root, journal)
        try:
            report, _ = _run_journal_to_completion(root, journal, recovered=False)
        except BaseException:
            # Leave the journal in place so the next invocation can resume;
            # staged data remains safe in quarantine.
            logger.error(
                "Retention failed before completion. Re-run --tidy-apply with "
                "the same plan to resume; staged files are held in %s",
                _quarantine_dir(root, plan["plan_id"]),
            )
            raise
        return report


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def run_retention_cli(args, prompt_func=input):
    """Entry point used by cli.main for --tidy / --tidy-apply.

    Exits the process on user-facing failures so the thin CLI wrappers
    (bin/github-backup, python -m github_backup) propagate a non-zero status.
    """
    root = os.path.realpath(args.output_directory)
    if not os.path.isdir(root):
        logger.error("Output directory does not exist: %s", root)
        sys.exit(1)

    try:
        if args.tidy_apply:
            plan = load_approved_plan(args.tidy_apply, root)
            report = apply_retention_plan(
                root, plan, assume_yes=args.assume_yes, prompt_func=prompt_func
            )
            logger.info(
                "Retention complete: removed %d entr%s, freed %s. Report: %s",
                report.get("removed_count", 0),
                "y" if report.get("removed_count") == 1 else "ies",
                format_bytes(report.get("freed_bytes", 0)),
                _report_path(root, report["plan_id"]),
            )
        else:
            plan = build_retention_plan(root)
            plan_path = write_plan_file(root, plan)
            for line in summarize_plan(plan).splitlines():
                logger.info(line)
            logger.info("Plan written to %s", plan_path)
            if plan["entries"]:
                logger.info(
                    "Review the plan, then execute with: "
                    "github-backup --tidy-apply %s (add --yes to skip the "
                    "interactive confirmation prompt)",
                    plan_path,
                )
    except RetentionError as exc:
        logger.error(str(exc))
        sys.exit(1)
