"""Stable backup identity across repository renames and ownership transfers.

GitHub identifies repositories by a numeric ``id`` that never changes when a
repository is renamed or transferred to another owner. Only ``full_name``
(``owner/name``) changes. Gists are already keyed by their stable gist id.

The on-disk layout keeps its historical, kind-specific naming rules:

- ``repositories/{name}``            for repositories of the backed-up account
- ``starred/{owner}/{name}``         for starred repositories
- ``gists/{gist_id}``                for gists (identity is the path already)

A small marker file (``MARKER_FILENAME``) in each repository/starred backup
directory records the stable identity, so a renamed/transferred repository
keeps using its original backup directory instead of getting a second,
partial copy. Two different repositories that merely share a name can never
be merged: the identity key is the GitHub id, never the path name.

The :class:`BackupLocationResolver` is used while backing up and never moves
data. The migration helpers below provide an explicit preview/apply workflow
for reorganising directories created before markers existed (or split by
older backup runs).
"""

from __future__ import annotations

import codecs
import json
import logging
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlparse

logger = logging.getLogger(__name__)

MARKER_FILENAME = ".github-backup.json"
MARKER_VERSION = 1

REPOSITORIES_DIRNAME = "repositories"
STARRED_DIRNAME = "starred"
GISTS_DIRNAME = "gists"

KIND_REPOSITORY = "repository"
KIND_STARRED = "starred"
KIND_GIST = "gist"

# ``home`` is the single directory that currently holds a repository's
# backup; ``shadow`` directories contain older data awaiting migration into
# the home and must never be treated as current by a backup run.
ROLE_HOME = "home"
ROLE_SHADOW = "shadow"

# Checkpoint files whose value is an ISO timestamp boundary. When two copies
# of the same repository are merged, the older boundary must win so data
# produced in the overlap window is re-fetched.
CHECKPOINT_FILENAMES = frozenset({"last_update", "reviews_last_update"})

# Files that differ between two copies of an item are moved here (inside the
# home) instead of being deleted, so migration never destroys data.
DUPLICATES_DIRNAME = ".migration-duplicates"

# Resolver decisions, returned for logging.
REUSE_HOME = "reuse-home"
REUSE_CLAIMED = "reuse-claimed"
NEW_DIRECTORY = "new-directory"
ISOLATED_CONFLICT = "isolated-conflict"
DEMOTED_DUPLICATE = "demoted-duplicate"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def repository_kind(repository):
    """Return the backup kind for an item from the listing.

    Gists are checked first because starred gists carry both flags, and on
    disk gists use the gist id regardless of who starred them.
    """
    if repository.get("is_gist"):
        return KIND_GIST
    if repository.get("is_starred"):
        return KIND_STARRED
    return KIND_REPOSITORY


def identity_key(repository):
    """Stable (kind, id) key for a listing item."""
    return repository_kind(repository), repository["id"]


def canonical_item_path(output_directory, repository):
    """Canonical backup directory for an item given its current API record."""
    kind = repository_kind(repository)
    if kind == KIND_GIST:
        return os.path.join(output_directory, GISTS_DIRNAME, str(repository["id"]))
    if kind == KIND_STARRED:
        return os.path.join(
            output_directory,
            STARRED_DIRNAME,
            repository["owner"]["login"],
            repository["name"],
        )
    return os.path.join(output_directory, REPOSITORIES_DIRNAME, repository["name"])


def canonical_path_for_record(output_directory, kind, record):
    """Canonical path for a verified API record (used by migration)."""
    if kind == KIND_GIST:
        return os.path.join(output_directory, GISTS_DIRNAME, str(record["id"]))
    if kind == KIND_STARRED:
        owner = record.get("owner") or {}
        return os.path.join(
            output_directory, STARRED_DIRNAME, owner.get("login", ""), record["name"]
        )
    return os.path.join(output_directory, REPOSITORIES_DIRNAME, record["name"])


# ---------------------------------------------------------------------------
# Marker read/write
# ---------------------------------------------------------------------------


def build_marker(
    repository, kind, role=ROLE_HOME, merged_into=None, first_seen_at=None
):
    return {
        "version": MARKER_VERSION,
        "kind": kind,
        "id": repository["id"],
        "node_id": repository.get("node_id"),
        "owner": (repository.get("owner") or {}).get("login"),
        "name": repository.get("name"),
        "full_name": repository.get("full_name")
        or "{0}/{1}".format(
            (repository.get("owner") or {}).get("login"), repository.get("name")
        ),
        "role": role,
        "merged_into": merged_into,
        "first_seen_at": first_seen_at or utc_now(),
        "last_seen_at": utc_now(),
    }


def load_marker(repo_cwd):
    """Return the parsed marker dict for a backup directory, else None."""
    marker_path = os.path.join(repo_cwd, MARKER_FILENAME)
    try:
        with codecs.open(marker_path, "r", encoding="utf-8") as f:
            marker = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        logger.warning("Ignoring unreadable marker %s: %s", marker_path, e)
        return None
    if not isinstance(marker, dict) or "id" not in marker or "kind" not in marker:
        logger.warning("Ignoring invalid marker %s", marker_path)
        return None
    return marker


def save_marker(repo_cwd, marker):
    """Atomically write a marker into a backup directory (it must exist)."""
    marker = dict(marker)
    marker["last_seen_at"] = utc_now()
    marker_path = os.path.join(repo_cwd, MARKER_FILENAME)
    temp_path = marker_path + ".temp"
    with codecs.open(temp_path, "w", encoding="utf-8") as f:
        json.dump(marker, f, ensure_ascii=False, sort_keys=True, indent=2)
    os.replace(temp_path, marker_path)


def update_repository_marker(repo_cwd, repository, kind):
    """Write/refresh the home marker for a directory we are backing up to.

    The stable id is never changed: a marker recording another repository
    means the resolver sent us to the wrong place, and the marker is left
    untouched rather than overwritten.
    """
    existing = load_marker(repo_cwd)
    if existing is not None:
        if existing.get("id") != repository["id"] or existing.get("kind") != kind:
            logger.warning(
                "Refusing to overwrite marker for %s with identity %s/%s",
                repo_cwd,
                kind,
                repository["id"],
            )
            return
        marker = build_marker(
            repository,
            kind,
            role=existing.get("role", ROLE_HOME),
            merged_into=existing.get("merged_into"),
            first_seen_at=existing.get("first_seen_at"),
        )
    else:
        marker = build_marker(repository, kind)
    save_marker(repo_cwd, marker)


# ---------------------------------------------------------------------------
# Backup directory scanning
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Entry:
    """One backed-up item directory on disk.

    Equality is identity-based (``eq=False``): two scanned directories that
    happen to hold the same field values are still distinct entries.
    """

    path: str
    relpath: str
    kind: str
    id: object = None
    role: str | None = None
    marker: dict | None = None
    # Set when a markerless legacy directory was verified against the API.
    verified_id: object = None

    @property
    def is_legacy(self):
        return self.marker is None

    @property
    def is_home(self):
        return self.role == ROLE_HOME


def _entry_from_marker(path, relpath, kind):
    marker = load_marker(path)
    if marker is None:
        return Entry(path=path, relpath=relpath, kind=kind)
    return Entry(
        path=path,
        relpath=relpath,
        kind=marker.get("kind", kind),
        id=marker.get("id"),
        role=marker.get("role", ROLE_HOME),
        marker=marker,
    )


def scan_entries(output_directory):
    """Scan an output directory, returning (entries_by_relpath, by_identity).

    Gists are indexed by gist id (their directory name, cross-checked with
    gist.json when present). Repository/starred directories without a marker
    are indexed as legacy entries and carry no identity until claimed.
    """
    entries = {}

    repositories_root = os.path.join(output_directory, REPOSITORIES_DIRNAME)
    if os.path.isdir(repositories_root):
        for name in os.listdir(repositories_root):
            path = os.path.join(repositories_root, name)
            if not os.path.isdir(path) or name.startswith("."):
                continue
            relpath = os.path.relpath(path, output_directory)
            entries[relpath] = _entry_from_marker(path, relpath, KIND_REPOSITORY)

    starred_root = os.path.join(output_directory, STARRED_DIRNAME)
    if os.path.isdir(starred_root):
        for owner in os.listdir(starred_root):
            owner_path = os.path.join(starred_root, owner)
            if not os.path.isdir(owner_path) or owner.startswith("."):
                continue
            for name in os.listdir(owner_path):
                path = os.path.join(owner_path, name)
                if not os.path.isdir(path) or name.startswith("."):
                    continue
                relpath = os.path.relpath(path, output_directory)
                entries[relpath] = _entry_from_marker(path, relpath, KIND_STARRED)

    gists_root = os.path.join(output_directory, GISTS_DIRNAME)
    if os.path.isdir(gists_root):
        for gist_id in os.listdir(gists_root):
            path = os.path.join(gists_root, gist_id)
            if not os.path.isdir(path) or gist_id.startswith("."):
                continue
            relpath = os.path.relpath(path, output_directory)
            stored_id = gist_id
            gist_json = os.path.join(path, "gist.json")
            if os.path.exists(gist_json):
                try:
                    with codecs.open(gist_json, "r", encoding="utf-8") as f:
                        stored_id = json.load(f).get("id", gist_id)
                except (OSError, ValueError):
                    stored_id = gist_id
            entries[relpath] = Entry(
                path=path,
                relpath=relpath,
                kind=KIND_GIST,
                id=stored_id,
                role=ROLE_HOME,
            )

    by_key = defaultdict(list)
    for entry in entries.values():
        identity = (
            entry.verified_id
            if entry.is_legacy and entry.verified_id is not None
            else entry.id
        )
        if identity is not None:
            by_key[(entry.kind, identity)].append(entry)
    return entries, by_key


def directory_size(path):
    """Total size of regular files below path (best effort)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _prefer_existing(entry_a, entry_b):
    """Pick the directory that more plausibly holds the long history."""
    size_a = directory_size(entry_a.path)
    size_b = directory_size(entry_b.path)
    if size_a != size_b:
        return entry_a if size_a > size_b else entry_b
    # Tie: keep the older directory.
    try:
        time_a = os.path.getmtime(entry_a.path)
        time_b = os.path.getmtime(entry_b.path)
    except OSError:
        return entry_a
    return entry_a if time_a <= time_b else entry_b


# ---------------------------------------------------------------------------
# Git remote evidence for markerless directories
# ---------------------------------------------------------------------------


def _parse_git_config_remote(config_text):
    for line in config_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("url") and "=" in stripped:
            value = stripped.split("=", 1)[1].strip()
            if value:
                return value
    return None


def read_clone_owner_name(repo_cwd):
    """Return (owner, name) recorded in the clone's origin URL, else None.

    Works for both regular clones (``repository/.git/config``) and bare
    mirrors (``repository/config``). Credentials and the ``.git`` suffix are
    stripped. Only the last two path components matter, so GitHub Enterprise
    hosts work without configuration.
    """
    config_paths = [
        os.path.join(repo_cwd, "repository", ".git", "config"),
        os.path.join(repo_cwd, "repository", "config"),
    ]
    remote_url = None
    for config_path in config_paths:
        try:
            with codecs.open(config_path, "r", encoding="utf-8", errors="replace") as f:
                remote_url = _parse_git_config_remote(f.read())
        except OSError:
            continue
        if remote_url:
            break
    if not remote_url:
        return None

    parsed = urlparse(remote_url)
    path = unquote(parsed.path) if parsed.scheme else remote_url.split(":", 1)[-1]
    components = [component for component in path.strip("/").split("/") if component]
    if len(components) < 2:
        return None
    owner, name = components[-2], components[-1]
    if name.endswith(".git"):
        name = name[:-4]
    if not owner or not name:
        return None
    return owner, name


def legacy_entry_owner_name(entry, backup_user):
    """Best (owner, name) guess for the repository a legacy entry once was.

    The clone remote is authoritative when present; otherwise the path
    layout gives the answer (starred stores owner/name, repositories stores
    only the name under the backed-up account).
    """
    remote = read_clone_owner_name(entry.path)
    if remote:
        return remote
    parts = entry.relpath.split(os.sep)
    if entry.kind == KIND_STARRED and len(parts) >= 3:
        return parts[-2], parts[-1]
    if entry.kind == KIND_REPOSITORY and len(parts) >= 2:
        return backup_user, parts[-1]
    return None


# ---------------------------------------------------------------------------
# Backup-time location resolution
# ---------------------------------------------------------------------------


class BackupLocationResolver:
    """Resolve where a listing item should be backed up.

    The resolver never merges directories on name similarity alone: identity
    is the GitHub id. When a name collision involves two different ids, the
    newcomer is isolated in a suffixed sibling directory instead of touching
    the existing backup.
    """

    def __init__(self, output_directory, verifier=None):
        self.output_directory = output_directory
        # verifier(kind, entry, repository) -> True (same), False (different
        # repository) or None (cannot tell). Only consulted for markerless
        # directories without a usable clone remote.
        self.verifier = verifier
        self.entries, self.by_key = scan_entries(output_directory)

    def _register(self, entry, identity=None):
        self.entries[entry.relpath] = entry
        if identity is None:
            identity = entry.verified_id if entry.is_legacy else entry.id
        if identity is not None:
            self.by_key[(entry.kind, identity)].append(entry)

    def _demote(self, entry, home):
        """Mark a duplicate home directory as a shadow of the real home."""
        marker = dict(entry.marker or {})
        marker.update(
            {
                "version": MARKER_VERSION,
                "kind": entry.kind,
                "id": entry.id,
                "role": ROLE_SHADOW,
                "merged_into": home.relpath,
            }
        )
        save_marker(entry.path, marker)
        entry.marker = marker
        entry.role = ROLE_SHADOW
        logger.warning(
            "Directory %s is a duplicate backup of %s; marking it as a shadow "
            "(run --migrate to consolidate). Backing up to %s instead.",
            entry.relpath,
            home.relpath,
            home.relpath,
        )

    def _unique_home(self, homes):
        home = homes[0]
        for candidate in homes[1:]:
            home = _prefer_existing(home, candidate)
        for duplicate in homes:
            if duplicate is not home:
                self._demote(duplicate, home)
        return home

    def _conflict_sibling(self, canonical, kind, repository):
        """Choose a non-colliding path for a different-id same-name repo."""
        identity = repository["id"]
        base = "{0}.{1}".format(canonical, identity)
        candidate = base
        counter = 1
        while os.path.exists(candidate):
            relpath = os.path.relpath(candidate, self.output_directory)
            existing = self.entries.get(relpath)
            if existing is not None and existing.id == identity:
                return candidate
            candidate = "{0}.{1}".format(base, counter)
            counter += 1
        return candidate

    def _claim_legacy(self, entry, repository, kind):
        marker = build_marker(repository, kind, role=ROLE_HOME)
        save_marker(entry.path, marker)
        entry.marker = marker
        entry.role = ROLE_HOME
        entry.id = repository["id"]
        entry.verified_id = None
        self.by_key[(kind, repository["id"])].append(entry)

    def resolve(self, repository):
        """Return (repo_cwd, reason) for a repository or starred listing item."""
        kind = repository_kind(repository)
        if kind == KIND_GIST:
            raise ValueError("gists are resolved via canonical_item_path()")

        canonical = canonical_item_path(self.output_directory, repository)

        identity = repository.get("id")
        if identity is None:
            # Records without a stable id (incomplete fixtures/records) fall
            # back to the historical name-based behaviour.
            return canonical, NEW_DIRECTORY
        key = (kind, identity)
        canonical_rel = os.path.relpath(canonical, self.output_directory)

        homes = [entry for entry in self.by_key.get(key, []) if entry.is_home]
        if homes:
            home = homes[0] if len(homes) == 1 else self._unique_home(homes)
            reason = REUSE_HOME
            if home.relpath != canonical_rel:
                logger.info(
                    "Repository %s (id %s) was renamed/transferred; continuing "
                    "backup in existing directory %s",
                    repository["full_name"],
                    identity,
                    home.relpath,
                )
            return home.path, reason

        existing = self.entries.get(canonical_rel)
        if existing is None:
            # Canonical location is free. A fresh directory is used and the
            # marker is written by the backup loop once it exists.
            entry = Entry(
                path=canonical,
                relpath=canonical_rel,
                kind=kind,
                id=identity,
                role=ROLE_HOME,
                marker=build_marker(repository, kind),
            )
            self._register(entry)
            return canonical, NEW_DIRECTORY

        # The canonical directory is already in use.
        if not existing.is_legacy:
            if existing.id == identity and existing.role == ROLE_SHADOW:
                # Should be rare: promote rather than create a sibling.
                existing.marker["role"] = ROLE_HOME
                existing.marker["merged_into"] = None
                save_marker(existing.path, existing.marker)
                existing.role = ROLE_HOME
                return existing.path, REUSE_HOME
            return self._isolate(canonical, kind, repository, existing)

        # Markerless directory from an older release. Only adopt it when its
        # identity can be confirmed; same name alone is not enough (a deleted
        # repository's name may have been reused for a different project).
        remote = read_clone_owner_name(canonical)
        if remote is not None:
            same_owner = remote[0].lower() == (repository["owner"]["login"].lower())
            same_name = remote[1].lower() == repository["name"].lower()
            if same_owner and same_name:
                self._claim_legacy(existing, repository, kind)
                return canonical, REUSE_CLAIMED
            return self._isolate(
                canonical,
                kind,
                repository,
                existing,
                detail="existing clone remote points to {0}/{1}".format(*remote),
            )

        if self.verifier is not None:
            verdict = self.verifier(kind, existing, repository)
            if verdict is True:
                self._claim_legacy(existing, repository, kind)
                return canonical, REUSE_CLAIMED
            if verdict is False:
                return self._isolate(
                    canonical,
                    kind,
                    repository,
                    existing,
                    detail="API identity does not match",
                )

        # No evidence either way. Preserve pre-marker behaviour (use the
        # name-based directory) rather than quarantining on every transient
        # API failure; the explicit --migrate workflow verifies identity.
        logger.warning(
            "Adopting existing directory %s for %s without a verified identity "
            "marker; run --migrate to verify pre-existing backups",
            canonical_rel,
            repository["full_name"],
        )
        self._claim_legacy(existing, repository, kind)
        return canonical, REUSE_CLAIMED

    def _isolate(self, canonical, kind, repository, existing, detail=None):
        target = self._conflict_sibling(canonical, kind, repository)
        message = (
            "Directory {0} already belongs to another repository"
            " and will not be merged with {1}".format(
                os.path.relpath(canonical, self.output_directory),
                repository["full_name"],
            )
        )
        if detail:
            message += " ({0})".format(detail)
        logger.warning(message)
        logger.warning("Backing %s up at %s instead", repository["full_name"], target)

        relpath = os.path.relpath(target, self.output_directory)
        entry = Entry(
            path=target,
            relpath=relpath,
            kind=kind,
            id=repository["id"],
            role=ROLE_HOME,
            marker=build_marker(repository, kind),
        )
        self._register(entry)
        return target, ISOLATED_CONFLICT


# ---------------------------------------------------------------------------
# Migration planning
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    entry: Entry
    role: str
    merged_into: str | None = None
    # Current API record, so a markerless entry can be claimed without
    # re-guessing owner/name at apply time.
    record: dict | None = None


@dataclass
class Merge:
    shadow: Entry
    home: Entry


@dataclass
class Rename:
    src: str
    dst: str


@dataclass
class MigrationPlan:
    claims: list = field(default_factory=list)
    merges: list = field(default_factory=list)
    renames: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    gists_seen: int = 0
    unchanged: int = 0

    @property
    def changes(self):
        return len(self.claims) + len(self.merges) + len(self.renames)


def live_records(repositories):
    """Map identity key -> API record for the current listing."""
    records = {}
    for repository in repositories:
        kind = repository_kind(repository)
        records[(kind, repository["id"])] = repository
    return records


def plan_migration(output_directory, repositories, verify=None, backup_user=None):
    """Build a :class:`MigrationPlan` without touching the filesystem.

    ``verify(kind, owner, name)`` returns the current API record (following
    GitHub's rename/transfer redirects) for a markerless directory, or None.
    """
    entries, by_key = scan_entries(output_directory)
    records = live_records(repositories)
    plan = MigrationPlan()

    # Verify markerless directories and learn their current identity.
    for entry in list(entries.values()):
        if entry.kind == KIND_GIST or not entry.is_legacy:
            continue
        guessed = legacy_entry_owner_name(entry, backup_user)
        if guessed is None or verify is None:
            continue
        record = verify(entry.kind, guessed[0], guessed[1])
        if record is not None and record.get("id") is not None:
            entry.verified_id = record["id"]
            by_key[(entry.kind, record["id"])].append(entry)
            records.setdefault((entry.kind, record["id"]), record)
        else:
            plan.notes.append(
                "Could not verify identity of {0}; left untouched".format(entry.relpath)
            )

    gist_keys = {key for key in by_key if key[0] == KIND_GIST}
    plan.gists_seen = len(gist_keys)

    group_keys = sorted(
        key for key in by_key if key[0] in (KIND_REPOSITORY, KIND_STARRED)
    )
    for key in group_keys:
        group = by_key[key]
        kind, identity = key
        record = records.get(key)
        if record is None:
            plan.notes.append(
                "No current API record for {0} (id {1}); left untouched".format(
                    kind, identity
                )
            )
            continue

        canonical = canonical_path_for_record(output_directory, kind, record)
        canonical_rel = os.path.relpath(canonical, output_directory)
        canonical_entry = entries.get(canonical_rel)

        # Markerless entries only join the group once verified, so every
        # entry here has a known identity.
        verified_entries = group
        if not verified_entries:
            continue

        homes = [e for e in verified_entries if e.is_home]
        if homes:
            home = next((e for e in homes if e.relpath == canonical_rel), None)
            if home is None:
                home = homes[0]
                for candidate in homes[1:]:
                    home = _prefer_existing(home, candidate)
        elif canonical_entry in verified_entries:
            home = canonical_entry
        else:
            home = verified_entries[0]
            for candidate in verified_entries[1:]:
                home = _prefer_existing(home, candidate)

        shadows = [e for e in verified_entries if e is not home]

        # If a duplicate marker claims "home" too, demote it in the plan.
        for duplicate in [e for e in shadows if e.is_home]:
            plan.claims.append(
                Claim(duplicate, ROLE_SHADOW, home.relpath, record=record)
            )

        if home.is_legacy:
            plan.claims.append(Claim(home, ROLE_HOME, record=record))
        for shadow in shadows:
            if shadow.is_legacy:
                plan.claims.append(
                    Claim(shadow, ROLE_SHADOW, home.relpath, record=record)
                )
            elif (
                shadow.role == ROLE_SHADOW
                and shadow.marker.get("merged_into") != home.relpath
            ):
                plan.claims.append(
                    Claim(shadow, ROLE_SHADOW, home.relpath, record=record)
                )

        for shadow in shadows:
            plan.merges.append(Merge(shadow, home))

        if home.relpath != canonical_rel:
            # The target may be a shadow of this same group; the merge step
            # removes it before the rename. Anything else (a foreign marker
            # or an unverifiable directory) blocks the rename.
            target_blocked = (
                canonical_entry is not None and canonical_entry not in shadows
            )
            if target_blocked:
                plan.conflicts.append(
                    "Cannot rename {0} to {1}: target already exists and does "
                    "not belong to this repository".format(home.relpath, canonical_rel)
                )
            else:
                plan.renames.append(Rename(home.path, canonical))

    if plan.changes == 0 and not plan.conflicts and not plan.notes:
        plan.unchanged = len(group_keys)
    return plan


def render_plan(plan, output_directory, apply_changes):
    lines = []
    title = "Repository identity migration"
    lines.append(title)
    lines.append("=" * len(title))
    mode = (
        "APPLY"
        if apply_changes
        else "PREVIEW (no changes will be made; re-run with --migrate-apply to execute)"
    )
    lines.append("Mode: {0}".format(mode))
    lines.append("")

    for claim in plan.claims:
        if claim.role == ROLE_HOME:
            lines.append("[claim ] {0} as the current home".format(claim.entry.relpath))
        else:
            lines.append(
                "[claim ] {0} as shadow of {1}".format(
                    claim.entry.relpath, claim.merged_into
                )
            )
    for merge in plan.merges:
        lines.append(
            "[merge ] {0} into {1} (history, checkpoints and attachments kept)".format(
                merge.shadow.relpath, merge.home.relpath
            )
        )
    for rename in plan.renames:
        lines.append(
            "[rename] {0} -> {1}".format(
                os.path.relpath(rename.src, output_directory),
                os.path.relpath(rename.dst, output_directory),
            )
        )
    for conflict in plan.conflicts:
        lines.append("[conflict] {0}".format(conflict))
    for note in plan.notes:
        lines.append("[note  ] {0}".format(note))

    if plan.gists_seen:
        lines.append(
            "[gist  ] {0} gist backup(s) already keyed by gist id; nothing to do".format(
                plan.gists_seen
            )
        )

    lines.append("")
    lines.append(
        "Summary: {0} claim(s), {1} merge(s), {2} rename(s), {3} conflict(s), "
        "{4} note(s)".format(
            len(plan.claims),
            len(plan.merges),
            len(plan.renames),
            len(plan.conflicts),
            len(plan.notes),
        )
    )
    if not apply_changes and plan.changes:
        lines.append("")
        lines.append(
            "This was a preview. Re-run with --migrate-apply to perform these changes."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Migration execution (crash-safe and idempotent)
# ---------------------------------------------------------------------------


def _read_text(path):
    with codecs.open(path, "r", encoding="utf-8") as f:
        return f.read()


def _move_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError:
        # Cross-device move fallback.
        shutil.move(src, dst)


def _quarantine(home_path, shadow_relpath, relpath, src_path):
    target = os.path.join(home_path, DUPLICATES_DIRNAME, shadow_relpath, relpath)
    if os.path.exists(target):
        root, ext = os.path.splitext(target)
        counter = 1
        while os.path.exists("{0}.{1}{2}".format(root, counter, ext)):
            counter += 1
        target = "{0}.{1}{2}".format(root, counter, ext)
    _move_file(src_path, target)
    return target


def _is_checkpoint(relpath):
    return os.path.basename(relpath) in CHECKPOINT_FILENAMES


def _merge_checkpoint(src_path, dst_path):
    """Keep the older incremental boundary and drop the newer checkpoint."""
    try:
        src_value = _read_text(src_path).strip()
        dst_value = _read_text(dst_path).strip()
    except OSError:
        return False
    boundary = min(value for value in (src_value, dst_value) if value)
    with codecs.open(dst_path + ".temp", "w", encoding="utf-8") as f:
        f.write(boundary)
    os.replace(dst_path + ".temp", dst_path)
    os.remove(src_path)
    return True


def merge_directory(shadow_path, home_path):
    """Merge a shadow copy into its home, preserving everything.

    Checkpoint boundaries are combined conservatively; conflicting data
    files are quarantined rather than deleted. Returns the number of files
    moved/quarantined. Safe to re-run: anything already merged is simply
    absent on the next pass.
    """
    shadow_relpath = os.path.basename(shadow_path.rstrip(os.sep))
    moved = 0

    for root, dirs, files in os.walk(shadow_path):
        for filename in files:
            if filename == MARKER_FILENAME or filename.endswith(".temp"):
                continue
            src_path = os.path.join(root, filename)
            relpath = os.path.relpath(src_path, shadow_path)
            dst_path = os.path.join(home_path, relpath)

            if not os.path.exists(dst_path):
                _move_file(src_path, dst_path)
                moved += 1
                continue

            if _is_checkpoint(relpath):
                if not _merge_checkpoint(src_path, dst_path):
                    # Never let the tree cleanup below delete a boundary we
                    # could not combine; preserve it in the quarantine area.
                    _quarantine(home_path, shadow_relpath, relpath, src_path)
                moved += 1
                continue

            try:
                same = _read_text(src_path) == _read_text(dst_path)
            except (OSError, UnicodeDecodeError):
                same = False
            if same:
                os.remove(src_path)
                moved += 1
                continue

            if os.path.getmtime(src_path) > os.path.getmtime(dst_path):
                _quarantine(home_path, shadow_relpath, relpath, dst_path)
                _move_file(src_path, dst_path)
            else:
                _quarantine(home_path, shadow_relpath, relpath, src_path)
            moved += 1

    # Remove the now-empty shadow tree; stray unreadable files make this
    # fail loudly so the shadow marker stays in place and the migration can
    # be retried.
    shutil.rmtree(shadow_path)
    return moved


def _rename_directory(src, dst):
    """Rename a backup directory, handling case-only renames on case-insensitive
    filesystems (macOS default) via a two-step move through a temp name."""
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.normcase(src) == os.path.normcase(dst) and src != dst:
        temp = dst + ".identity-rename-tmp"
        counter = 1
        while os.path.exists(temp):
            temp = "{0}.{1}".format(dst + ".identity-rename-tmp", counter)
            counter += 1
        os.rename(src, temp)
        os.rename(temp, dst)
    else:
        os.rename(src, dst)


def _remove_empty_parents(path, stop_at):
    path = os.path.dirname(path)
    stop_at = os.path.abspath(stop_at)
    while os.path.abspath(path) != stop_at and os.path.isdir(path):
        try:
            os.rmdir(path)
        except OSError:
            break
        path = os.path.dirname(path)


def apply_claims(plan):
    """Phase 1 of :func:`apply_plan`: record home/shadow roles on disk.

    Separated so an interrupted migration can be inspected/resumed: once
    this finishes, exactly one directory per repository is marked current
    even if none of the data has moved yet.
    """
    for claim in plan.claims:
        marker = claim.entry.marker
        if marker is None:
            # Build from the verified current record rather than guessing
            # from the (possibly stale) on-disk path.
            record = claim.record or {}
            owner = record.get("owner") or {}
            marker = {
                "version": MARKER_VERSION,
                "kind": claim.entry.kind,
                "id": record.get("id", claim.entry.verified_id or claim.entry.id),
                "node_id": record.get("node_id"),
                "owner": owner.get("login"),
                "name": record.get("name"),
                "full_name": record.get("full_name"),
                "first_seen_at": utc_now(),
            }
        else:
            marker = dict(marker)
        marker["role"] = claim.role
        marker["merged_into"] = claim.merged_into
        save_marker(claim.entry.path, marker)
        claim.entry.marker = marker
        claim.entry.role = claim.role
        claim.entry.id = marker["id"]


def apply_plan(plan, output_directory):
    """Execute a migration plan.

    All role claims are written first, before any data moves: at every point
    in time each repository has exactly one directory that looks current, so
    an interrupted or failed migration can never leave two live copies.
    Re-running with a freshly planned plan resumes any unfinished work.
    """
    # Phase 1: claims (atomic per-file marker writes).
    apply_claims(plan)

    # Phase 2: consolidate shadows into homes.
    for merge in plan.merges:
        if not os.path.isdir(merge.shadow.path):
            continue
        logger.info("Merging %s into %s", merge.shadow.relpath, merge.home.relpath)
        merge_directory(merge.shadow.path, merge.home.path)

    # Phase 3: move homes to their current canonical names.
    for rename in plan.renames:
        if not os.path.isdir(rename.src):
            continue
        logger.info("Renaming %s to %s", rename.src, rename.dst)
        _rename_directory(rename.src, rename.dst)
        _remove_empty_parents(rename.src, output_directory)

    return {
        "claims": len(plan.claims),
        "merges": len(plan.merges),
        "renames": len(plan.renames),
        "conflicts": len(plan.conflicts),
    }


# ---------------------------------------------------------------------------
# API verification
# ---------------------------------------------------------------------------


def fetch_repository_record(fetch, api_host, owner, name):
    """GET /repos/{owner}/{name}; GitHub follows renames/transfers with a 301.

    Returns the current record (with the stable id) or None on any failure.
    """
    template = "https://{0}/repos/{1}/{2}".format(
        api_host, quote(owner, safe=""), quote(name, safe="")
    )
    try:
        data = fetch(template)
    except Exception as e:  # 404/451, rate limits, network errors
        logger.info("Could not verify %s/%s: %s", owner, name, e)
        return None
    if isinstance(data, list):
        return data[0] if data else None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def run_migration(args, output_directory, authenticated_user, apply_changes=False):
    """Preview (or, when requested, execute) identity-based migration.

    Returns a process exit code. Imports from :mod:`github_backup.github_backup`
    happen lazily to avoid a circular import at module load time.
    """
    from .github_backup import (
        get_github_api_host,
        retrieve_data,
        retrieve_repositories,
    )

    # The migration needs to see every live repository, regardless of the
    # resource include flags used for the backup itself. Starred/gist
    # listings are only forced when such backup directories already exist.
    migration_args = type(args)(**vars(args))
    migration_args.repository = None
    migration_args.all_starred = args.all_starred or os.path.isdir(
        os.path.join(output_directory, STARRED_DIRNAME)
    )
    migration_args.include_gists = args.include_gists or os.path.isdir(
        os.path.join(output_directory, GISTS_DIRNAME)
    )
    migration_args.include_starred_gists = migration_args.include_gists

    repositories = retrieve_repositories(migration_args, authenticated_user)

    api_host = get_github_api_host(args)

    def verify(kind, owner, name):
        return fetch_repository_record(
            lambda template: retrieve_data(args, template, paginated=False),
            api_host,
            owner,
            name,
        )

    plan = plan_migration(
        output_directory,
        repositories,
        verify=verify,
        backup_user=args.user,
    )
    print(render_plan(plan, output_directory, apply_changes))

    if apply_changes:
        apply_plan(plan, output_directory)
        if plan.conflicts:
            logger.warning(
                "Migration finished with %d unresolved conflict(s); review the "
                "output above",
                len(plan.conflicts),
            )
    return 0
