"""Offline consistency audit for github-backup output directories.

The auditor scans an existing backup directory *without* contacting GitHub
and *without* modifying anything. It looks for the kinds of damage that
file-count checks and log grepping cannot prove:

* truncated or otherwise unparseable JSON (issue/pull/discussion records,
  gist metadata, account dumps, attachment manifests, run records);
* missing metadata (resource directories whose index file was never
  written, records without identity fields, filename/record mismatches);
* attachment/record contradictions (a manifest claims a download
  succeeded but the file is gone or has a different size, a file on disk
  is referenced by no manifest entry, orphan attachment directories);
* checkpoint/resource contradictions (malformed checkpoints, stored items
  newer than their checkpoint, a checkpoint advanced by a run that
  recorded the resource as failed);
* leftovers of interrupted writes (``*.temp`` files) and incomplete
  clones;
* broken per-run archive indexes (``backup-runs/latest.json``).

Design rules:

* **Resilient** - one unreadable file or a permission error on one
  directory never aborts the scan; the problem is recorded as a finding
  and the remaining samples are checked.
* **Deterministic** - the report for an unchanged directory is identical
  on every run. Nothing depends on the current wall-clock time, inode
  numbers or dictionary iteration order.
* **Conservative** - resources that were simply never enabled, empty
  repositories, inaccessible gists without a clone, recorded download
  failures and legacy/old-version layouts are never reported as damage.
  Findings are graded ``error`` (the data on disk is inconsistent),
  ``warning`` (damage is likely but an interrupted run can explain it)
  and ``info`` (worth knowing, definitely not corruption).
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

SEVERITY_ORDER = (SEVERITY_ERROR, SEVERITY_WARNING, SEVERITY_INFO)
_SEVERITY_RANK = {name: index for index, name in enumerate(SEVERITY_ORDER)}

SCHEMA_VERSION = 1

# Directories that hold arbitrary user data (git clones) rather than files
# this tool writes atomically; temp files inside them must not be mistaken
# for interrupted backup writes.
_CLONE_DIR_NAMES = frozenset({"repository", "wiki", ".git"})

# Account-level JSON dumps, each a JSON array.
_ACCOUNT_LIST_FILES = frozenset(
    {"starred.json", "watched.json", "followers.json", "following.json"}
)

# Resource directory -> (manifest item_type, item timestamp key)
_ITEM_RESOURCES = {
    "issues": ("issue", "updated_at"),
    "pulls": ("pull", "updated_at"),
    "discussions": ("discussion", "updatedAt"),
}

_SINGLE_FILE_RESOURCES = {
    "labels": "labels.json",
    "hooks": "hooks.json",
}

_FINALIZED_STATUSES = frozenset(
    {"completed", "completed_with_errors", "failed", "interrupted", "skipped"}
)

_TEMP_SUFFIX_RE = re.compile(r"^(?P<base>.*)\.(?P<token>\d+-[0-9a-f]{8})\.temp$")
_DIGITS_RE = re.compile(r"^\d+$")
_ISO_TS_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$"
)


def parse_timestamp(value):
    """Parse the ISO-8601 timestamps used in checkpoints/records.

    Returns a timezone-aware ``datetime`` or ``None``. Pure and side-effect
    free so results are independent of the local clock.
    """
    from datetime import datetime, timezone

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or not _ISO_TS_RE.match(text):
        return None
    try:
        if text.endswith("Z"):
            # Fractional seconds are legal in GraphQL timestamps; drop them
            # for comparison, checkpoint granularity is whole seconds.
            body = text[:-1].split(".", 1)[0]
            parsed = datetime.strptime(body, "%Y-%m-%dT%H:%M:%S")
            return parsed.replace(tzinfo=timezone.utc)
        normalized = text if text[-3] == ":" else text[:-2] + ":" + text[-2:]
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def intended_target_for_temp(path):
    """Return the final path a ``*.temp`` file was meant to become.

    Both naming schemes are handled: the legacy ``path + ".temp"`` form and
    the process-unique ``path.<pid>-<hex>.temp`` form.
    """
    if not path.endswith(".temp"):
        return None
    stem = path[: -len(".temp")]
    match = _TEMP_SUFFIX_RE.match(path)
    if match:
        return match.group("base")
    return stem


class BackupScanner:
    """Walk a backup root and collect graded, machine-readable findings."""

    def __init__(self, output_directory):
        self.root = os.path.abspath(output_directory)
        self.findings = []
        self.scanned = {
            "repository_units": 0,
            "starred_units": 0,
            "gist_units": 0,
            "resource_dirs": 0,
            "json_files": 0,
            "attachment_manifests": 0,
            "run_records": 0,
            "temp_files": 0,
        }

    # ------------------------------------------------------------------
    # Finding / filesystem helpers
    # ------------------------------------------------------------------

    def _rel(self, path):
        """Stable, POSIX-style path relative to the scanned root."""
        return os.path.relpath(path, self.root).replace(os.sep, "/")

    def finding(self, severity, code, path, message, details=None):
        finding = {
            "severity": severity,
            "code": code,
            "path": self._rel(path) if path else ".",
            "message": message,
        }
        if details:
            finding["details"] = details
        self.findings.append(finding)

    def _children(self, path):
        """Return ``[(name, is_dir, is_file)]`` sorted by name.

        Symlinks and entries that cannot be stat'ed are skipped. Any
        error reading the directory is a finding, never a fatal error.
        """
        rows = []
        try:
            entries = os.scandir(path)
        except OSError as exc:
            self.finding(
                SEVERITY_WARNING,
                "scan_error",
                path,
                "directory could not be read: {0}".format(exc),
                {"error": type(exc).__name__},
            )
            return []
        try:
            for entry in entries:
                try:
                    is_dir = entry.is_dir(follow_symlinks=False)
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError as exc:
                    self.finding(
                        SEVERITY_WARNING,
                        "scan_error",
                        os.path.join(path, entry.name),
                        "entry could not be inspected: {0}".format(exc),
                        {"error": type(exc).__name__},
                    )
                    continue
                rows.append((entry.name, is_dir, is_file))
        finally:
            entries.close()
        rows.sort(key=lambda row: row[0])
        return rows

    def _is_dir(self, path):
        try:
            return os.path.isdir(path) and not os.path.islink(path)
        except OSError:
            return False

    def _is_file(self, path):
        try:
            return os.path.isfile(path)
        except OSError:
            return False

    def _stat(self, path):
        try:
            return os.stat(path)
        except OSError as exc:
            self.finding(
                SEVERITY_WARNING,
                "scan_error",
                path,
                "file could not be stat'ed: {0}".format(exc),
                {"error": type(exc).__name__},
            )
            return None

    def _read_bytes(self, path):
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except OSError as exc:
            self.finding(
                SEVERITY_WARNING,
                "scan_error",
                path,
                "file could not be read: {0}".format(exc),
                {"error": type(exc).__name__},
            )
            return None

    def _load_json(self, path):
        """Parse a JSON file, recording truncation/corruption findings.

        Returns the parsed value, or ``None`` when the file is missing or
        unparseable (a finding has been recorded in the latter case).
        """
        if not self._is_file(path):
            return None
        self.scanned["json_files"] += 1
        raw = self._read_bytes(path)
        if raw is None:
            return None
        if not raw.strip():
            self.finding(
                SEVERITY_ERROR,
                "json_invalid",
                path,
                "JSON file is empty (0 bytes of content)",
                {"reason": "empty_file"},
            )
            return None
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            self.finding(
                SEVERITY_ERROR,
                "json_invalid",
                path,
                "JSON file is not valid UTF-8: {0}".format(exc),
                {"reason": "encoding_error"},
            )
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            self.finding(
                SEVERITY_ERROR,
                "json_invalid",
                path,
                "JSON is truncated or malformed: {0}".format(exc.msg),
                {
                    "reason": "parse_error",
                    "line": exc.lineno,
                    "column": exc.colno,
                    "position": exc.pos,
                },
            )
            return None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def scan(self):
        if not os.path.isdir(self.root) or os.path.islink(self.root):
            self.finding(
                SEVERITY_ERROR,
                "backup_root_invalid",
                self.root,
                "backup directory does not exist or is not a directory",
            )
            return self._report()

        # Pass 1: interrupted-write debris anywhere the tool itself writes.
        self._scan_temp_files(self.root)

        # Pass 2: the backup contents, area by area.
        self._scan_legacy_checkpoint()
        self._scan_account()
        self._scan_gists()
        self._scan_unit_tree(os.path.join(self.root, "repositories"), depth=1)
        self._scan_unit_tree(os.path.join(self.root, "starred"), depth=2)
        self._scan_run_archive()

        return self._report()

    def _report(self):
        counts = {severity: 0 for severity in SEVERITY_ORDER}
        for item in self.findings:
            counts[item["severity"]] += 1
        ordered = sorted(
            self.findings,
            key=lambda item: (
                _SEVERITY_RANK[item["severity"]],
                item["path"],
                item["code"],
                item["message"],
            ),
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "root": self.root,
            "summary": {
                "scanned": dict(sorted(self.scanned.items())),
                "errors": counts[SEVERITY_ERROR],
                "warnings": counts[SEVERITY_WARNING],
                "infos": counts[SEVERITY_INFO],
                "ok": counts[SEVERITY_ERROR] == 0 and counts[SEVERITY_WARNING] == 0,
            },
            "findings": ordered,
        }

    # ------------------------------------------------------------------
    # Pass 1: leftover temp files
    # ------------------------------------------------------------------

    def _scan_temp_files(self, path):
        for name, is_dir, _ in self._children(path):
            child = os.path.join(path, name)
            if is_dir:
                if name in _CLONE_DIR_NAMES:
                    # Clones contain arbitrary user data; a ``*.temp`` file
                    # there is user content, not backup debris.
                    continue
                self._scan_temp_files(child)
                continue
            if name.endswith(".lock"):
                # flock sidecars are normal and may outlive the process.
                continue
            if name.endswith(".temp"):
                intended = intended_target_for_temp(child)
                self.scanned["temp_files"] += 1
                details = {}
                if intended:
                    details["intended_target"] = self._rel(intended)
                    details["intended_target_exists"] = os.path.exists(intended)
                self.finding(
                    SEVERITY_WARNING,
                    "temp_file_left",
                    child,
                    "temporary file left behind by an interrupted write",
                    details or None,
                )

    # ------------------------------------------------------------------
    # Root / account / gist areas
    # ------------------------------------------------------------------

    def _scan_legacy_checkpoint(self):
        path = os.path.join(self.root, "last_update")
        if not self._is_file(path):
            return
        raw = self._read_bytes(path)
        if raw is None:
            return
        value = raw.decode("utf-8", errors="replace").strip()
        if parse_timestamp(value) is None:
            self.finding(
                SEVERITY_WARNING,
                "checkpoint_invalid",
                path,
                "legacy checkpoint does not contain an ISO-8601 timestamp",
                {"checkpoint": "last_update", "value": value[:64]},
            )
        else:
            # Historical, expected while migrating per-resource checkpoints;
            # informational only and never corruption.
            self.finding(
                SEVERITY_INFO,
                "legacy_checkpoint_present",
                path,
                "legacy global incremental checkpoint is still present",
            )

    def _scan_account(self):
        account_dir = os.path.join(self.root, "account")
        if not self._is_dir(account_dir):
            return
        children = self._children(account_dir)
        meaningful = [
            (name, is_dir)
            for name, is_dir, _ in children
            if not name.endswith((".temp", ".lock"))
        ]
        if not meaningful:
            self.finding(
                SEVERITY_WARNING,
                "account_dir_empty",
                account_dir,
                "account directory exists but contains no account data files",
            )
            return
        for name, _, is_file in children:
            if is_file and name in _ACCOUNT_LIST_FILES:
                data = self._load_json(os.path.join(account_dir, name))
                if data is not None and not isinstance(data, list):
                    self.finding(
                        SEVERITY_WARNING,
                        "unexpected_json_type",
                        os.path.join(account_dir, name),
                        "account dump must be a JSON array",
                        {"actual_type": type(data).__name__},
                    )

    def _scan_gists(self):
        gists_dir = os.path.join(self.root, "gists")
        if not self._is_dir(gists_dir):
            return
        for gist_id, is_dir, _ in self._children(gists_dir):
            if not is_dir:
                continue
            self.scanned["gist_units"] += 1
            gist_dir = os.path.join(gists_dir, gist_id)
            gist_json = os.path.join(gist_dir, "gist.json")
            data = self._load_json(gist_json)
            if data is None:
                # Missing gist.json is missing metadata. A missing clone, on
                # the other hand, is normal for an inaccessible gist and is
                # not reported on its own.
                if not self._is_file(gist_json):
                    self.finding(
                        SEVERITY_WARNING,
                        "gist_metadata_missing",
                        gist_json,
                        "gist directory has no gist.json metadata file",
                    )
            else:
                self._check_identity_record(
                    gist_json, data, "id", gist_id, required=True
                )
            self._check_clone_dir(gist_dir, "repository")

    # ------------------------------------------------------------------
    # Repository units (repositories/<repo>, starred/<owner>/<repo>)
    # ------------------------------------------------------------------

    def _scan_unit_tree(self, top, depth):
        if not self._is_dir(top):
            return
        if depth == 1:
            for name, is_dir, _ in self._children(top):
                if is_dir:
                    self._scan_repository_unit(os.path.join(top, name))
            return
        for owner, owner_is_dir, _ in self._children(top):
            if not owner_is_dir:
                continue
            owner_dir = os.path.join(top, owner)
            for repo, repo_is_dir, _ in self._children(owner_dir):
                if repo_is_dir:
                    self._scan_repository_unit(os.path.join(owner_dir, repo))

    def _scan_repository_unit(self, unit_dir):
        rel = self._rel(unit_dir)
        if rel.startswith("starred/"):
            self.scanned["starred_units"] += 1
        else:
            self.scanned["repository_units"] += 1

        for name, is_dir, _ in self._children(unit_dir):
            if not is_dir:
                continue
            resource_dir = os.path.join(unit_dir, name)
            if name in _ITEM_RESOURCES:
                self.scanned["resource_dirs"] += 1
                self._scan_item_resource(unit_dir, name)
            elif name == "milestones":
                self.scanned["resource_dirs"] += 1
                self._scan_numbered_records(resource_dir, "milestone")
            elif name == "security-advisories":
                self.scanned["resource_dirs"] += 1
                self._scan_advisory_records(resource_dir)
            elif name in _SINGLE_FILE_RESOURCES:
                self.scanned["resource_dirs"] += 1
                self._scan_single_file_resource(
                    resource_dir, _SINGLE_FILE_RESOURCES[name]
                )
            elif name == "releases":
                self.scanned["resource_dirs"] += 1
                self._scan_releases(unit_dir)
            elif name in ("repository", "wiki"):
                self._check_clone_dir(unit_dir, name)
            # Unknown directories are left alone (future/old-version
            # resources, user files).

    def _scan_numbered_records(self, resource_dir, kind):
        for name, _, is_file in self._children(resource_dir):
            if not is_file or not name.endswith(".json"):
                continue
            stem = name[: -len(".json")]
            if not _DIGITS_RE.match(stem):
                continue
            path = os.path.join(resource_dir, name)
            data = self._load_json(path)
            if data is not None:
                self._check_identity_record(path, data, "number", stem, required=True)

    def _scan_advisory_records(self, resource_dir):
        for name, _, is_file in self._children(resource_dir):
            if not is_file or not name.endswith(".json"):
                continue
            stem = name[: -len(".json")]
            path = os.path.join(resource_dir, name)
            data = self._load_json(path)
            if data is not None:
                self._check_identity_record(path, data, "ghsa_id", stem, required=True)

    def _scan_single_file_resource(self, resource_dir, index_name):
        index_path = os.path.join(resource_dir, index_name)
        data = self._load_json(index_path)
        if data is None and not self._is_file(index_path):
            self.finding(
                SEVERITY_WARNING,
                "resource_index_missing",
                index_path,
                "{0} directory exists but its {1} data file was never written".format(
                    os.path.basename(resource_dir), index_name
                ),
            )
            return
        if data is not None and not isinstance(data, list):
            self.finding(
                SEVERITY_WARNING,
                "unexpected_json_type",
                index_path,
                "{0} must be a JSON array".format(index_name),
                {"actual_type": type(data).__name__},
            )

    def _check_identity_record(self, path, data, key, expected, required=True):
        """Validate an object record's identity field vs its filename."""
        if not isinstance(data, dict):
            self.finding(
                SEVERITY_WARNING,
                "unexpected_json_type",
                path,
                "record must be a JSON object",
                {"actual_type": type(data).__name__},
            )
            return
        if key not in data or data.get(key) in (None, ""):
            if required:
                self.finding(
                    SEVERITY_WARNING,
                    "record_missing_field",
                    path,
                    "record is missing required metadata field '{0}'".format(key),
                    {"field": key},
                )
            return
        if str(data.get(key)) != str(expected):
            self.finding(
                SEVERITY_WARNING,
                "record_identity_mismatch",
                path,
                "record field '{0}' ({1!r}) does not match its filename ({2!r})".format(
                    key, data.get(key), expected
                ),
                {"field": key, "stored": data.get(key), "expected": expected},
            )

    # ------------------------------------------------------------------
    # Issues / pulls / discussions
    # ------------------------------------------------------------------

    def _scan_item_resource(self, unit_dir, resource_name):
        item_type, timestamp_key = _ITEM_RESOURCES[resource_name]
        resource_dir = os.path.join(unit_dir, resource_name)
        item_stems = set()
        newest_item_ts = None

        for name, _, is_file in self._children(resource_dir):
            if not is_file or not name.endswith(".json"):
                continue
            stem = name[: -len(".json")]
            if not _DIGITS_RE.match(stem):
                continue
            item_stems.add(stem)
            path = os.path.join(resource_dir, name)
            data = self._load_json(path)
            if not isinstance(data, dict):
                if data is not None:
                    self.finding(
                        SEVERITY_WARNING,
                        "unexpected_json_type",
                        path,
                        "record must be a JSON object",
                        {"actual_type": type(data).__name__},
                    )
                continue
            if "number" not in data or data.get("number") in (None, ""):
                self.finding(
                    SEVERITY_WARNING,
                    "record_missing_field",
                    path,
                    "record is missing required metadata field 'number'",
                    {"field": "number"},
                )
            elif str(data.get("number")) != stem:
                self.finding(
                    SEVERITY_WARNING,
                    "record_identity_mismatch",
                    path,
                    "record number ({0!r}) does not match its filename ({1!r})".format(
                        data.get("number"), stem
                    ),
                    {
                        "field": "number",
                        "stored": data.get("number"),
                        "expected": int(stem),
                    },
                )
            timestamp = parse_timestamp(data.get(timestamp_key))
            if timestamp is not None and (
                newest_item_ts is None or timestamp > newest_item_ts
            ):
                newest_item_ts = timestamp

        self._scan_checkpoints(resource_dir, resource_name, item_stems, newest_item_ts)
        self._scan_attachments(resource_dir, item_type, item_stems)

    def _scan_checkpoints(self, resource_dir, resource_name, item_stems, newest_ts):
        # reviews_last_update is pulls-only and only format-checked.
        checkpoint_names = ["last_update"]
        if resource_name == "pulls":
            checkpoint_names.append("reviews_last_update")
        checkpoint_values = {}
        for checkpoint_name in checkpoint_names:
            path = os.path.join(resource_dir, checkpoint_name)
            if not self._is_file(path):
                continue
            raw = self._read_bytes(path)
            if raw is None:
                continue
            value = raw.decode("utf-8", errors="replace").strip()
            parsed = parse_timestamp(value)
            checkpoint_values[checkpoint_name] = (value, parsed, path)
            if parsed is None:
                self.finding(
                    SEVERITY_WARNING,
                    "checkpoint_invalid",
                    path,
                    "checkpoint does not contain an ISO-8601 timestamp",
                    {"checkpoint": checkpoint_name, "value": value[:64]},
                )

        checkpoint = checkpoint_values.get("last_update")
        if (
            checkpoint is not None
            and newest_ts is not None
            and checkpoint[1] is not None
            and newest_ts > checkpoint[1]
        ):
            # Items newer than the resource checkpoint can only be on disk
            # when the run that wrote them ended before advancing the
            # checkpoint; the next incremental round would refetch them.
            self.finding(
                SEVERITY_WARNING,
                "checkpoint_older_than_items",
                checkpoint[2],
                "checkpoint ({0}) is older than the newest stored item ({1}); "
                "the run that wrote these items likely did not finish".format(
                    checkpoint[0], newest_ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                ),
                {
                    "checkpoint": checkpoint[0],
                    "newest_item_timestamp": newest_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )

    def _scan_attachments(self, resource_dir, item_type, item_stems):
        attachments_root = os.path.join(resource_dir, "attachments")
        if not self._is_dir(attachments_root):
            return
        for number, is_dir, _ in self._children(attachments_root):
            if not is_dir or not _DIGITS_RE.match(number):
                continue
            item_dir = os.path.join(attachments_root, number)
            if number not in item_stems:
                self.finding(
                    SEVERITY_WARNING,
                    "attachment_orphan_dir",
                    item_dir,
                    "attachment directory has no matching {0} record {1}.json".format(
                        os.path.basename(resource_dir), number
                    ),
                    {"item_number": int(number)},
                )
            self._scan_attachment_item(item_dir, number, item_type)

    def _scan_attachment_item(self, item_dir, number, item_type):
        manifest_path = os.path.join(item_dir, "manifest.json")
        on_disk = set()
        for name, _, is_file in self._children(item_dir):
            if not is_file:
                continue
            if name == "manifest.json" or name.startswith("manifest_"):
                continue
            if name.endswith((".lock", ".temp")):
                continue
            on_disk.add(name)

        if not self._is_file(manifest_path):
            self.finding(
                SEVERITY_WARNING,
                "attachment_manifest_missing",
                manifest_path,
                "attachment directory exists but its manifest.json was never written",
                {"item_number": int(number)},
            )
            return

        manifest = self._load_json(manifest_path)
        if manifest is None:
            # json_invalid already reported; the on-disk files are still
            # cross-checked below via a conservative warning.
            if on_disk:
                self.finding(
                    SEVERITY_WARNING,
                    "attachment_unrecorded",
                    item_dir,
                    "files on disk cannot be reconciled against the unreadable "
                    "manifest: {0}".format(", ".join(sorted(on_disk))),
                    {"files": sorted(on_disk)},
                )
            return
        self.scanned["attachment_manifests"] += 1

        if not isinstance(manifest, dict):
            self.finding(
                SEVERITY_WARNING,
                "attachment_manifest_malformed",
                manifest_path,
                "manifest must be a JSON object",
                {"actual_type": type(manifest).__name__},
            )
            return

        entries = manifest.get("attachments")
        if not isinstance(entries, list):
            self.finding(
                SEVERITY_WARNING,
                "attachment_manifest_malformed",
                manifest_path,
                "manifest is missing an 'attachments' array",
            )
            entries = []

        manifest_number = manifest.get("item_number")
        if manifest_number is None:
            manifest_number = manifest.get("issue_number")
        if manifest_number is not None and str(manifest_number) != str(number):
            self.finding(
                SEVERITY_WARNING,
                "manifest_item_mismatch",
                manifest_path,
                "manifest item_number ({0!r}) does not match its directory ({1!r})".format(
                    manifest_number, number
                ),
                {
                    "field": "item_number",
                    "stored": manifest_number,
                    "expected": int(number),
                },
            )
        manifest_type = manifest.get("item_type") or manifest.get("issue_type")
        if manifest_type and manifest_type != item_type:
            self.finding(
                SEVERITY_WARNING,
                "manifest_item_mismatch",
                manifest_path,
                "manifest item_type ({0!r}) does not match resource ({1!r})".format(
                    manifest_type, item_type
                ),
                {"field": "item_type", "stored": manifest_type, "expected": item_type},
            )

        seen_urls = set()
        saved_files = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                self.finding(
                    SEVERITY_WARNING,
                    "attachment_manifest_malformed",
                    manifest_path,
                    "attachment entry #{0} is not a JSON object".format(index),
                    {"entry_index": index},
                )
                continue
            url = entry.get("url")
            if not url:
                self.finding(
                    SEVERITY_WARNING,
                    "attachment_manifest_malformed",
                    manifest_path,
                    "attachment entry #{0} has no url".format(index),
                    {"entry_index": index},
                )
                continue
            if url in seen_urls:
                self.finding(
                    SEVERITY_WARNING,
                    "manifest_duplicate_url",
                    manifest_path,
                    "attachment url appears more than once in the manifest: {0}".format(
                        url
                    ),
                    {"url": url},
                )
            seen_urls.add(url)

            if not entry.get("success"):
                # A recorded failure is truthful state, never damage.
                self.finding(
                    SEVERITY_INFO,
                    "attachment_recorded_failure",
                    manifest_path,
                    "manifest records a failed attachment download: {0}".format(url),
                    {
                        "url": url,
                        "http_status": entry.get("http_status"),
                        "error": entry.get("error"),
                    },
                )
                continue

            saved_as = entry.get("saved_as")
            if not saved_as:
                self.finding(
                    SEVERITY_WARNING,
                    "attachment_record_incomplete",
                    manifest_path,
                    "successful attachment entry has no saved_as: {0}".format(url),
                    {"url": url},
                )
                continue
            if saved_as in saved_files:
                self.finding(
                    SEVERITY_WARNING,
                    "manifest_duplicate_file",
                    manifest_path,
                    "more than one successful entry points at {0}".format(saved_as),
                    {"saved_as": saved_as},
                )
            saved_files.add(saved_as)
            target = os.path.join(item_dir, saved_as)
            if not self._is_file(target):
                self.finding(
                    SEVERITY_ERROR,
                    "attachment_file_missing",
                    target,
                    "manifest reports the attachment as downloaded but the file "
                    "is missing: {0}".format(url),
                    {"url": url},
                )
                continue
            expected_size = entry.get("size_bytes")
            if isinstance(expected_size, int) and expected_size > 0:
                stat = self._stat(target)
                if stat is not None and stat.st_size != expected_size:
                    self.finding(
                        SEVERITY_WARNING,
                        "attachment_size_mismatch",
                        target,
                        "file size {0} does not match manifest size_bytes {1}".format(
                            stat.st_size, expected_size
                        ),
                        {
                            "actual_size": stat.st_size,
                            "expected_size": expected_size,
                            "url": url,
                        },
                    )

        unrecorded = on_disk - saved_files
        for name in sorted(unrecorded):
            self.finding(
                SEVERITY_WARNING,
                "attachment_unrecorded",
                os.path.join(item_dir, name),
                "file on disk is referenced by no successful manifest entry",
            )

    # ------------------------------------------------------------------
    # Releases and their (optional) asset directories
    # ------------------------------------------------------------------

    def _scan_releases(self, unit_dir):
        releases_dir = os.path.join(unit_dir, "releases")
        release_data = {}
        for name, _, is_file in self._children(releases_dir):
            if not is_file or not name.endswith(".json"):
                continue
            stem = name[: -len(".json")]
            path = os.path.join(releases_dir, name)
            data = self._load_json(path)
            release_data[stem] = data
            if isinstance(data, dict):
                tag = data.get("tag_name")
                if not tag:
                    self.finding(
                        SEVERITY_WARNING,
                        "record_missing_field",
                        path,
                        "release record is missing required metadata field 'tag_name'",
                        {"field": "tag_name"},
                    )
                elif tag.replace("/", "__") != stem:
                    self.finding(
                        SEVERITY_WARNING,
                        "record_identity_mismatch",
                        path,
                        "release tag_name ({0!r}) does not match its filename ({1!r})".format(
                            tag, stem
                        ),
                        {"field": "tag_name", "stored": tag, "expected": stem},
                    )
            elif data is not None:
                self.finding(
                    SEVERITY_WARNING,
                    "unexpected_json_type",
                    path,
                    "release record must be a JSON object",
                    {"actual_type": type(data).__name__},
                )

        for name, is_dir, _ in self._children(releases_dir):
            if not is_dir:
                continue
            asset_dir = os.path.join(releases_dir, name)
            data = release_data.get(name)
            if not isinstance(data, dict):
                # No matching release JSON: the record is written before any
                # asset, so the asset directory alone means an interrupted
                # round or external files.
                self.finding(
                    SEVERITY_WARNING,
                    "release_assets_without_record",
                    asset_dir,
                    "release asset directory exists but {0}.json is missing".format(
                        name
                    ),
                )
                continue
            assets = data.get("assets")
            if not isinstance(assets, list):
                continue
            expected = {
                asset.get("name")
                for asset in assets
                if isinstance(asset, dict) and asset.get("name")
            }
            for asset_name, _, is_file in self._children(asset_dir):
                if not is_file or asset_name.endswith((".lock", ".temp")):
                    continue
                if asset_name not in expected:
                    # Extra files are suspicious but cannot be confused with
                    # missing downloads: absent assets may simply mean
                    # --assets was never used, so that direction is not
                    # reported.
                    self.finding(
                        SEVERITY_WARNING,
                        "release_asset_unrecorded",
                        os.path.join(asset_dir, asset_name),
                        "file is not listed in the release record's assets",
                        {"release": name},
                    )

    # ------------------------------------------------------------------
    # Clone sanity (filesystem-only; never invokes git)
    # ------------------------------------------------------------------

    def _check_clone_dir(self, unit_dir, name):
        clone_dir = os.path.join(unit_dir, name)
        if not self._is_dir(clone_dir):
            # A missing clone is normal: the resource may not have been
            # enabled, or (for gists) the source may have been inaccessible.
            return
        non_bare = self._is_dir(os.path.join(clone_dir, ".git"))
        bare = (
            self._is_file(os.path.join(clone_dir, "HEAD"))
            and self._is_dir(os.path.join(clone_dir, "objects"))
            and self._is_dir(os.path.join(clone_dir, "refs"))
        )
        if not non_bare and not bare:
            self.finding(
                SEVERITY_WARNING,
                "clone_incomplete",
                clone_dir,
                "{0} directory exists but is not a complete git clone "
                "(no .git or bare repository markers)".format(name),
            )

    # ------------------------------------------------------------------
    # Per-run archive records and the latest pointer
    # ------------------------------------------------------------------

    def _scan_run_archive(self):
        runs_dir = os.path.join(self.root, "backup-runs")
        if not self._is_dir(runs_dir):
            return

        run_dirs = []
        latest_pointer = None
        for name, is_dir, is_file in self._children(runs_dir):
            if is_dir:
                run_dirs.append(name)
            elif is_file and name == "latest.json":
                latest_pointer = os.path.join(runs_dir, name)

        finalized_any = False
        for name in run_dirs:
            record_path = os.path.join(runs_dir, name, "run.json")
            if not self._is_file(record_path):
                self.finding(
                    SEVERITY_WARNING,
                    "run_record_missing",
                    record_path,
                    "run directory exists but its run.json record was never written",
                )
                continue
            self.scanned["run_records"] += 1
            record = self._load_json(record_path)
            if isinstance(record, dict):
                if self._scan_run_record(record_path, record):
                    finalized_any = True

        if latest_pointer is not None:
            self._scan_latest_pointer(latest_pointer, runs_dir)
        elif finalized_any:
            self.finding(
                SEVERITY_INFO,
                "latest_pointer_missing",
                os.path.join(runs_dir, "latest.json"),
                "finalized run records exist but latest.json is missing",
            )

    def _scan_run_record(self, record_path, record):
        status = record.get("status")
        finalized = status in _FINALIZED_STATUSES
        if status == "running":
            self.finding(
                SEVERITY_INFO,
                "run_record_running",
                record_path,
                "run record is still marked 'running'; the run may have been "
                "interrupted",
                {"run_id": record.get("run_id")},
            )
        elif finalized and not record.get("finished_at"):
            self.finding(
                SEVERITY_WARNING,
                "run_record_unfinalized",
                record_path,
                "run record has final status {0!r} but no finished_at".format(status),
                {"status": status},
            )

        repositories = record.get("repositories")
        if repositories is None:
            repositories = []
        if not isinstance(repositories, list):
            self.finding(
                SEVERITY_WARNING,
                "run_record_malformed",
                record_path,
                "run record 'repositories' is not an array",
            )
            repositories = []

        started_at = parse_timestamp(record.get("started_at"))
        finished_at = parse_timestamp(record.get("finished_at"))

        for index, entry in enumerate(repositories):
            if not isinstance(entry, dict):
                self.finding(
                    SEVERITY_WARNING,
                    "run_record_malformed",
                    record_path,
                    "repository entry #{0} is not an object".format(index),
                    {"entry_index": index},
                )
                continue
            relative_dir = entry.get("directory")
            if not isinstance(relative_dir, str) or not relative_dir:
                continue
            unit_path = os.path.join(self.root, relative_dir)
            resources = entry.get("resources")
            if not isinstance(resources, list):
                if "resources" in entry:
                    self.finding(
                        SEVERITY_WARNING,
                        "run_record_malformed",
                        record_path,
                        "repository entry {0!r} has a non-array 'resources'".format(
                            relative_dir
                        ),
                        {"directory": relative_dir},
                    )
                continue
            for resource_entry in resources:
                if not isinstance(resource_entry, dict):
                    continue
                self._check_run_resource(
                    record_path,
                    relative_dir,
                    unit_path,
                    resource_entry,
                    started_at,
                    finished_at,
                )

        account_resources = record.get("account_resources")
        if isinstance(account_resources, list):
            account_dir = os.path.join(self.root, "account")
            for resource_entry in account_resources:
                if not isinstance(resource_entry, dict):
                    continue
                written = resource_entry.get("written", 0)
                if (
                    resource_entry.get("status") == "completed"
                    and isinstance(written, int)
                    and written > 0
                    and not os.path.isdir(account_dir)
                ):
                    self.finding(
                        SEVERITY_WARNING,
                        "run_record_missing_directory",
                        account_dir,
                        "run record reports {0} completed account resource "
                        "item(s) but the account directory is gone".format(written),
                        {
                            "run_record": self._rel(record_path),
                            "resource": resource_entry.get("name"),
                        },
                    )
        elif "account_resources" in record:
            self.finding(
                SEVERITY_WARNING,
                "run_record_malformed",
                record_path,
                "run record 'account_resources' is not an array",
            )
        return finalized

    def _check_run_resource(
        self,
        record_path,
        relative_dir,
        unit_path,
        resource_entry,
        started_at,
        finished_at,
    ):
        name = resource_entry.get("name")
        status = resource_entry.get("status")
        written = resource_entry.get("written", 0)
        if (
            status == "completed"
            and isinstance(written, int)
            and written > 0
            and not os.path.isdir(unit_path)
        ):
            self.finding(
                SEVERITY_WARNING,
                "run_record_missing_directory",
                unit_path,
                "run record reports {0} completed item(s) for {1} but its "
                "backup directory is gone".format(written, name),
                {"run_record": self._rel(record_path), "resource": name},
            )

        if status != "failed":
            return
        checkpoint_rel = {
            "issues": "issues/last_update",
            "pulls": "pulls/last_update",
            "discussions": "discussions/last_update",
        }.get(name)
        if checkpoint_rel is None or started_at is None or finished_at is None:
            return
        checkpoint_path = os.path.join(unit_path, *checkpoint_rel.split("/"))
        stat = self._stat(checkpoint_path)
        if stat is None:
            return
        checkpoint_mtime = None
        try:
            checkpoint_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        except (OSError, OverflowError, ValueError):
            return
        # The checkpoint must never advance for a resource the run recorded
        # as failed. An mtime inside the run's window is proof the on-disk
        # state contradicts the record. A small grace covers filesystem time
        # granularity; all inputs come from disk, not the current clock.
        grace_seconds = 2
        if (
            started_at.timestamp()
            <= stat.st_mtime
            <= (finished_at.timestamp() + grace_seconds)
        ):
            self.finding(
                SEVERITY_WARNING,
                "checkpoint_advanced_after_failure",
                checkpoint_path,
                "checkpoint was written during a run that recorded the {0} "
                "resource as failed".format(name),
                {
                    "run_record": self._rel(record_path),
                    "resource": name,
                    "checkpoint_mtime": checkpoint_mtime.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "run_started_at": started_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "run_finished_at": finished_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )

    def _scan_latest_pointer(self, pointer_path, runs_dir):
        pointer = self._load_json(pointer_path)
        if pointer is None:
            self.finding(
                SEVERITY_ERROR,
                "latest_pointer_invalid",
                pointer_path,
                "backup-runs/latest.json is not valid JSON",
            )
            return
        if not isinstance(pointer, dict):
            self.finding(
                SEVERITY_WARNING,
                "latest_pointer_invalid",
                pointer_path,
                "backup-runs/latest.json must be a JSON object",
                {"actual_type": type(pointer).__name__},
            )
            return
        relative_record = pointer.get("record")
        if not isinstance(relative_record, str):
            self.finding(
                SEVERITY_WARNING,
                "latest_pointer_invalid",
                pointer_path,
                "latest.json is missing its 'record' path",
            )
            return
        target = os.path.join(runs_dir, *relative_record.split("/"))
        if not self._is_file(target):
            self.finding(
                SEVERITY_ERROR,
                "latest_pointer_target_missing",
                target,
                "latest.json points at a run record that does not exist",
                {"record": relative_record},
            )
            return
        # Read without adding a second json_invalid finding; the record was
        # already (or will be) scanned individually.
        raw = self._read_bytes(target)
        record = None
        if raw is not None:
            try:
                record = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                record = None
        if isinstance(record, dict):
            pointer_run_id = pointer.get("run_id")
            target_run_id = record.get("run_id")
            if pointer_run_id and target_run_id and pointer_run_id != target_run_id:
                self.finding(
                    SEVERITY_WARNING,
                    "latest_pointer_mismatch",
                    pointer_path,
                    "latest.json run_id ({0!r}) does not match the target "
                    "record ({1!r})".format(pointer_run_id, target_run_id),
                    {
                        "pointer_run_id": pointer_run_id,
                        "record_run_id": target_run_id,
                    },
                )
            pointer_status = pointer.get("status")
            target_status = record.get("status")
            if pointer_status and target_status and pointer_status != target_status:
                self.finding(
                    SEVERITY_WARNING,
                    "latest_pointer_mismatch",
                    pointer_path,
                    "latest.json status ({0!r}) does not match the target "
                    "record ({1!r})".format(pointer_status, target_status),
                    {
                        "pointer_status": pointer_status,
                        "record_status": target_status,
                    },
                )


def scan_backup(output_directory):
    """Scan a backup root and return a deterministic, machine-readable report.

    The report shape is::

        {
          "schema_version": 1,
          "root": "/abs/path",
          "summary": {"scanned": {...}, "errors": n, "warnings": n,
                      "infos": n, "ok": bool},
          "findings": [{"severity": ..., "code": ..., "path": ...,
                        "message": ..., "details": {...}}, ...]
        }
    """
    return BackupScanner(output_directory).scan()


def filter_findings(findings, min_severity):
    rank = _SEVERITY_RANK[min_severity]
    return [item for item in findings if _SEVERITY_RANK[item["severity"]] <= rank]


def render_text(report, min_severity=SEVERITY_INFO):
    lines = []
    summary = report["summary"]
    lines.append("Backup audit for: {0}".format(report["root"]))
    lines.append(
        "scanned: {0}".format(
            ", ".join(
                "{0}={1}".format(key, value)
                for key, value in sorted(summary["scanned"].items())
            )
        )
    )
    lines.append(
        "findings: {0} error(s), {1} warning(s), {2} info item(s)".format(
            summary["errors"], summary["warnings"], summary["infos"]
        )
    )
    visible = filter_findings(report["findings"], min_severity)
    if visible:
        lines.append("")
    for item in visible:
        lines.append(
            "{0}: [{1}] {2}: {3}".format(
                item["severity"].upper(),
                item["code"],
                item["path"],
                item["message"],
            )
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m github_backup.backup_audit",
        description=(
            "Offline consistency audit of a github-backup directory. "
            "Does not contact GitHub and never modifies the backup."
        ),
    )
    parser.add_argument("output_directory", help="backup directory to audit")
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="report format (default: json)",
    )
    parser.add_argument(
        "--min-severity",
        choices=SEVERITY_ORDER,
        default=SEVERITY_INFO,
        help="minimum severity to show in the output (default: info)",
    )
    parser.add_argument(
        "--fail-on",
        choices=SEVERITY_ORDER + ("never",),
        default=SEVERITY_ERROR,
        help="exit non-zero when a finding of at least this severity exists "
        "(default: error)",
    )
    args = parser.parse_args(argv)

    report = scan_backup(args.output_directory)
    if args.format == "json":
        visible = filter_findings(report["findings"], args.min_severity)
        output_report = dict(report)
        output_report["findings"] = visible
        sys.stdout.write(json.dumps(output_report, indent=2, sort_keys=True) + "\n")
    else:
        sys.stdout.write(render_text(report, args.min_severity))

    if args.fail_on != "never":
        rank = _SEVERITY_RANK[args.fail_on]
        if any(_SEVERITY_RANK[item["severity"]] <= rank for item in report["findings"]):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
