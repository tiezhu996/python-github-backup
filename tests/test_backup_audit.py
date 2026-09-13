"""Tests for the offline backup consistency auditor.

The auditor must never contact GitHub and never modify the backup. These
tests build synthetic backup trees covering ordinary repositories, starred
repositories, gists, old-version layouts and interrupted runs, then assert
that real damage is graded correctly while normal states (resources never
enabled, empty repositories, inaccessible gists, recorded download
failures, legacy checkpoints) are not reported as corruption.
"""

import json
import os
import time
from datetime import datetime, timezone

import pytest

from github_backup.backup_audit import (
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
    filter_findings,
    intended_target_for_temp,
    main,
    parse_timestamp,
    render_text,
    scan_backup,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def root(tmp_path):
    return tmp_path


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, (dict, list)):
        path.write_text(json.dumps(content), encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return path


def codes(report, severity=None):
    findings = report["findings"]
    if severity is not None:
        findings = [f for f in findings if f["severity"] == severity]
    return [f["code"] for f in findings]


def find(report, code):
    return [f for f in report["findings"] if f["code"] == code]


def issue_payload(number, updated_at="2026-02-01T00:00:00Z", **extra):
    payload = {"number": number, "updated_at": updated_at}
    payload.update(extra)
    return payload


def attachment_manifest(number, item_type, entries):
    return {
        "item_number": number,
        "item_type": item_type,
        "repository": "owner/repo",
        "manifest_updated_at": "2026-02-02T00:00:00+00:00",
        "attachments": entries,
    }


def successful_entry(
    url="https://github.com/user-attachments/assets/a", saved_as="a", size_bytes=10
):
    return {
        "url": url,
        "success": True,
        "http_status": 200,
        "size_bytes": size_bytes,
        "saved_as": saved_as,
    }


def healthy_backup(root):
    """A backup exercising every resource type with zero real problems."""
    repo = root / "repositories" / "good"

    # ordinary clone + wiki clone
    (repo / "repository" / ".git").mkdir(parents=True)
    (repo / "wiki" / ".git").mkdir(parents=True)

    write(repo / "issues" / "1.json", issue_payload(1))
    write(repo / "issues" / "2.json", issue_payload(2, "2026-01-01T00:00:00Z"))
    write(repo / "issues" / "last_update", "2026-03-01T00:00:00Z")
    write(
        repo / "issues" / "attachments" / "1" / "manifest.json",
        attachment_manifest(1, "issue", [successful_entry()]),
    )
    (repo / "issues" / "attachments" / "1" / "a").write_bytes(b"x" * 10)
    (repo / "issues" / "attachments" / "1" / ".manifest.lock").write_text("")

    write(repo / "pulls" / "3.json", issue_payload(3))
    write(repo / "pulls" / "last_update", "2026-03-01T00:00:00Z")
    write(repo / "pulls" / "reviews_last_update", "2026-03-01T00:00:00Z")

    write(
        repo / "discussions" / "4.json",
        {"number": 4, "updatedAt": "2026-02-01T00:00:00Z"},
    )
    write(repo / "discussions" / "last_update", "2026-03-01T00:00:00Z")

    write(repo / "milestones" / "1.json", {"number": 1, "title": "m1"})
    write(
        repo / "security-advisories" / "GHSA-aaaa-bbbb-cccc.json",
        {"ghsa_id": "GHSA-aaaa-bbbb-cccc", "summary": "s"},
    )
    write(repo / "labels" / "labels.json", [{"name": "bug"}])
    write(repo / "hooks" / "hooks.json", [])
    write(
        repo / "releases" / "v1.0.json",
        {
            "tag_name": "v1.0",
            "assets": [{"name": "bin.zip", "size": 10, "url": "u"}],
        },
    )
    asset_dir = repo / "releases" / "v1.0"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "bin.zip").write_bytes(b"x" * 10)

    # starred repository, same layout
    starred = root / "starred" / "owner" / "other"
    write(starred / "issues" / "9.json", issue_payload(9))
    (starred / "repository" / "HEAD").parent.mkdir(parents=True)
    (starred / "repository" / "HEAD").write_text("ref: refs/heads/main\n")
    (starred / "repository" / "objects").mkdir()
    (starred / "repository" / "refs").mkdir()

    # gist: metadata + clone
    write(
        root / "gists" / "gist1" / "gist.json",
        {"id": "gist1", "updated_at": "2026-02-01T00:00:00Z"},
    )
    (root / "gists" / "gist1" / "repository" / ".git").mkdir(parents=True)

    # account dumps
    write(root / "account" / "starred.json", [])
    write(root / "account" / "followers.json", [])
    return root


# ---------------------------------------------------------------------------
# Healthy backups produce no errors/warnings
# ---------------------------------------------------------------------------


class TestHealthyBackups:

    def test_full_backup_is_clean(self, root):
        healthy_backup(root)
        report = scan_backup(str(root))
        assert report["summary"]["errors"] == 0, report["findings"]
        assert report["summary"]["warnings"] == 0, report["findings"]

    def test_minimal_repo_with_only_issues_is_clean(self, root):
        repo = root / "repositories" / "mini"
        write(repo / "issues" / "1.json", issue_payload(1))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []

    def test_disabled_or_empty_resources_are_not_damage(self, root):
        # Empty issues/pulls dirs (disabled features, empty repos), no
        # checkpoint, no JSON files.
        repo = root / "repositories" / "empty"
        (repo / "issues").mkdir(parents=True)
        (repo / "pulls").mkdir()
        (repo / "milestones").mkdir()
        (repo / "releases").mkdir()
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []

    def test_inaccessible_gist_without_clone_is_not_damage(self, root):
        write(
            root / "gists" / "blocked" / "gist.json",
            {"id": "blocked", "updated_at": "2026-02-01T00:00:00Z"},
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []

    def test_recorded_attachment_failures_are_info_only(self, root):
        repo = root / "repositories" / "r"
        write(repo / "issues" / "1.json", issue_payload(1))
        write(
            repo / "issues" / "attachments" / "1" / "manifest.json",
            attachment_manifest(
                1,
                "issue",
                [
                    {"url": "https://x/404", "success": False, "http_status": 404},
                    {"url": "https://x/503", "success": False, "http_status": 503},
                ],
            ),
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []
        assert codes(report, SEVERITY_INFO).count("attachment_recorded_failure") == 2

    def test_legacy_global_checkpoint_is_info_only(self, root):
        write(root / "last_update", "2026-01-01T00:00:00Z")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []
        assert codes(report, SEVERITY_INFO) == ["legacy_checkpoint_present"]

    def test_no_backup_runs_dir_is_clean(self, root):
        healthy_backup(root)
        assert not (root / "backup-runs").exists()
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []

    def test_locks_are_not_flagged(self, root):
        repo = root / "repositories" / "r"
        write(repo / "issues" / "1.json", issue_payload(1))
        (repo / "issues" / "attachments").mkdir(parents=True)
        (repo / "issues" / "attachments" / ".x.lock").write_text("")
        write(
            repo / "releases" / "v1.json",
            {"tag_name": "v1", "assets": [{"name": "a.zip"}]},
        )
        adir = repo / "releases" / "v1"
        adir.mkdir(parents=True)
        (adir / "a.zip.lock").write_text("")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == []
        assert codes(report, SEVERITY_ERROR) == []


# ---------------------------------------------------------------------------
# Truncated / invalid JSON
# ---------------------------------------------------------------------------


class TestInvalidJson:

    def test_truncated_item_json_is_error_and_scan_continues(self, root):
        repo = root / "repositories" / "r"
        write(repo / "issues" / "1.json", '{"number": 1, "comment_data": [')
        write(repo / "issues" / "2.json", issue_payload(2))
        report = scan_backup(str(root))
        errors = find(report, "json_invalid")
        assert len(errors) == 1
        assert errors[0]["path"].endswith("issues/1.json")
        assert errors[0]["details"]["reason"] == "parse_error"
        # the sibling file was still inspected
        assert report["summary"]["scanned"]["json_files"] >= 2

    def test_empty_json_file_is_error(self, root):
        write(root / "repositories" / "r" / "issues" / "1.json", "")
        report = scan_backup(str(root))
        error = find(report, "json_invalid")[0]
        assert error["details"]["reason"] == "empty_file"

    def test_corrupt_manifest_and_run_record_are_errors(self, root):
        repo = root / "repositories" / "r"
        write(repo / "issues" / "attachments" / "1" / "manifest.json", "{bad")
        write(root / "backup-runs" / "20260101T000000Z-p1-x" / "run.json", "[")
        report = scan_backup(str(root))
        paths = " ".join(f["path"] for f in find(report, "json_invalid"))
        assert "manifest.json" in paths
        assert "run.json" in paths

    def test_unreadable_file_does_not_stop_scan(self, root, monkeypatch):
        repo = root / "repositories" / "r"
        bad = write(repo / "issues" / "1.json", issue_payload(1))
        write(repo / "issues" / "2.json", issue_payload(2))

        real_open = open

        def fake_open(path, *args, **kwargs):
            if str(path) == str(bad):
                raise PermissionError("denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fake_open)
        report = scan_backup(str(root))
        scan_errors = find(report, "scan_error")
        assert scan_errors, "permission error must be recorded as a finding"
        assert scan_errors[0]["path"].endswith("issues/1.json")
        # the sibling file was still read and the scan completed normally
        assert report["summary"]["scanned"]["json_files"] >= 1


# ---------------------------------------------------------------------------
# Missing metadata / identity mismatches
# ---------------------------------------------------------------------------


class TestMissingMetadata:

    def test_gist_directory_without_gist_json(self, root):
        (root / "gists" / "ghost").mkdir(parents=True)
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["gist_metadata_missing"]

    def test_gist_json_missing_id(self, root):
        write(root / "gists" / "g" / "gist.json", {"updated_at": "x"})
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["record_missing_field"]

    def test_item_filename_number_mismatch(self, root):
        write(root / "repositories" / "r" / "issues" / "1.json", issue_payload(2))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["record_identity_mismatch"]

    @pytest.mark.parametrize(
        "resource,filename,payload,expected_code",
        [
            ("milestones", "5.json", {"title": "m"}, "record_missing_field"),
            (
                "security-advisories",
                "GHSA-x.json",
                {"summary": "s"},
                "record_missing_field",
            ),
            ("releases", "v1.json", {"name": "r"}, "record_missing_field"),
        ],
    )
    def test_records_missing_identity_fields(
        self, root, resource, filename, payload, expected_code
    ):
        write(root / "repositories" / "r" / resource / filename, payload)
        report = scan_backup(str(root))
        assert expected_code in codes(report, SEVERITY_WARNING)

    def test_labels_dir_without_labels_json(self, root):
        (root / "repositories" / "r" / "labels").mkdir(parents=True)
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["resource_index_missing"]

    def test_account_dir_empty(self, root):
        (root / "account").mkdir()
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["account_dir_empty"]

    def test_account_dump_wrong_type(self, root):
        write(root / "account" / "followers.json", {"not": "a list"})
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["unexpected_json_type"]


# ---------------------------------------------------------------------------
# Attachments vs manifest
# ---------------------------------------------------------------------------


class TestAttachmentConsistency:

    def _issue(self, root, number=1):
        return write(
            root / "repositories" / "r" / "issues" / f"{number}.json",
            issue_payload(number),
        )

    def test_successful_manifest_entry_without_file_is_error(self, root):
        self._issue(root)
        write(
            root
            / "repositories"
            / "r"
            / "issues"
            / "attachments"
            / "1"
            / "manifest.json",
            attachment_manifest(1, "issue", [successful_entry(saved_as="a")]),
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == ["attachment_file_missing"]
        assert codes(report, SEVERITY_WARNING) == []

    def test_unrecorded_file_on_disk_is_warning(self, root):
        self._issue(root)
        write(
            root
            / "repositories"
            / "r"
            / "issues"
            / "attachments"
            / "1"
            / "manifest.json",
            attachment_manifest(1, "issue", []),
        )
        adir = root / "repositories" / "r" / "issues" / "attachments" / "1"
        (adir / "rogue.png").write_bytes(b"x")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == ["attachment_unrecorded"]

    def test_attachment_dir_without_manifest_is_warning(self, root):
        self._issue(root)
        adir = root / "repositories" / "r" / "issues" / "attachments" / "1"
        adir.mkdir(parents=True)
        (adir / "x").write_bytes(b"x")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["attachment_manifest_missing"]

    def test_orphan_attachment_dir(self, root):
        # attachments/2 exists but issues/2.json does not
        write(root / "repositories" / "r" / "issues" / "1.json", issue_payload(1))
        write(
            root
            / "repositories"
            / "r"
            / "issues"
            / "attachments"
            / "2"
            / "manifest.json",
            attachment_manifest(2, "issue", []),
        )
        report = scan_backup(str(root))
        assert "attachment_orphan_dir" in codes(report, SEVERITY_WARNING)

    def test_size_mismatch_is_warning(self, root):
        self._issue(root)
        write(
            root
            / "repositories"
            / "r"
            / "issues"
            / "attachments"
            / "1"
            / "manifest.json",
            attachment_manifest(1, "issue", [successful_entry(size_bytes=99)]),
        )
        adir = root / "repositories" / "r" / "issues" / "attachments" / "1"
        (adir / "a").write_bytes(b"x" * 10)
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["attachment_size_mismatch"]
        assert codes(report, SEVERITY_ERROR) == []

    def test_manifest_item_number_and_type_mismatch(self, root):
        self._issue(root)
        write(
            root
            / "repositories"
            / "r"
            / "issues"
            / "attachments"
            / "1"
            / "manifest.json",
            attachment_manifest(2, "discussion", []),
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == [
            "manifest_item_mismatch",
            "manifest_item_mismatch",
        ]

    def test_duplicate_url_and_saved_as(self, root):
        self._issue(root)
        write(
            root
            / "repositories"
            / "r"
            / "issues"
            / "attachments"
            / "1"
            / "manifest.json",
            attachment_manifest(
                1,
                "issue",
                [
                    successful_entry(url="u", saved_as="a"),
                    successful_entry(url="u", saved_as="a"),
                ],
            ),
        )
        adir = root / "repositories" / "r" / "issues" / "attachments" / "1"
        (adir / "a").write_bytes(b"x" * 10)
        report = scan_backup(str(root))
        warnings = codes(report, SEVERITY_WARNING)
        assert "manifest_duplicate_url" in warnings
        assert "manifest_duplicate_file" in warnings

    def test_pull_and_discussion_attachment_layouts(self, root):
        write(
            root / "repositories" / "r" / "pulls" / "7.json",
            {"number": 7, "updated_at": "2026-02-01T00:00:00Z"},
        )
        write(
            root
            / "repositories"
            / "r"
            / "pulls"
            / "attachments"
            / "7"
            / "manifest.json",
            attachment_manifest(7, "pull", [successful_entry()]),
        )
        adir = root / "repositories" / "r" / "pulls" / "attachments" / "7"
        (adir / "a").write_bytes(b"x" * 10)

        write(
            root / "repositories" / "r" / "discussions" / "8.json",
            {"number": 8, "updatedAt": "2026-02-01T00:00:00Z"},
        )
        write(
            root
            / "repositories"
            / "r"
            / "discussions"
            / "attachments"
            / "8"
            / "manifest.json",
            attachment_manifest(8, "discussion", [successful_entry()]),
        )
        adir = root / "repositories" / "r" / "discussions" / "attachments" / "8"
        (adir / "a").write_bytes(b"x" * 10)

        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []


# ---------------------------------------------------------------------------
# Release assets
# ---------------------------------------------------------------------------


class TestReleaseAssets:

    def test_asset_dir_without_release_json_is_warning(self, root):
        adir = root / "repositories" / "r" / "releases" / "v2"
        adir.mkdir(parents=True)
        (adir / "a.zip").write_bytes(b"x")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["release_assets_without_record"]

    def test_unlisted_asset_file_is_warning(self, root):
        write(
            root / "repositories" / "r" / "releases" / "v1.json",
            {"tag_name": "v1", "assets": []},
        )
        adir = root / "repositories" / "r" / "releases" / "v1"
        adir.mkdir(parents=True)
        (adir / "extra.zip").write_bytes(b"x")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["release_asset_unrecorded"]

    def test_missing_assets_are_not_damage(self, root):
        # --assets may never have been enabled; the record lists assets but
        # nothing was downloaded and no asset directory exists.
        write(
            root / "repositories" / "r" / "releases" / "v1.json",
            {"tag_name": "v1", "assets": [{"name": "a.zip"}]},
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []


# ---------------------------------------------------------------------------
# Checkpoint contradictions
# ---------------------------------------------------------------------------


class TestCheckpoints:

    def test_malformed_checkpoint_is_warning(self, root):
        write(root / "repositories" / "r" / "issues" / "last_update", "nope")
        write(root / "repositories" / "r" / "issues" / "1.json", issue_payload(1))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["checkpoint_invalid"]

    def test_checkpoint_older_than_items_is_warning(self, root):
        idir = root / "repositories" / "r" / "issues"
        write(idir / "last_update", "2026-01-01T00:00:00Z")
        write(idir / "1.json", issue_payload(1, "2026-05-01T00:00:00Z"))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["checkpoint_older_than_items"]

    def test_checkpoint_newer_than_items_is_fine(self, root):
        idir = root / "repositories" / "r" / "issues"
        write(idir / "last_update", "2026-05-01T00:00:00Z")
        write(idir / "1.json", issue_payload(1, "2026-01-01T00:00:00Z"))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == []

    def test_checkpoint_with_no_items_is_fine(self, root):
        # Empty repository: checkpoint written, zero JSON files.
        write(
            root / "repositories" / "r" / "issues" / "last_update",
            "2026-05-01T00:00:00Z",
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == []


# ---------------------------------------------------------------------------
# Leftover temp files and incomplete clones
# ---------------------------------------------------------------------------


class TestTempAndClones:

    @pytest.mark.parametrize(
        "temp_name,expected_final",
        [
            ("1.json.temp", "1.json"),
            ("manifest.json.temp", "manifest.json"),
            ("1.json.1234-abcdef01.temp", "1.json"),
            (
                "run.json.999-deadbeef.temp",
                "run.json",
            ),
            ("abc.temp", "abc"),
        ],
    )
    def test_temp_target_derivation(self, temp_name, expected_final):
        target = intended_target_for_temp("/x/" + temp_name)
        assert os.path.basename(target) == expected_final

    def test_leftover_temp_files_are_warnings(self, root):
        write(root / "repositories" / "r" / "issues" / "1.json.temp", "{}")
        write(
            root / "backup-runs" / "run.json.4242-01234567.temp",
            "{}",
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == [
            "temp_file_left",
            "temp_file_left",
        ]
        findings = find(report, "temp_file_left")
        intended = {f["details"]["intended_target"] for f in findings}
        assert "repositories/r/issues/1.json" in intended
        assert "backup-runs/run.json" in intended

    def test_temp_files_inside_clones_are_user_data(self, root):
        repo = root / "repositories" / "r"
        (repo / "repository" / ".git").mkdir(parents=True)
        (repo / "repository" / "notes.temp").write_text("user content")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == []

    def test_incomplete_clone_directory_is_warning(self, root):
        repo = root / "repositories" / "r"
        (repo / "repository").mkdir(parents=True)
        (repo / "repository" / "partial-object").write_text("x")
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["clone_incomplete"]

    def test_missing_clone_is_not_damage(self, root):
        write(root / "repositories" / "r" / "issues" / "1.json", issue_payload(1))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == []


# ---------------------------------------------------------------------------
# Run archive records and latest pointer
# ---------------------------------------------------------------------------


def _run_record(runs_dir, run_id, status, started, finished, repositories=None):
    run_dir = runs_dir / run_id
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "started_at": started,
        "updated_at": finished or started,
        "finished_at": finished,
        "repositories": repositories or [],
        "account_resources": [],
        "errors": [],
        "summary": {},
    }
    write(run_dir / "run.json", record)
    return record


class TestRunArchive:

    def test_finalized_record_is_clean(self, root):
        runs = root / "backup-runs"
        _run_record(
            runs,
            "20260101T000000Z-p1-aaaaaa",
            "completed",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
            repositories=[
                {
                    "name": "owner/r",
                    "kind": "repository",
                    "directory": "repositories/r",
                    "status": "completed",
                    "resources": [
                        {"name": "issues", "status": "completed", "written": 0}
                    ],
                }
            ],
        )
        (root / "repositories" / "r" / "issues").mkdir(parents=True)
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == []
        assert codes(report, SEVERITY_ERROR) == []
        # No latest pointer yet; that is only ever informational.
        assert codes(report, SEVERITY_INFO) == ["latest_pointer_missing"]

    def test_running_record_is_info_not_damage(self, root):
        _run_record(
            root / "backup-runs",
            "20260101T000000Z-p1-aaaaaa",
            "running",
            "2026-01-01T00:00:00Z",
            None,
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == []
        assert codes(report, SEVERITY_INFO) == ["run_record_running"]

    def test_finalized_without_finished_at_is_warning(self, root):
        _run_record(
            root / "backup-runs",
            "20260101T000000Z-p1-aaaaaa",
            "completed",
            "2026-01-01T00:00:00Z",
            None,
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["run_record_unfinalized"]

    def test_run_directory_without_run_json(self, root):
        (root / "backup-runs" / "20260101T000000Z-p1-aaaaaa").mkdir(parents=True)
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["run_record_missing"]

    def test_completed_writes_but_directory_gone(self, root):
        _run_record(
            root / "backup-runs",
            "20260101T000000Z-p1-aaaaaa",
            "completed",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
            repositories=[
                {
                    "name": "owner/gone",
                    "directory": "repositories/gone",
                    "status": "completed",
                    "resources": [
                        {"name": "issues", "status": "completed", "written": 3}
                    ],
                }
            ],
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["run_record_missing_directory"]

    def test_checkpoint_advanced_after_failed_resource(self, root):
        started = "2026-01-01T00:00:00Z"
        finished = "2026-01-01T00:05:00Z"
        _run_record(
            root / "backup-runs",
            "20260101T000000Z-p1-aaaaaa",
            "completed_with_errors",
            started,
            finished,
            repositories=[
                {
                    "name": "owner/r",
                    "directory": "repositories/r",
                    "status": "completed_with_errors",
                    "resources": [{"name": "issues", "status": "failed", "written": 0}],
                }
            ],
        )
        checkpoint = write(
            root / "repositories" / "r" / "issues" / "last_update",
            "2026-01-01T00:02:00Z",
        )
        middle = datetime(2026, 1, 1, 0, 2, 30, tzinfo=timezone.utc).timestamp()
        os.utime(checkpoint, (middle, middle))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["checkpoint_advanced_after_failure"]

    def test_old_checkpoint_surviving_failed_run_is_not_flagged(self, root):
        # Checkpoint predates the failed run window: it is from an earlier
        # successful round and must not be reported.
        started = "2026-01-01T00:00:00Z"
        finished = "2026-01-01T00:05:00Z"
        _run_record(
            root / "backup-runs",
            "20260101T000000Z-p1-aaaaaa",
            "completed_with_errors",
            started,
            finished,
            repositories=[
                {
                    "name": "owner/r",
                    "directory": "repositories/r",
                    "status": "completed_with_errors",
                    "resources": [{"name": "issues", "status": "failed", "written": 0}],
                }
            ],
        )
        checkpoint = write(
            root / "repositories" / "r" / "issues" / "last_update",
            "2025-01-01T00:00:00Z",
        )
        old = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
        os.utime(checkpoint, (old, old))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == []

    def test_latest_pointer_dangling_is_error(self, root):
        write(
            root / "backup-runs" / "latest.json",
            {"run_id": "x", "status": "completed", "record": "x/run.json"},
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == ["latest_pointer_target_missing"]

    def test_latest_pointer_mismatch_is_warning(self, root):
        runs = root / "backup-runs"
        _run_record(
            runs,
            "20260101T000000Z-p1-aaaaaa",
            "completed",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
        )
        write(
            runs / "latest.json",
            {
                "run_id": "different",
                "status": "failed",
                "record": "20260101T000000Z-p1-aaaaaa/run.json",
            },
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_ERROR) == []
        assert codes(report, SEVERITY_WARNING) == [
            "latest_pointer_mismatch",
            "latest_pointer_mismatch",
        ]

    def test_completed_account_resource_without_account_dir(self, root):
        _run_record(
            root / "backup-runs",
            "20260101T000000Z-p1-aaaaaa",
            "completed",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
        )
        record_path = root / "backup-runs" / "20260101T000000Z-p1-aaaaaa" / "run.json"
        record = json.loads(record_path.read_text())
        record["account_resources"] = [
            {"name": "followers", "status": "completed", "written": 1}
        ]
        record_path.write_text(json.dumps(record))
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_WARNING) == ["run_record_missing_directory"]

    def test_missing_latest_with_finalized_records_is_info(self, root):
        _run_record(
            root / "backup-runs",
            "20260101T000000Z-p1-aaaaaa",
            "completed",
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:05:00Z",
        )
        report = scan_backup(str(root))
        assert codes(report, SEVERITY_INFO) == ["latest_pointer_missing"]


# ---------------------------------------------------------------------------
# Determinism, report shape and resilience
# ---------------------------------------------------------------------------


class TestReportProperties:

    def test_repeated_scans_are_identical(self, root):
        healthy_backup(root)
        write(root / "repositories" / "good" / "issues" / "9.json.temp", "x")
        first = scan_backup(str(root))
        time.sleep(0.01)
        second = scan_backup(str(root))
        assert first == second

    def test_findings_are_sorted_and_machine_readable(self, root):
        write(root / "repositories" / "b" / "issues" / "1.json", "{")
        write(root / "repositories" / "a" / "issues" / "1.json", "")
        report = scan_backup(str(root))
        paths = [f["path"] for f in report["findings"]]
        assert paths == sorted(paths)
        for finding in report["findings"]:
            assert set(finding) >= {
                "severity",
                "code",
                "path",
                "message",
            }
            assert finding["severity"] in (
                SEVERITY_ERROR,
                SEVERITY_WARNING,
                SEVERITY_INFO,
            )
        # JSON-serializable
        json.dumps(report)

    def test_report_summary_counts(self, root):
        healthy_backup(root)
        report = scan_backup(str(root))
        scanned = report["summary"]["scanned"]
        assert scanned["repository_units"] == 1
        assert scanned["starred_units"] == 1
        assert scanned["gist_units"] == 1
        assert scanned["attachment_manifests"] == 1
        assert report["summary"]["ok"] is True

    def test_unreadable_directory_is_recorded_not_fatal(self, root, monkeypatch):
        healthy_backup(root)

        real_scandir = os.scandir

        def fake_scandir(path, *args, **kwargs):
            if str(path) == str(root / "repositories" / "good" / "issues"):
                raise PermissionError("denied")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", fake_scandir)
        report = scan_backup(str(root))
        assert find(report, "scan_error"), report["findings"]
        # gists and starred were still scanned
        assert report["summary"]["scanned"]["gist_units"] == 1
        assert report["summary"]["scanned"]["starred_units"] == 1

    def test_nonexistent_root(self, tmp_path):
        report = scan_backup(str(tmp_path / "missing"))
        assert codes(report, SEVERITY_ERROR) == ["backup_root_invalid"]

    def test_filter_findings(self):
        findings = [
            {"severity": SEVERITY_ERROR},
            {"severity": SEVERITY_WARNING},
            {"severity": SEVERITY_INFO},
        ]
        assert len(filter_findings(findings, SEVERITY_WARNING)) == 2
        assert len(filter_findings(findings, SEVERITY_ERROR)) == 1

    def test_render_text_contains_findings(self, root):
        write(root / "repositories" / "r" / "issues" / "1.json", "")
        report = scan_backup(str(root))
        text = render_text(report)
        assert "ERROR" in text
        assert "json_invalid" in text

    def test_parse_timestamp_variants(self):
        assert parse_timestamp("2026-01-01T00:00:00Z") is not None
        assert parse_timestamp("2026-01-01T00:00:00.123Z") is not None
        assert parse_timestamp("2026-01-01T00:00:00+00:00") is not None
        assert parse_timestamp("not a timestamp") is None
        assert parse_timestamp("") is None
        assert parse_timestamp(None) is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:

    def test_clean_backup_exits_zero_text(self, root, capsys):
        healthy_backup(root)
        exit_code = main([str(root), "--format", "text"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "0 error(s), 0 warning(s)" in out

    def test_dirty_backup_exits_one_json(self, root, capsys):
        write(root / "repositories" / "r" / "issues" / "1.json", "{")
        exit_code = main([str(root)])
        assert exit_code == 1
        out = capsys.readouterr().out
        report = json.loads(out)
        assert report["summary"]["errors"] == 1

    def test_fail_on_never_exits_zero_with_errors(self, root, capsys):
        write(root / "repositories" / "r" / "issues" / "1.json", "{")
        exit_code = main([str(root), "--fail-on", "never"])
        assert exit_code == 0

    def test_fail_on_warning_for_temp_file(self, root):
        write(root / "repositories" / "r" / "issues" / "1.json.temp", "{}")
        assert main([str(root), "--format", "text"]) == 0
        assert main([str(root), "--fail-on", "warning", "--format", "text"]) == 1

    def test_min_severity_filters_output(self, root, capsys):
        write(root / "last_update", "2026-01-01T00:00:00Z")
        exit_code = main([str(root), "--min-severity", "error"])
        assert exit_code == 0
        report = json.loads(capsys.readouterr().out)
        assert report["findings"] == []
