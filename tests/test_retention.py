"""Tests for the offline retention tidy-up pass."""

import json
import os
from pathlib import Path

import pytest

from github_backup import retention
from github_backup.github_backup import parse_args
from github_backup.retention import (
    BackupLockedError,
    InvalidPlanError,
    PlanMismatchError,
    RetentionError,
    apply_retention_plan,
    backup_lock,
    build_retention_plan,
    load_approved_plan,
    run_retention_cli,
    write_plan_file,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def write(path, contents=b""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(contents, (bytes, bytearray)):
        path.write_bytes(bytes(contents))
    else:
        path.write_text(contents)
    return path


def write_manifest(directory, saved=(), failed=()):
    """Write an attachments manifest.json listing saved and failed entries."""
    attachments = [
        {"success": True, "saved_as": name, "url": "https://example/{0}".format(name)}
        for name in saved
    ]
    attachments.extend(
        {
            "success": False,
            "saved_as": None,
            "url": "https://example/failed-{0}".format(name),
            "http_status": 404,
        }
        for name in failed
    )
    return write(
        Path(directory) / "manifest.json", json.dumps({"attachments": attachments})
    )


def removable_paths(plan):
    return {entry["path"]: entry for entry in plan["entries"]}


def kept_categories(plan):
    return plan["kept_summary"]


@pytest.fixture
def backup_tree(tmp_path):
    """A realistic tree mixing live data, checkpoints and garbage."""
    root = tmp_path

    # Live issue data + checkpoint + referenced/orphan attachments
    issue_dir = root / "repositories" / "repo-a" / "issues"
    write(issue_dir / "1.json", "{}")
    write(issue_dir / "last_update", "2026-01-01T00:00:00Z")
    write(issue_dir / "1.json.temp", b"partial")
    attach = issue_dir / "attachments" / "1"
    write(attach / "kept.png", b"x" * 100)
    write(attach / "orphan.png", b"y" * 50)
    write_manifest(attach, saved=["kept.png"])

    # Pull data with reviews checkpoint and an attachment dir with no manifest
    pulls_dir = root / "repositories" / "repo-a" / "pulls"
    write(pulls_dir / "2.json", "{}")
    write(pulls_dir / "reviews_last_update", "2026-02-01T00:00:00Z")
    unknown_attach = pulls_dir / "attachments" / "9"
    write(unknown_attach / "mystery.bin", b"z")

    # Release assets (no manifest exists for these) must be retained
    assets_dir = root / "repositories" / "repo-a" / "releases" / "v1"
    write(assets_dir / "binary.tar.gz", b"asset")
    write(root / "repositories" / "repo-a" / "releases" / "v1.json", "{}")

    # A git clone and wiki clone must be retained wholesale
    write(root / "repositories" / "repo-a" / "repository" / ".git" / "HEAD", "ref")
    write(root / "repositories" / "repo-a" / "wiki" / ".git" / "HEAD", "ref")

    # Starred repository with the same layout
    star_attach = (
        root
        / "starred"
        / "owner"
        / "starred-repo"
        / "discussions"
        / "attachments"
        / "7"
    )
    write(star_attach / "pic.jpg", b"p")
    write(star_attach / "leftover.jpg.temp", b"t")
    write_manifest(star_attach, saved=["pic.jpg"])

    # Gists
    write(root / "gists" / "gistid" / "gist.json", "{}")
    write(root / "gists" / "gistid" / "repository" / ".git" / "HEAD", "ref")

    # Account-level data
    write(root / "account" / "starred.json", "[]")

    # Old/unknown resources, retained by policy
    write(root / "repositories" / "repo-a" / "old-version-dir" / "data", b"old")
    write(root / "unknown-toplevel.txt", b"?")
    write(root / "legacy-backup" / "something", b"?")

    # Empty attachments container from an interrupted backup
    empty_item = root / "repositories" / "repo-b" / "issues" / "attachments" / "42"
    empty_item.mkdir(parents=True)

    return root


# ---------------------------------------------------------------------------
# Scanning / classification
# ---------------------------------------------------------------------------


def test_plan_flags_temp_files_and_orphan_attachments(backup_tree):
    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)

    assert "repositories/repo-a/issues/1.json.temp" in victims
    assert victims["repositories/repo-a/issues/1.json.temp"]["category"] == "temp_file"

    orphan = victims["repositories/repo-a/issues/attachments/1/orphan.png"]
    assert orphan["category"] == "orphan_attachment"
    assert orphan["size_bytes"] == 50

    assert (
        "starred/owner/starred-repo/discussions/attachments/7/leftover.jpg.temp"
        in victims
    )


def test_plan_keeps_referenced_attachments_and_manifest(backup_tree):
    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)

    assert "repositories/repo-a/issues/attachments/1/kept.png" not in victims
    assert "repositories/repo-a/issues/attachments/1/manifest.json" not in victims
    assert "starred/owner/starred-repo/discussions/attachments/7/pic.jpg" not in victims
    assert kept_categories(plan)["referenced_attachment"]["count"] >= 4


def test_attachments_without_readable_manifest_are_kept(backup_tree):
    # No manifest exists for pulls/attachments/9: nothing there is removable.
    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)
    assert "repositories/repo-a/pulls/attachments/9/mystery.bin" not in victims
    assert kept_categories(plan)["unverified_attachment"]["count"] == 1

    # A corrupt manifest likewise forces retention.
    manifest = (
        backup_tree
        / "repositories"
        / "repo-a"
        / "issues"
        / "attachments"
        / "1"
        / "manifest.json"
    )
    manifest.write_text("{not json")
    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)
    assert "repositories/repo-a/issues/attachments/1/orphan.png" not in victims
    assert kept_categories(plan)["unverified_attachment"]["count"] >= 2


def test_checkpoints_clones_and_release_assets_are_kept(backup_tree):
    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)

    for protected in (
        "repositories/repo-a/issues/last_update",
        "repositories/repo-a/pulls/reviews_last_update",
        "repositories/repo-a/issues/1.json",
        "repositories/repo-a/pulls/2.json",
        "repositories/repo-a/releases/v1/binary.tar.gz",
        "repositories/repo-a/repository/.git/HEAD",
        "repositories/repo-a/wiki/.git/HEAD",
        "gists/gistid/gist.json",
        "gists/gistid/repository/.git/HEAD",
        "account/starred.json",
    ):
        assert protected not in victims, protected

    assert kept_categories(plan)["checkpoint"]["count"] == 2
    assert "backup_data" in kept_categories(plan)


def test_unrecognized_and_old_version_directories_are_kept(backup_tree):
    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)

    assert "unknown-toplevel.txt" not in victims
    assert "legacy-backup/something" not in victims
    assert "repositories/repo-a/old-version-dir/data" not in victims

    sample = "\n".join(plan["unrecognized_sample"])
    assert "unknown-toplevel.txt" in sample
    assert "repositories/repo-a/old-version-dir/" in sample
    assert kept_categories(plan)["unrecognized"]["count"] >= 3


def test_empty_attachments_containers_are_removable_but_other_dirs_are_not(backup_tree):
    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)

    empty_item = "repositories/repo-b/issues/attachments/42/"
    empty_root = "repositories/repo-b/issues/attachments/"
    assert victims[empty_item]["category"] == "empty_directory"
    assert victims[empty_root]["category"] == "empty_directory"

    # An attachments dir that still holds a manifest is not empty.
    assert "repositories/repo-a/issues/attachments/" not in victims


def test_orphans_removed_but_manifest_keeps_its_item_directory(backup_tree):
    # An item dir whose manifest lists nothing that is on disk: orphan files
    # are removed, but manifest.json itself is retained data, so the item
    # directory stays.
    item = backup_tree / "repositories" / "repo-c" / "pulls" / "attachments" / "3"
    write(item / "junk.png", b"j")
    write(item / "junk.png.temp", b"t")
    write_manifest(item, saved=[])

    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)
    assert "repositories/repo-c/pulls/attachments/3/junk.png" in victims
    assert victims["repositories/repo-c/pulls/attachments/3/junk.png"]["category"] == (
        "orphan_attachment"
    )
    assert "repositories/repo-c/pulls/attachments/3/junk.png.temp" in victims
    assert "repositories/repo-c/pulls/attachments/3/manifest.json" not in victims
    assert "repositories/repo-c/pulls/attachments/3/" not in victims


def test_temp_only_attachments_dir_without_manifest_is_removed_entirely(backup_tree):
    # Crash leftover (.temp), no manifest ever written: the whole empty
    # container goes once the leftover is removed.
    item = backup_tree / "repositories" / "repo-c" / "pulls" / "attachments" / "3"
    write(item / "junk.png.temp", b"t")

    plan = build_retention_plan(backup_tree)
    victims = removable_paths(plan)
    assert "repositories/repo-c/pulls/attachments/3/junk.png.temp" in victims
    assert victims["repositories/repo-c/pulls/attachments/3/"]["category"] == (
        "empty_directory"
    )
    assert victims["repositories/repo-c/pulls/attachments/"]["category"] == (
        "empty_directory"
    )


def test_plan_is_deterministic_and_size_accounted(backup_tree):
    first = build_retention_plan(backup_tree)
    second = build_retention_plan(backup_tree)
    assert first["plan_id"] == second["plan_id"]

    paths = [entry["path"] for entry in first["entries"]]
    # Files alphabetically first, then directories deepest-first.
    assert paths == sorted(
        paths,
        key=lambda p: (
            0 if not p.endswith("/") else 1,
            0 if not p.endswith("/") else -p.count("/"),
            p,
        ),
    )
    assert all(
        not p.endswith("/")
        for p in paths[: sum(1 for p in paths if not p.endswith("/"))]
    )

    expected_bytes = sum(
        entry["size_bytes"]
        for entry in first["entries"]
        if entry["category"] != "empty_directory"
    )
    assert first["total_remove_bytes"] == expected_bytes
    assert first["total_remove_bytes"] == 50 + len(b"partial") + len(b"t")


def test_symlinks_are_never_removed(tmp_path):
    attach = tmp_path / "repositories" / "r" / "issues" / "attachments" / "1"
    attach.mkdir(parents=True)
    target = tmp_path / "outside.png"
    write(target, b"outside")
    os.symlink(target, attach / "link.png")
    write_manifest(attach, saved=[])

    plan = build_retention_plan(tmp_path)
    assert "repositories/r/issues/attachments/1/link.png" not in removable_paths(plan)
    # The symlink also protects its parent directory from removal.
    assert "repositories/r/issues/attachments/1/" not in removable_paths(plan)


def test_empty_backup_directory_produces_empty_plan(tmp_path):
    plan = build_retention_plan(tmp_path)
    assert plan["entries"] == []
    assert plan["total_remove_bytes"] == 0


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------


def test_lock_is_mutually_exclusive(tmp_path):
    with backup_lock(str(tmp_path), "backup"):
        with pytest.raises(BackupLockedError):
            with backup_lock(str(tmp_path), "tidy"):
                pass  # pragma: no cover

    # Lock is released even when the holder raises.
    with pytest.raises(RuntimeError):
        with backup_lock(str(tmp_path), "backup"):
            raise RuntimeError("boom")

    with backup_lock(str(tmp_path), "tidy"):
        assert os.path.exists(tmp_path / retention.LOCK_FILENAME)


def test_apply_refuses_while_backup_holds_lock(backup_tree):
    plan = build_retention_plan(backup_tree)
    with backup_lock(str(backup_tree), "backup"):
        with pytest.raises(BackupLockedError):
            apply_retention_plan(str(backup_tree), plan, assume_yes=True)
    # Nothing was touched.
    assert (backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp").exists()


# ---------------------------------------------------------------------------
# Plan files / approval
# ---------------------------------------------------------------------------


def test_plan_file_roundtrip_and_tamper_detection(backup_tree):
    plan = build_retention_plan(backup_tree)
    path = write_plan_file(backup_tree, plan)

    loaded = load_approved_plan(path, backup_tree)
    assert loaded["plan_id"] == plan["plan_id"]

    # Tamper with an entry after approval.
    with open(path) as handle:
        tampered = json.load(handle)
    tampered["entries"][0]["size_bytes"] = 999999
    with open(path, "w") as handle:
        json.dump(tampered, handle)
    with pytest.raises(InvalidPlanError):
        load_approved_plan(path, backup_tree)


def test_plan_rejected_for_different_root(backup_tree, tmp_path):
    other = tmp_path / "other-backup"
    other.mkdir()
    plan = build_retention_plan(backup_tree)
    path = write_plan_file(backup_tree, plan)
    with pytest.raises(InvalidPlanError):
        load_approved_plan(path, other)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_apply_removes_only_planned_entries_and_writes_report(backup_tree):
    plan = build_retention_plan(backup_tree)
    report = apply_retention_plan(str(backup_tree), plan, assume_yes=True)

    assert not (
        backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp"
    ).exists()
    assert not (
        backup_tree
        / "repositories"
        / "repo-a"
        / "issues"
        / "attachments"
        / "1"
        / "orphan.png"
    ).exists()
    # Only attachments containers are removed; the empty issues/ resource dir
    # itself is retained.
    assert not (
        backup_tree / "repositories" / "repo-b" / "issues" / "attachments"
    ).exists()
    assert (backup_tree / "repositories" / "repo-b" / "issues").is_dir()

    # Referenced data survives.
    assert (
        backup_tree
        / "repositories"
        / "repo-a"
        / "issues"
        / "attachments"
        / "1"
        / "kept.png"
    ).exists()
    assert (
        backup_tree / "repositories" / "repo-a" / "releases" / "v1" / "binary.tar.gz"
    ).exists()
    assert (backup_tree / "repositories" / "repo-a" / "issues" / "last_update").exists()
    assert (backup_tree / "legacy-backup" / "something").exists()

    assert report["removed_count"] == plan["total_remove_count"]
    assert report["freed_bytes"] == plan["total_remove_bytes"]

    # Journal and quarantine are gone; report is kept for review.
    metadata = backup_tree / retention.METADATA_DIRNAME
    assert not list(metadata.glob("journal-*.json"))
    assert not list(metadata.glob("quarantine-*"))
    assert list(metadata.glob("report-*.json"))

    # Re-scanning after applying yields an empty plan (idempotent).
    assert build_retention_plan(backup_tree)["entries"] == []


def test_apply_requires_confirmation_unless_yes(backup_tree):
    plan = build_retention_plan(backup_tree)
    victim = backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp"

    answers = iter(["n"])
    with pytest.raises(RetentionError):
        apply_retention_plan(
            str(backup_tree), plan, prompt_func=lambda _: next(answers)
        )
    assert victim.exists()

    answers = iter(["y"])
    report = apply_retention_plan(
        str(backup_tree), plan, prompt_func=lambda _: next(answers)
    )
    assert report["removed_count"] >= 1
    assert not victim.exists()


def test_apply_aborts_when_directory_changed_after_approval(backup_tree):
    plan = build_retention_plan(backup_tree)

    # A new garbage file appears between --tidy and --tidy-apply: the stored
    # plan no longer describes the tree.
    write(backup_tree / "repositories" / "repo-a" / "issues" / "new.temp", b"new")

    with pytest.raises(PlanMismatchError):
        apply_retention_plan(str(backup_tree), plan, assume_yes=True)

    # Nothing from the old plan was deleted.
    assert (backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp").exists()
    assert (backup_tree / "repositories" / "repo-a" / "issues" / "new.temp").exists()


def test_interrupted_apply_is_resumed_from_journal(backup_tree, monkeypatch):
    plan = build_retention_plan(backup_tree)
    real_replace = os.replace
    victim_marker = os.path.join("issues", "1.json.temp")
    calls = {"n": 0}

    def crashing_replace(source, destination):
        real_replace(source, destination)
        # Only fail the staging move of the victim - never journal/report
        # atomic writes inside the metadata directory.
        if str(source).endswith(
            victim_marker
        ) and retention.METADATA_DIRNAME not in str(source):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated crash after staging")

    monkeypatch.setattr(retention.os, "replace", crashing_replace)
    with pytest.raises(RuntimeError):
        apply_retention_plan(str(backup_tree), plan, assume_yes=True)
    monkeypatch.undo()

    quarantine = (
        backup_tree
        / retention.METADATA_DIRNAME
        / "quarantine-{0}".format(plan["plan_id"][:12])
    )
    # Bytes were staged, not unlinked, and the journal survived.
    assert quarantine.exists()
    journals = list((backup_tree / retention.METADATA_DIRNAME).glob("journal-*.json"))
    assert len(journals) == 1

    # Resume with the same (already approved) plan - no re-confirmation.
    report = apply_retention_plan(str(backup_tree), plan, assume_yes=True)
    assert report["removed_count"] == plan["total_remove_count"]
    assert report["recovered_after_interruption"] is True
    assert not quarantine.exists()
    assert not (
        backup_tree
        / "repositories"
        / "repo-a"
        / "issues"
        / "attachments"
        / "1"
        / "orphan.png"
    ).exists()
    assert (
        backup_tree
        / "repositories"
        / "repo-a"
        / "issues"
        / "attachments"
        / "1"
        / "kept.png"
    ).exists()

    # A further run is a stable no-op.
    assert build_retention_plan(backup_tree)["entries"] == []


def test_resume_completes_an_interrupted_other_plan_first(backup_tree):
    # A journal left in "committed" state by a previous run for a *different*
    # (already purged) plan must be finalized before the new plan runs.
    stale_plan_id = "0" * 64
    os.makedirs(retention.metadata_dir(str(backup_tree)), exist_ok=True)
    journal = {
        "format_version": retention.RETENTION_FORMAT_VERSION,
        "plan_id": stale_plan_id,
        "root": str(backup_tree),
        "state": "committed",
        "quarantine": os.path.relpath(
            retention._quarantine_dir(str(backup_tree), stale_plan_id),
            str(backup_tree),
        ),
        "entries": [],
    }
    with open(retention._journal_path(str(backup_tree), stale_plan_id), "w") as fh:
        json.dump(journal, fh)

    new_plan = build_retention_plan(backup_tree)
    report = apply_retention_plan(str(backup_tree), new_plan, assume_yes=True)
    assert report["removed_count"] == new_plan["total_remove_count"]
    assert not list((backup_tree / retention.METADATA_DIRNAME).glob("journal-*.json"))
    # The new plan really was applied afterwards.
    assert not (
        backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp"
    ).exists()


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def test_parse_args_tidy_modes(tmp_path):
    args = parse_args(["--tidy", "-o", str(tmp_path)])
    assert args.tidy is True
    assert args.tidy_apply is None
    assert args.user is None

    args = parse_args(["--tidy-apply", "/tmp/plan.json", "-o", str(tmp_path), "--yes"])
    assert args.tidy_apply == "/tmp/plan.json"
    assert args.assume_yes is True

    with pytest.raises(SystemExit):
        parse_args(["-o", str(tmp_path)])  # USER required without tidy mode
    with pytest.raises(SystemExit):
        parse_args(["user", "--tidy", "--tidy-apply", "/tmp/plan.json"])


def test_cli_preview_writes_plan_and_deletes_nothing(backup_tree):
    args = parse_args(["--tidy", "-o", str(backup_tree)])
    run_retention_cli(args)

    plans = list((backup_tree / retention.METADATA_DIRNAME).glob("plan-*.json"))
    assert len(plans) == 1
    assert (backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp").exists()

    # The plan file itself lives in the ignored metadata dir and a rescan
    # still finds the same garbage set.
    plan = build_retention_plan(backup_tree)
    assert plan["total_remove_count"] >= 1


def test_cli_apply_executes_plan_file(backup_tree):
    preview_args = parse_args(["--tidy", "-o", str(backup_tree)])
    run_retention_cli(preview_args)
    plan_path = next((backup_tree / retention.METADATA_DIRNAME).glob("plan-*.json"))

    apply_args = parse_args(
        ["--tidy-apply", str(plan_path), "-o", str(backup_tree), "--yes"]
    )
    run_retention_cli(apply_args)

    assert not (
        backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp"
    ).exists()
    assert list((backup_tree / retention.METADATA_DIRNAME).glob("report-*.json"))


def test_cli_apply_rejects_tampered_plan(backup_tree):
    preview_args = parse_args(["--tidy", "-o", str(backup_tree)])
    run_retention_cli(preview_args)
    plan_path = next((backup_tree / retention.METADATA_DIRNAME).glob("plan-*.json"))

    with open(plan_path) as handle:
        payload = json.load(handle)
    payload["entries"] = []
    with open(plan_path, "w") as handle:
        json.dump(payload, handle)

    apply_args = parse_args(
        ["--tidy-apply", str(plan_path), "-o", str(backup_tree), "--yes"]
    )
    with pytest.raises(SystemExit):
        run_retention_cli(apply_args)
    assert (backup_tree / "repositories" / "repo-a" / "issues" / "1.json.temp").exists()


def test_cli_apply_missing_directory_exits(tmp_path):
    missing = tmp_path / "does-not-exist"
    args = parse_args(["--tidy", "-o", str(missing)])
    with pytest.raises(SystemExit):
        run_retention_cli(args)
