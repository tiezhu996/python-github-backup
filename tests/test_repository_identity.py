"""Tests for stable backup identity across renames/transfers and for the
preview/apply migration workflow (github_backup.identity)."""

import json
import os

import pytest

from github_backup import github_backup
from github_backup.identity import (
    DUPLICATES_DIRNAME,
    KIND_REPOSITORY,
    KIND_STARRED,
    MARKER_FILENAME,
    ROLE_HOME,
    ROLE_SHADOW,
    BackupLocationResolver,
    apply_claims,
    apply_plan,
    canonical_item_path,
    load_marker,
    merge_directory,
    plan_migration,
    read_clone_owner_name,
    render_plan,
    save_marker,
    update_repository_marker,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _repo(repo_id=1, name="repo", owner="alice", **extra):
    data = {
        "id": repo_id,
        "node_id": "R_{0}".format(repo_id),
        "name": name,
        "full_name": "{0}/{1}".format(owner, name),
        "owner": {"login": owner},
        "clone_url": "https://github.com/{0}/{1}.git".format(owner, name),
        "private": False,
        "fork": False,
        "has_wiki": False,
        "updated_at": "2026-09-01T00:00:00Z",
    }
    data.update(extra)
    return data


def _starred(repo_id=1, name="repo", owner="bob", **extra):
    data = _repo(repo_id, name, owner, is_starred=True)
    data.update(extra)
    return data


def _gist(gist_id="g1", owner="alice"):
    return {
        "id": gist_id,
        "is_gist": True,
        "owner": {"login": owner},
        "updated_at": "2026-09-01T00:00:00Z",
        "git_pull_url": "https://gist.github.com/{0}.git".format(gist_id),
    }


def _write_entry_marker(root, relpath, kind, repo_id, full_name=None, role=ROLE_HOME):
    path = root / relpath
    path.mkdir(parents=True, exist_ok=True)
    owner, name = (full_name or "alice/repo").split("/")
    save_marker(
        str(path),
        {
            "version": 1,
            "kind": kind,
            "id": repo_id,
            "owner": owner,
            "name": name,
            "full_name": full_name or "alice/repo",
            "role": role,
            "merged_into": None,
            "first_seen_at": "2026-01-01T00:00:00+00:00",
        },
    )
    return path


def _write_clone(root, relpath, url):
    git_dir = root / relpath / "repository" / ".git"
    git_dir.mkdir(parents=True, exist_ok=True)
    (git_dir / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n"
        '[remote "origin"]\n\turl = {0}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n'.format(
            url
        )
    )


def _write_bare_clone(root, relpath, url):
    repo_dir = root / relpath / "repository"
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "config").write_text('[remote "origin"]\n\turl = {0}\n'.format(url))


# ---------------------------------------------------------------------------
# Canonical naming
# ---------------------------------------------------------------------------


class TestCanonicalPaths:
    def test_each_kind_has_its_own_naming_rule(self, tmp_path):
        assert canonical_item_path(str(tmp_path), _repo(1, "x")) == str(
            tmp_path / "repositories" / "x"
        )
        assert canonical_item_path(str(tmp_path), _starred(1, "x", "bob")) == str(
            tmp_path / "starred" / "bob" / "x"
        )
        assert canonical_item_path(str(tmp_path), _gist("g1")) == str(
            tmp_path / "gists" / "g1"
        )

    def test_gist_path_ignores_owner_changes(self, tmp_path):
        gist = _gist("g1", owner="alice")
        path_before = canonical_item_path(str(tmp_path), gist)
        gist["owner"] = {"login": "org-after-transfer"}
        assert canonical_item_path(str(tmp_path), gist) == path_before


# ---------------------------------------------------------------------------
# Backup-time resolution
# ---------------------------------------------------------------------------


class TestBackupReuse:
    def test_renamed_repository_keeps_original_directory(self, tmp_path):
        old = _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        resolver = BackupLocationResolver(str(tmp_path))

        path, reason = resolver.resolve(_repo(1, "new-name", "alice"))

        assert os.path.dirname(path.rstrip(os.sep)) == str(tmp_path / "repositories")
        assert path == str(old)

    def test_transferred_starred_repository_keeps_original_directory(self, tmp_path):
        old = _write_entry_marker(
            tmp_path,
            "starred/old-owner/thing",
            KIND_STARRED,
            7,
            "old-owner/thing",
        )
        resolver = BackupLocationResolver(str(tmp_path))

        path, _ = resolver.resolve(_starred(7, "thing", "new-owner"))

        assert path == str(old)

    def test_reused_directory_marker_is_refreshed_with_new_full_name(self, tmp_path):
        old = _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        resolver = BackupLocationResolver(str(tmp_path))
        path, _ = resolver.resolve(_repo(1, "new-name", "alice"))

        save = load_marker(path)
        update_repository_marker(path, _repo(1, "new-name", "alice"), KIND_REPOSITORY)
        marker = load_marker(path)
        assert marker["id"] == 1
        assert marker["full_name"] == "alice/new-name"
        assert marker["role"] == ROLE_HOME
        # Identity history is preserved.
        assert marker["first_seen_at"] == save["first_seen_at"]
        assert old.exists()

    def test_new_repository_uses_canonical_directory(self, tmp_path):
        resolver = BackupLocationResolver(str(tmp_path))
        path, reason = resolver.resolve(_repo(9, "fresh"))
        assert path == str(tmp_path / "repositories" / "fresh")
        assert reason == "new-directory"

    def test_same_name_different_source_is_isolated_not_merged(self, tmp_path):
        """The core safety guarantee: repositories/tools belongs to id 1;
        an unrelated new repository also named 'tools' (id 2) must never be
        written into it."""
        tools_old = _write_entry_marker(
            tmp_path, "repositories/tools", KIND_REPOSITORY, 1, "alice/tools"
        )
        (tools_old / "issues").mkdir()
        (tools_old / "issues" / "1.json").write_text('{"number": 1}')

        resolver = BackupLocationResolver(str(tmp_path))
        path, reason = resolver.resolve(_repo(2, "tools", "alice"))

        assert path == str(tmp_path / "repositories" / "tools.2")
        assert reason == "isolated-conflict"
        assert load_marker(str(tools_old))["id"] == 1
        assert (tools_old / "issues" / "1.json").exists()

        # The isolated directory claims its own identity when used.
        os.makedirs(path)
        update_repository_marker(path, _repo(2, "tools", "alice"), KIND_REPOSITORY)
        resolver2 = BackupLocationResolver(str(tmp_path))
        again, _ = resolver2.resolve(_repo(2, "tools", "alice"))
        assert again == path

    def test_starred_and_repository_with_same_id_stay_separate_trees(self, tmp_path):
        _write_entry_marker(
            tmp_path, "repositories/thing", KIND_REPOSITORY, 5, "alice/thing"
        )
        resolver = BackupLocationResolver(str(tmp_path))
        path, _ = resolver.resolve(_starred(5, "thing", "alice"))
        assert path == str(tmp_path / "starred" / "alice" / "thing")

    def test_marker_refusing_foreign_overwrite(self, tmp_path):
        path = _write_entry_marker(
            tmp_path, "repositories/tools", KIND_REPOSITORY, 1, "alice/tools"
        )
        update_repository_marker(str(path), _repo(2, "tools", "alice"), KIND_REPOSITORY)
        assert load_marker(str(path))["id"] == 1


class TestLegacyAdoption:
    def test_markerless_dir_with_matching_clone_remote_is_claimed(self, tmp_path):
        _write_clone(tmp_path, "repositories/repo", "https://github.com/alice/repo.git")
        resolver = BackupLocationResolver(str(tmp_path))

        path, reason = resolver.resolve(_repo(1, "repo", "alice"))

        assert path == str(tmp_path / "repositories" / "repo")
        assert reason == "reuse-claimed"
        assert load_marker(path)["id"] == 1

    def test_markerless_dir_with_bare_clone_and_credentials(self, tmp_path):
        _write_bare_clone(
            tmp_path,
            "repositories/repo",
            "https://x-oauth-basic:secret@github.com/alice/repo.git",
        )
        resolver = BackupLocationResolver(str(tmp_path))
        path, _ = resolver.resolve(_repo(1, "repo", "alice"))
        assert path == str(tmp_path / "repositories" / "repo")
        assert read_clone_owner_name(str(tmp_path / "repositories" / "repo")) == (
            "alice",
            "repo",
        )

    def test_markerless_dir_with_foreign_remote_is_isolated_without_api(self, tmp_path):
        # The directory once held old-org/tools (id 1); that repository moved
        # away while a new, unrelated alice/tools (id 2) now owns the name.
        _write_clone(
            tmp_path, "repositories/tools", "https://github.com/old-org/tools.git"
        )
        resolver = BackupLocationResolver(str(tmp_path))
        path, reason = resolver.resolve(_repo(2, "tools", "alice"))
        assert path == str(tmp_path / "repositories" / "tools.2")
        assert reason == "isolated-conflict"
        assert not (tmp_path / "repositories" / "tools" / MARKER_FILENAME).exists()

    def test_markerless_dir_verified_same_via_api_is_claimed(self, tmp_path):
        legacy = tmp_path / "repositories" / "repo"
        legacy.mkdir(parents=True)
        calls = []

        def verifier(kind, entry, repository):
            calls.append(entry.relpath)
            return True

        resolver = BackupLocationResolver(str(tmp_path), verifier=verifier)
        path, reason = resolver.resolve(_repo(1, "repo", "alice"))
        assert path == str(legacy)
        assert reason == "reuse-claimed"
        assert calls == ["repositories/repo"]

    def test_markerless_dir_verified_different_via_api_is_isolated(self, tmp_path):
        (tmp_path / "repositories" / "tools").mkdir(parents=True)
        resolver = BackupLocationResolver(str(tmp_path), verifier=lambda *a: False)
        path, reason = resolver.resolve(_repo(2, "tools", "alice"))
        assert path == str(tmp_path / "repositories" / "tools.2")
        assert reason == "isolated-conflict"

    def test_markerless_dir_unverifiable_falls_back_to_name_path(self, tmp_path):
        (tmp_path / "repositories" / "repo").mkdir(parents=True)
        resolver = BackupLocationResolver(str(tmp_path), verifier=lambda *a: None)
        path, reason = resolver.resolve(_repo(1, "repo", "alice"))
        assert path == str(tmp_path / "repositories" / "repo")
        assert reason == "reuse-claimed"


class TestDuplicateHomes:
    def test_second_home_is_demoted_to_shadow(self, tmp_path):
        old = _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        partial = _write_entry_marker(
            tmp_path, "repositories/new-name", KIND_REPOSITORY, 1, "alice/new-name"
        )
        # Old directory holds more history; it must remain the home.
        (old / "issues").mkdir(exist_ok=True)
        (old / "issues" / "1.json").write_text("{}")

        resolver = BackupLocationResolver(str(tmp_path))
        path, _ = resolver.resolve(_repo(1, "new-name", "alice"))

        assert path == str(old)
        assert load_marker(str(old))["role"] == ROLE_HOME
        shadow = load_marker(str(partial))
        assert shadow["role"] == ROLE_SHADOW
        assert shadow["merged_into"] == "repositories/old-name"

    def test_backup_never_writes_into_shadow_after_restart(self, tmp_path):
        old = _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        _write_entry_marker(
            tmp_path,
            "repositories/new-name",
            KIND_REPOSITORY,
            1,
            "alice/new-name",
            role=ROLE_SHADOW,
        )
        # Post-demotion restart: exactly one home is returned.
        resolver = BackupLocationResolver(str(tmp_path))
        path, _ = resolver.resolve(_repo(1, "new-name", "alice"))
        assert path == str(old)


# ---------------------------------------------------------------------------
# End-to-end backup_repositories integration
# ---------------------------------------------------------------------------


class TestBackupRepositoriesIdentity:
    def test_incremental_checkpoint_survives_rename(
        self, create_args, tmp_path, monkeypatch
    ):
        old = _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        issues = old / "issues"
        issues.mkdir()
        (issues / "last_update").write_text("2026-01-01T00:00:00Z")

        seen = []

        def fake_backup_issues(passed_args, repo_cwd, repository, repos_template):
            seen.append((repo_cwd, passed_args.since))

        monkeypatch.setattr(github_backup, "backup_issues", fake_backup_issues)
        args = create_args(incremental=True, include_issues=True)

        github_backup.backup_repositories(
            args, str(tmp_path), [_repo(1, "new-name", "alice")]
        )

        assert seen == [(str(old), "2026-01-01T00:00:00Z")]
        # New canonical directory was never created.
        assert not (tmp_path / "repositories" / "new-name").exists()
        assert (issues / "last_update").read_text() == "2026-09-01T00:00:00Z"

    def test_no_second_copy_for_transferred_starred(
        self, create_args, tmp_path, monkeypatch
    ):
        old = _write_entry_marker(
            tmp_path,
            "starred/old-owner/thing",
            KIND_STARRED,
            7,
            "old-owner/thing",
        )
        seen = []

        def fake_backup_labels(passed_args, repo_cwd, repository, repos_template):
            seen.append(repo_cwd)

        monkeypatch.setattr(github_backup, "backup_labels", fake_backup_labels)
        args = create_args(include_labels=True)

        github_backup.backup_repositories(
            args, str(tmp_path), [_starred(7, "thing", "new-owner")]
        )

        assert seen == [str(old)]
        assert not (tmp_path / "starred" / "new-owner").exists()

    def test_gist_continues_in_id_directory(self, create_args, tmp_path, monkeypatch):
        import unittest.mock as mock

        with mock.patch("github_backup.github_backup.fetch_repository") as fetch:
            args = create_args(include_gists=True)
            github_backup.backup_repositories(args, str(tmp_path), [_gist("g1")])
            assert fetch.called
        assert (tmp_path / "gists" / "g1" / "gist.json").is_file()
        assert not (tmp_path / "gists" / "g1" / MARKER_FILENAME).exists()


# ---------------------------------------------------------------------------
# Migration: planning
# ---------------------------------------------------------------------------


class TestMigrationPlanning:
    def test_legacy_rename_is_planned_as_claim_and_rename(self, tmp_path):
        _write_clone(
            tmp_path, "repositories/old-name", "https://github.com/alice/old-name.git"
        )

        def verify(kind, owner, name):
            assert (owner, name) == ("alice", "old-name")
            return _repo(1, "new-name", "alice")

        plan = plan_migration(
            str(tmp_path), [_repo(1, "new-name", "alice")], verify, "alice"
        )
        assert [(c.entry.relpath, c.role) for c in plan.claims] == [
            ("repositories/old-name", ROLE_HOME)
        ]
        assert len(plan.merges) == 0
        assert len(plan.renames) == 1
        assert plan.renames[0].dst == str(tmp_path / "repositories" / "new-name")

    def test_split_directories_are_planned_as_shadow_merge(self, tmp_path):
        _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        _write_entry_marker(
            tmp_path, "repositories/new-name", KIND_REPOSITORY, 1, "alice/new-name"
        )
        plan = plan_migration(str(tmp_path), [_repo(1, "new-name", "alice")])
        roles = {(c.entry.relpath, c.role) for c in plan.claims}
        assert ("repositories/old-name", ROLE_SHADOW) in roles
        assert plan.merges[0].shadow.relpath == "repositories/old-name"
        assert plan.merges[0].home.relpath == "repositories/new-name"
        assert plan.renames == []

    def test_foreign_directory_blocks_rename_with_conflict(self, tmp_path):
        # id 1 currently lives at 'new-name' but its only on-disk home is
        # 'old-name'; canonical path is owned by unrelated id 2.
        _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        _write_entry_marker(
            tmp_path, "repositories/new-name", KIND_REPOSITORY, 2, "alice/new-name"
        )
        plan = plan_migration(
            str(tmp_path),
            [_repo(1, "new-name", "alice"), _repo(2, "new-name", "alice")],
        )
        assert plan.renames == []
        assert any("new-name" in c for c in plan.conflicts)
        assert not any(
            c.entry.relpath == "repositories/new-name" and c.role == ROLE_SHADOW
            for c in plan.claims
        )

    def test_unverifiable_legacy_directory_is_left_untouched(self, tmp_path):
        (tmp_path / "repositories" / "mystery").mkdir(parents=True)
        plan = plan_migration(
            str(tmp_path),
            [_repo(1, "mystery", "alice")],
            verify=lambda *a: None,
            backup_user="alice",
        )
        assert plan.claims == []
        assert plan.merges == []
        assert plan.renames == []
        assert any("mystery" in n for n in plan.notes)

    def test_gists_never_generate_operations(self, tmp_path):
        gist_dir = tmp_path / "gists" / "g1"
        gist_dir.mkdir(parents=True)
        (gist_dir / "gist.json").write_text(json.dumps({"id": "g1"}))
        plan = plan_migration(
            str(tmp_path), [_gist("g1")], verify=None, backup_user="alice"
        )
        assert plan.changes == 0
        assert plan.gists_seen == 1

    def test_preview_text_is_explicit_about_not_changing_anything(self, tmp_path):
        _write_clone(
            tmp_path, "repositories/old-name", "https://github.com/alice/old-name.git"
        )
        plan = plan_migration(
            str(tmp_path),
            [_repo(1, "new-name", "alice")],
            verify=lambda *a: _repo(1, "new-name", "alice"),
            backup_user="alice",
        )
        text = render_plan(plan, str(tmp_path), False)
        assert "PREVIEW" in text
        assert "--migrate-apply" in text
        assert "repositories/old-name -> repositories/new-name" in text


# ---------------------------------------------------------------------------
# Migration: execution
# ---------------------------------------------------------------------------


class TestMigrationApply:
    def test_rename_executes_and_is_idempotent(self, tmp_path):
        old = tmp_path / "repositories" / "old-name"
        old.mkdir(parents=True)
        (old / "labels.json").write_text("[]")

        def verify(kind, owner, name):
            return _repo(1, "new-name", "alice")

        plan = plan_migration(
            str(tmp_path), [_repo(1, "new-name", "alice")], verify, "alice"
        )
        apply_plan(plan, str(tmp_path))

        new = tmp_path / "repositories" / "new-name"
        assert new.is_dir()
        assert not old.exists()
        assert (new / "labels.json").is_file()
        assert load_marker(str(new))["id"] == 1

        # Re-planning must be a no-op.
        plan_again = plan_migration(
            str(tmp_path), [_repo(1, "new-name", "alice")], verify, "alice"
        )
        assert plan_again.changes == 0

    def test_starred_transfer_renames_and_cleans_empty_owner_dir(self, tmp_path):
        _write_entry_marker(
            tmp_path,
            "starred/old-owner/thing",
            KIND_STARRED,
            7,
            "old-owner/thing",
        )
        plan = plan_migration(
            str(tmp_path), [_starred(7, "thing", "new-owner")], backup_user="alice"
        )
        apply_plan(plan, str(tmp_path))
        assert (tmp_path / "starred" / "new-owner" / "thing").is_dir()
        assert not (tmp_path / "starred" / "old-owner").exists()

    def test_merge_preserves_history_checkpoints_and_attachments(self, tmp_path):
        old = tmp_path / "repositories" / "old-name"
        new = tmp_path / "repositories" / "new-name"
        _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        _write_entry_marker(
            tmp_path, "repositories/new-name", KIND_REPOSITORY, 1, "alice/new-name"
        )

        # Old history + old incremental boundary.
        (old / "issues").mkdir()
        (old / "issues" / "1.json").write_text('{"number": 1}')
        (old / "issues" / "last_update").write_text("2025-01-01T00:00:00Z")
        attachments = old / "issues" / "attachments" / "1"
        attachments.mkdir(parents=True)
        (attachments / "diagram.png").write_bytes(b"png-bytes")
        manifest = {
            "item_number": 1,
            "attachments": [
                {
                    "url": "https://x/diagram.png",
                    "saved_as": "diagram.png",
                    "success": True,
                }
            ],
        }
        (attachments / "manifest.json").write_text(json.dumps(manifest))
        # New partial history with a newer boundary; overlap must be re-fetched,
        # so the merged boundary has to be the OLDER timestamp.
        (new / "issues").mkdir()
        (new / "issues" / "2.json").write_text('{"number": 2}')
        (new / "issues" / "last_update").write_text("2026-08-01T00:00:00Z")
        (new / "pulls").mkdir()
        (new / "pulls" / "reviews_last_update").write_text("2026-08-02T00:00:00Z")

        plan = plan_migration(str(tmp_path), [_repo(1, "new-name", "alice")])
        apply_plan(plan, str(tmp_path))

        assert not old.exists()
        assert (new / "issues" / "1.json").is_file()
        assert (new / "issues" / "2.json").is_file()
        assert (
            new / "issues" / "attachments" / "1" / "diagram.png"
        ).read_bytes() == b"png-bytes"
        moved_manifest = json.loads(
            (new / "issues" / "attachments" / "1" / "manifest.json").read_text()
        )
        assert moved_manifest["attachments"][0]["saved_as"] == "diagram.png"
        assert (new / "issues" / "last_update").read_text() == "2025-01-01T00:00:00Z"
        assert (
            new / "pulls" / "reviews_last_update"
        ).read_text() == "2026-08-02T00:00:00Z"

        # The next incremental backup resumes from the older boundary.
        resolver = BackupLocationResolver(str(tmp_path))
        path, _ = resolver.resolve(_repo(1, "new-name", "alice"))
        assert path == str(new)

    def test_conflicting_data_files_are_quarantined_not_deleted(self, tmp_path):
        shadow = tmp_path / "repositories" / "old-name"
        home = tmp_path / "repositories" / "new-name"
        shadow.mkdir(parents=True)
        home.mkdir(parents=True)
        (home / "1.json").write_text("new version")
        (shadow / "1.json").write_text("old unique content")
        os.utime(shadow / "1.json", (1, 1))
        os.utime(home / "1.json", (2_000_000_000, 2_000_000_000))

        merge_directory(str(shadow), str(home))

        assert (home / "1.json").read_text() == "new version"
        quarantined = list((home / DUPLICATES_DIRNAME).rglob("1.json"))
        assert len(quarantined) == 1
        assert quarantined[0].read_text() == "old unique content"
        assert not shadow.exists()

    def test_interrupted_migration_never_has_two_current_copies(self, tmp_path):
        old = tmp_path / "repositories" / "old-name"
        new = tmp_path / "repositories" / "new-name"
        _write_entry_marker(
            tmp_path, "repositories/old-name", KIND_REPOSITORY, 1, "alice/old-name"
        )
        _write_entry_marker(
            tmp_path, "repositories/new-name", KIND_REPOSITORY, 1, "alice/new-name"
        )
        (old / "issues").mkdir()
        (old / "issues" / "1.json").write_text("{}")
        (old / "issues" / "last_update").write_text("2025-01-01T00:00:00Z")
        (new / "issues").mkdir()
        (new / "issues" / "2.json").write_text("{}")
        (new / "issues" / "last_update").write_text("2026-08-01T00:00:00Z")

        plan = plan_migration(str(tmp_path), [_repo(1, "new-name", "alice")])

        # Simulate a crash after the claim phase, before any data moves.
        apply_claims(plan)

        assert load_marker(str(old))["role"] == ROLE_SHADOW
        assert load_marker(str(new))["role"] == ROLE_HOME
        # A backup run started at this moment sees exactly one current copy.
        resolver = BackupLocationResolver(str(tmp_path))
        path, _ = resolver.resolve(_repo(1, "new-name", "alice"))
        assert path == str(new)
        assert (old / "issues" / "1.json").is_file()  # shadow data untouched

        # Re-run (resume) completes the consolidation.
        resume_plan = plan_migration(str(tmp_path), [_repo(1, "new-name", "alice")])
        apply_plan(resume_plan, str(tmp_path))
        assert not old.exists()
        assert (new / "issues" / "1.json").is_file()
        assert (new / "issues" / "2.json").is_file()
        assert (new / "issues" / "last_update").read_text() == "2025-01-01T00:00:00Z"

    def test_merge_is_safe_to_resume_halfway(self, tmp_path):
        shadow = tmp_path / "repositories" / "old-name"
        home = tmp_path / "repositories" / "new-name"
        shadow.mkdir(parents=True)
        home.mkdir(parents=True)
        (shadow / "a.json").write_text("a")
        (shadow / "sub").mkdir()
        (shadow / "sub" / "b.json").write_text("b")

        # First half: move one file then "crash" by merging again from the
        # partially emptied shadow tree.
        os.makedirs(home / "sub", exist_ok=True)
        os.replace(str(shadow / "a.json"), str(home / "a.json"))
        merge_directory(str(shadow), str(home))

        assert (home / "a.json").is_file()
        assert (home / "sub" / "b.json").is_file()
        assert not shadow.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
