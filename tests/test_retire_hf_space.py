from __future__ import annotations

import contextlib
import io
import types
import unittest
from unittest.mock import Mock, patch

from tools import retire_hf_space as retirement


class RepositoryNotFoundError(Exception):
    pass


class RetireHostedSpaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = Mock()
        self.owner.whoami.return_value = {"name": "config-h"}
        self.owner.space_info.side_effect = [
            types.SimpleNamespace(id=retirement.SPACE_ID, private=True),
            RepositoryNotFoundError(),
        ]
        self.owner.list_spaces.return_value = [types.SimpleNamespace(id="config-h/another-space")]
        self.public = Mock()
        self.public.model_info.return_value = types.SimpleNamespace(
            private=False, sha=retirement.MODEL_REVISION
        )
        self.factory = Mock(side_effect=[self.owner, self.public])

    def run_retirement(self) -> dict:
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = self.factory
        errors = types.ModuleType("huggingface_hub.errors")
        errors.RepositoryNotFoundError = RepositoryNotFoundError
        with patch.dict("sys.modules", {"huggingface_hub": hub, "huggingface_hub.errors": errors}), patch.dict(
            "os.environ", {"HF_TOKEN": "test-only"}
        ), contextlib.redirect_stdout(io.StringIO()):
            return retirement.retire()

    def test_deletes_only_the_named_space_and_keeps_public_model(self) -> None:
        result = self.run_retirement()
        self.owner.delete_repo.assert_called_once_with(
            repo_id=retirement.SPACE_ID, repo_type="space", missing_ok=True
        )
        self.owner.list_spaces.assert_called_once_with(author="config-h")
        self.assertEqual(self.public.model_info.call_count, 2)
        self.public.model_info.assert_called_with(
            repo_id=retirement.MODEL_ID, revision=retirement.MODEL_REVISION
        )
        self.factory.assert_any_call(token=False)
        self.assertFalse(result["space_exists"])
        self.assertTrue(result["space_deleted"])
        self.assertTrue(result["model_still_public"])
        self.assertEqual(
            {call[0] for call in self.owner.method_calls},
            {"whoami", "space_info", "delete_repo", "list_spaces"},
        )
        self.assertEqual({call[0] for call in self.public.method_calls}, {"model_info"})

    def test_already_deleted_space_is_safe_to_retry(self) -> None:
        self.owner.space_info.side_effect = RepositoryNotFoundError()
        result = self.run_retirement()
        self.owner.delete_repo.assert_called_once_with(
            repo_id=retirement.SPACE_ID, repo_type="space", missing_ok=True
        )
        self.assertFalse(result["space_exists"])

    def test_unexpected_owner_is_rejected_before_mutation(self) -> None:
        self.owner.whoami.return_value = {"name": "another-account"}
        with self.assertRaisesRegex(RuntimeError, "Space owner"):
            self.run_retirement()
        self.assertEqual(self.owner.method_calls, [("whoami", (), {})])

    def test_unexpected_repository_id_is_rejected(self) -> None:
        self.owner.space_info.side_effect = None
        self.owner.space_info.return_value = types.SimpleNamespace(id=retirement.MODEL_ID)
        with self.assertRaisesRegex(RuntimeError, "Unexpected retirement target"):
            self.run_retirement()
        self.owner.delete_repo.assert_not_called()

    def test_unavailable_public_model_prevents_deletion(self) -> None:
        self.public.model_info.return_value.private = True
        with self.assertRaisesRegex(RuntimeError, "public model release"):
            self.run_retirement()
        self.owner.delete_repo.assert_not_called()

    def test_paused_or_private_space_is_not_reported_as_deleted(self) -> None:
        self.owner.space_info.side_effect = None
        self.owner.space_info.return_value = types.SimpleNamespace(
            id=retirement.SPACE_ID, private=True
        )
        with patch.object(retirement.time, "sleep"), self.assertRaisesRegex(
            RuntimeError, "still exists"
        ):
            self.run_retirement()

    def test_space_in_owner_listing_is_not_reported_as_deleted(self) -> None:
        self.owner.list_spaces.return_value = [types.SimpleNamespace(id=retirement.SPACE_ID)]
        self.owner.space_info.side_effect = RepositoryNotFoundError()
        with patch.object(retirement.time, "sleep"), self.assertRaisesRegex(
            RuntimeError, "still exists"
        ):
            self.run_retirement()


if __name__ == "__main__":
    unittest.main()
