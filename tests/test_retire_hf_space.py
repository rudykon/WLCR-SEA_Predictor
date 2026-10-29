from __future__ import annotations

import contextlib
import io
import types
import unittest
from enum import Enum
from unittest.mock import Mock, patch

from tools import retire_hf_space as retirement


class Stage(Enum):
    PAUSED = "PAUSED"
    RUNNING = "RUNNING"


class RetireHostedSpaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.owner = Mock()
        self.owner.whoami.return_value = {"name": "config-h"}
        self.owner.space_info.return_value = types.SimpleNamespace(
            id=retirement.SPACE_ID, private=False
        )
        self.owner.get_space_runtime.side_effect = [
            types.SimpleNamespace(stage=Stage.RUNNING),
            types.SimpleNamespace(stage=Stage.PAUSED),
        ]
        self.owner.space_info.side_effect = [
            types.SimpleNamespace(id=retirement.SPACE_ID, private=False),
            types.SimpleNamespace(id=retirement.SPACE_ID, private=True),
        ]
        self.public = Mock()
        self.public.model_info.return_value = types.SimpleNamespace(
            private=False, sha=retirement.MODEL_REVISION
        )
        self.factory = Mock(side_effect=[self.owner, self.public])

    def run_retirement(self) -> dict:
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = self.factory
        with patch.dict("sys.modules", {"huggingface_hub": hub}), patch.dict(
            "os.environ", {"HF_TOKEN": "test-only"}
        ), contextlib.redirect_stdout(io.StringIO()):
            return retirement.retire()

    def test_retires_only_the_named_space_and_reads_the_public_model(self) -> None:
        result = self.run_retirement()
        self.owner.pause_space.assert_called_once_with(repo_id=retirement.SPACE_ID)
        self.owner.update_repo_settings.assert_called_once_with(
            repo_id=retirement.SPACE_ID, repo_type="space", private=True
        )
        self.public.model_info.assert_called_once_with(
            repo_id=retirement.MODEL_ID, revision=retirement.MODEL_REVISION
        )
        self.factory.assert_any_call(token=False)
        self.assertEqual(result["stage"], "PAUSED")
        self.assertTrue(result["private"])
        self.assertTrue(result["model_still_public"])
        self.assertEqual(
            {call[0] for call in self.owner.method_calls},
            {"whoami", "space_info", "get_space_runtime", "pause_space", "update_repo_settings"},
        )

    def test_already_retired_space_needs_no_mutation(self) -> None:
        self.owner.space_info.side_effect = None
        self.owner.space_info.return_value.private = True
        self.owner.get_space_runtime.side_effect = None
        self.owner.get_space_runtime.return_value = types.SimpleNamespace(stage="PAUSED")
        result = self.run_retirement()
        self.owner.pause_space.assert_not_called()
        self.owner.update_repo_settings.assert_not_called()
        self.assertEqual(result["stage"], "PAUSED")

    def test_unexpected_owner_is_rejected_before_mutation(self) -> None:
        self.owner.whoami.return_value = {"name": "another-account"}
        with self.assertRaisesRegex(RuntimeError, "Space owner"):
            self.run_retirement()
        self.assertEqual(self.owner.method_calls, [("whoami", (), {})])

    def test_visibility_change_that_restarts_space_is_paused_again(self) -> None:
        self.owner.get_space_runtime.side_effect = [
            types.SimpleNamespace(stage="RUNNING"),
            types.SimpleNamespace(stage="RUNNING"),
            types.SimpleNamespace(stage="PAUSED"),
        ]
        self.owner.space_info.side_effect = [
            types.SimpleNamespace(id=retirement.SPACE_ID, private=False),
            types.SimpleNamespace(id=retirement.SPACE_ID, private=True),
            types.SimpleNamespace(id=retirement.SPACE_ID, private=True),
        ]
        with patch.object(retirement.time, "sleep"):
            result = self.run_retirement()
        self.assertEqual(self.owner.pause_space.call_count, 2)
        self.assertEqual(result["stage"], "PAUSED")


if __name__ == "__main__":
    unittest.main()
