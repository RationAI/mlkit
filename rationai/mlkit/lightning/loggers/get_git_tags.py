import logging
import os
from typing import Any

import git
from mlflow.utils.mlflow_tags import (
    MLFLOW_GIT_BRANCH,
    MLFLOW_GIT_COMMIT,
    MLFLOW_GIT_REPO_URL,
)


log = logging.getLogger(__name__)


def get_git_tags() -> dict[str, Any]:
    repo = git.Repo(path=os.getenv("ORIG_WORKING_DIR", os.getcwd()))
    if repo.head.is_detached:
        log.warning("Cannot get git branch ('detached HEAD' state)")

    return {
        MLFLOW_GIT_COMMIT: repo.head.commit.hexsha,
        MLFLOW_GIT_REPO_URL: repo.remotes.origin.url,  # not in the UI
        "git.repo_url": repo.remotes.origin.url,
        MLFLOW_GIT_BRANCH: repo.active_branch.name,  # not in the UI
        "git.branch": repo.active_branch.name,
    }
