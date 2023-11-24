#!/bin/bash

TEMPLATE_REPO="git@gitlab.ics.muni.cz:rationai/digital-pathology/templates/python.git"
TEMPLATE_BRANCH="master"
COMMIT_FILE="last-template-commit.txt"


echo "🔄 Syncing with the template..."

# Exit on error
set -e

# Remove template remote if exists
if git remote | grep -q template; then
    git remote remove template
fi

# Add template remote
git remote add template "${TEMPLATE_REPO}"
git fetch template

# Get last synced commit hash or fallback to the first commit
if [ -s "${COMMIT_FILE}" ]
then
    COMMIT_HASH=$(cat "${COMMIT_FILE}")
else
    COMMIT_HASH=$(git rev-list --max-parents=0 "template/${TEMPLATE_BRANCH}")
fi

# Merge template/master into current branch from the last synced commit hash
LATEST_COMMIT=$(git rev-parse "template/${TEMPLATE_BRANCH}")
if [ "$COMMIT_HASH" != "$LATEST_COMMIT" ]; then
    set +e
    git cherry-pick "${COMMIT_HASH}".."template/${TEMPLATE_BRANCH}"
    RETURN_CODE=$?
    set -e

    if [ $RETURN_CODE -eq 0 ]; then
        git rev-parse template/master > "${COMMIT_FILE}"
        echo "🚀 The repository has been successfully synced with the template."
    elif [ $RETURN_CODE -eq 1 ]; then
        git rev-parse template/master > "${COMMIT_FILE}"
        echo "❗️ Auto-syncing failed. Please resolve the conflicts manually."
    else
        echo "❌ There was a problem with syncing the repository with the template."
    fi
    git cherry-pick --quit
else
    echo "🚀 The repository is already up to date with the template."
fi

# Clean up
git remote remove template
