#!/bin/bash
if [[ $GITHUB_TOKEN ]]; then
  use $GITHUB_TOKEN
fi
git config --global url."https://x-access-token:$GITHUB_TOKEN@github.com/end-to-end-mlops-databricks-3".insteadOf "https://github.com/end-to-end-mlops-databricks-3"
