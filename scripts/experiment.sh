#!/usr/bin/env bash

set -e

EXPERIMENT_CONDA_ENV_BASE="${EXPERIMENT_CONDA_ENV_BASE:-experiment-base-deps}"
EXPERIMENT_CONDA_ENV_PREFIX="${EXPERIMENT_CONDA_ENV_PREFIX:-experiment_}"
EXPERIMENT_CONDA_ENV_SETUP_SCRIPT="${EXPERIMENT_CONDA_ENV_SETUP_SCRIPT:-env_setup.sh}"

EXPERIMENT_TMUX_SESSION_PREFIX="${EXPERIMENT_TMUX_SESSION_PREFIX:-experiment_}"

experiment_repository="$1"
shift

experiment_repository_project_path="$1"
shift

experiment_name="$1"
shift

if [[ "$experiment_name" =~ [^a-zA-Z0-9_] ]]; then
  echo "Invalid experiment name '${experiment_name}'" >&2
  exit 1
fi

experiment_dir="$1"
shift

if [ -d "$experiment_dir" ]; then
  echo "Experiment dir '${experiment_dir}' already exists" >&2
  exit 1
fi

mkdir "$experiment_dir"
cd "$experiment_dir"

echo "Cloning repository..."

git clone "$experiment_repository" .
cd "$experiment_repository_project_path"

echo "Creating venv..."

conda_env_name="${EXPERIMENT_CONDA_ENV_PREFIX}${experiment_name}"
conda create --name "$conda_env_name" --clone "$EXPERIMENT_CONDA_ENV_BASE"

echo "Running project setup..."

conda run --name "$conda_env_name" "$EXPERIMENT_CONDA_ENV_SETUP_SCRIPT"

echo "Creating experiment script..."

experiment_script_file="${PWD}/experiment_script.sh"
cat >"$experiment_script_file" <<EOF
#!${SHELL}
cd "${PWD}"
conda run --name "${conda_env_name}" $@ 2>&1 | tee experiment_log.log
exec "${SHELL}"
EOF
chmod +x "$experiment_script_file"

echo "Starting session..."

tmux_session_name="${EXPERIMENT_TMUX_SESSION_PREFIX}${experiment_name}"
tmux new-session -d -s "$tmux_session_name" "$experiment_script_file"

echo "Done..."
