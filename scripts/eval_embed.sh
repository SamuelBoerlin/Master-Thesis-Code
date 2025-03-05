#!/usr/bin/env bash

set -e

EXPERIMENT_REPOSITORY="${EXPERIMENT_REPOSITORY:-https://github.com/SamuelBoerlin/Master-Thesis-Code.git}"
EXPERIMENT_REPOSITORY_PROJECT_PATH="${EXPERIMENT_REPOSITORY_PROJECT_PATH:-.}"

EXPERIMENT_BASE_DIR="${EXPERIMENT_BASE_DIR:-/nas/experiments}"

EVAL_BASE_DIR="${EVAL_BASE_DIR:-/nas/eval}"

experiment_name="$1"
shift

experiment_name="$(date '+%Y_%m_%d_%+s')_${experiment_name}"
experiment_dir="${EXPERIMENT_BASE_DIR}/${experiment_name}"
eval_output_dir="${EVAL_BASE_DIR}/${experiment_name}"

EXPERIMENT_CONDA_ENV_BASE="${EXPERIMENT_CONDA_ENV_BASE:-rvs-experiment-base-deps}" ./experiment.sh "$EXPERIMENT_REPOSITORY" "$EXPERIMENT_REPOSITORY_PROJECT_PATH" "$experiment_name" "$experiment_dir" ervs_embed "$@" --output-dir "$eval_output_dir"
