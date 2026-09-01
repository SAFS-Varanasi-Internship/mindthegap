# This script exists to reproduce LOCALLY what SkyPilot injects REMOTELY.
# On a VM it must do nothing: SkyPilot already sets HF_TOKEN/HF_BUCKET from the
# task YAML `envs:` block plus `--env-file .env` at launch, and the SKYPILOT_*
# client settings below only make sense on the machine that submits jobs.
#
# SKYPILOT_NUM_NODES is the guard because SkyPilot injects it into both `setup`
# and `run` (SKYPILOT_NODE_RANK is only set for `run`).
if [ -z "$SKYPILOT_NUM_NODES" ]; then

  if [ -f "$PIXI_PROJECT_ROOT/.env" ]; then
    set -a
    . "$PIXI_PROJECT_ROOT/.env"
    set +a
  fi

  # Tasks build "https://user:pass@$SKYPILOT_API_ENDPOINT", so strip any scheme
  # in case .env stores the endpoint as a full URL instead of a bare host.
  SKYPILOT_API_ENDPOINT="${SKYPILOT_API_ENDPOINT#http://}"
  SKYPILOT_API_ENDPOINT="${SKYPILOT_API_ENDPOINT#https://}"
  export SKYPILOT_API_ENDPOINT

  # SkyPilot resolves the project config (.sky.yaml, e.g. active_workspace) by
  # checking for it relative to the current working directory, not the repo
  # root. Pin it explicitly so workspace settings (like the escience workspace's
  # enabled clouds) still apply when running `sky` from a subdirectory such as
  # examples/vector_database.
  if [ -f "$PIXI_PROJECT_ROOT/.sky.yaml" ]; then
    export SKYPILOT_PROJECT_CONFIG="$PIXI_PROJECT_ROOT/.sky.yaml"
  fi

fi
