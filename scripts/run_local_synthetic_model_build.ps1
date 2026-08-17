$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

# The bind mount persists synthetic data and model artefacts on the developer
# machine while the CPU-only training dependencies remain isolated in Docker.
docker build -f infra/docker/Dockerfile.training -t smartbank-ai-training:local .
docker run --rm -v "${repo}:/workspace" smartbank-ai-training:local
