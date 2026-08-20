param(
  [switch]$KeepRunning
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:ENVIRONMENT = 'development'
$tokenBytes = New-Object byte[] 32
$random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$random.GetBytes($tokenBytes)
$random.Dispose()
$env:SERVICE_AUTH_TOKEN = -join ($tokenBytes | ForEach-Object { $_.ToString('x2') })
$env:LOG_LEVEL = 'INFO'

try {
  # Agent Dockerfiles inherit from this local image. Build it first so Docker
  # never tries to resolve a private development image from a public registry.
  docker compose -f infra/docker/docker-compose.yml build base-builder
  # Starting the orchestrator brings up its declared agent and ChromaDB
  # dependencies, but deliberately excludes the image-only base-builder service.
  docker compose -f infra/docker/docker-compose.yml up --build -d orchestrator
  docker compose -f infra/docker/docker-compose.yml ps

  $health = $null
  for ($attempt = 1; $attempt -le 30; $attempt++) {
    try {
      $health = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/health' -TimeoutSec 5
      break
    } catch {
      Start-Sleep -Seconds 2
    }
  }

  if ($null -eq $health) {
    throw 'The orchestrator health endpoint did not become available within 60 seconds.'
  }

  $payload = [ordered]@{
    contract_version = '2026-08-01'
    request_type = 'fraud_check'
    tenant_id = '4'
    correlation_id = [guid]::NewGuid().ToString()
    requested_at = [DateTime]::UtcNow.ToString('o')
    payload = [ordered]@{
      transaction_id = 'LOCAL-SMOKE-001'
      amount_ngn = 10000
      channel = 'mobile'
      hour_of_day = 10
      day_of_week = 1
      merchant_category = 'groceries'
      origin_region = 'Lagos'
      sender_30d_avg_amount = 8500
      sender_txn_count_1h = 1
    }
  } | ConvertTo-Json -Depth 5

  $headers = @{
    'X-Service-Token' = $env:SERVICE_AUTH_TOKEN
    'X-Client-ID' = 'smartbank-platform'
  }
  $route = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8001/v1/route' -ContentType 'application/json' -Headers $headers -Body $payload -TimeoutSec 15

  if ($route.human_review_required -ne $true) {
    throw 'The orchestration response did not enforce human review.'
  }
  if ($route.status -notin @('advisory', 'unavailable')) {
    throw "Unexpected advisory status: $($route.status)"
  }

  Write-Host "ORCHESTRATOR_HEALTH=$($health.status)"
  Write-Host "ADVISORY_STATUS=$($route.status)"
  Write-Host "HUMAN_REVIEW_REQUIRED=$($route.human_review_required)"
  Write-Host "CONTRACT_VERSION=$($route.contract_version)"
  Write-Host 'LOCAL_COMPOSE_VALIDATION=PASS'
} finally {
  Remove-Item Env:SERVICE_AUTH_TOKEN -ErrorAction SilentlyContinue
  Remove-Item Env:ENVIRONMENT -ErrorAction SilentlyContinue
  Remove-Item Env:LOG_LEVEL -ErrorAction SilentlyContinue
  if (-not $KeepRunning) {
    docker compose -f infra/docker/docker-compose.yml down --remove-orphans
  }
}
