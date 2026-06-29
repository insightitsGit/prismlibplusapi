# deploy/azure_driver_run.ps1
# Deploy 2-container PrismDriver end-to-end benchmark to Azure Container Apps:
#   Container 1 — wrapper-sim (DB node) :8001
#   Container 2 — benchmark app (PrismDriver) :8000
# Then run run_driver_benchmark.py and capture container logs.

$ErrorActionPreference = "Stop"
$WarningPreference = "SilentlyContinue"
Set-Location (Split-Path $PSScriptRoot -Parent)

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:AZURE_CORE_NO_COLOR = "true"

$RG         = if ($env:RG)         { $env:RG }         else { "rg-prism-driver-e2e" }
$LOCATION   = if ($env:LOCATION)   { $env:LOCATION }   else { "westus2" }
$ACR_NAME   = if ($env:ACR_NAME)   { $env:ACR_NAME }   else { "prismdriverbench" + (Get-Random -Maximum 99999) }
$ENV_NAME   = "cae-prism-driver"
$APP_WRAPPER = "prism-wrapper-sim"
$APP_BENCH   = "prism-benchmark"
$WRAPPER_TAG = "prism-wrapper-sim:latest"
$BENCH_TAG   = "prism-benchmark:latest"
$USERS       = if ($env:BENCH_USERS)     { $env:BENCH_USERS }     else { "20" }
$DURATION    = if ($env:BENCH_DURATION)  { $env:BENCH_DURATION }  else { "45" }
$WARMUP      = if ($env:BENCH_WARMUP)   { $env:BENCH_WARMUP }    else { "1000" }
$SKIP_BUILD  = if ($env:SKIP_BUILD)     { $env:SKIP_BUILD }      else { "0" }
$LOG_DIR     = "benchmark/results/azure_e2e_logs"

function Log($msg) { Write-Host ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg) }

Log "STEP 1/10  Resource group $RG ($LOCATION)"
az group create --name $RG --location $LOCATION --output none

Log "STEP 2/10  Container Registry $ACR_NAME"
az acr create --resource-group $RG --name $ACR_NAME `
  --sku Basic --admin-enabled true --output none 2>$null
if ($LASTEXITCODE -ne 0) {
  Log "       ACR may already exist — continuing"
}

if ($SKIP_BUILD -eq "1") {
  Log "STEP 3/10  SKIP_BUILD=1 — using existing images"
} else {
  Log "STEP 3/10  Building wrapper-sim image (ACR build)"
  az acr build --registry $ACR_NAME --image $WRAPPER_TAG `
    --file benchmark/Dockerfile.wrapper . --no-logs --output none
  if ($LASTEXITCODE -ne 0) { throw "wrapper-sim ACR build failed" }

  Log "       Building benchmark app image (ACR build)"
  az acr build --registry $ACR_NAME --image $BENCH_TAG `
    --file benchmark/Dockerfile . --no-logs --output none
  if ($LASTEXITCODE -ne 0) { throw "benchmark app ACR build failed" }
}

$ACR_SERVER = "$ACR_NAME.azurecr.io"
$ACR_USER   = az acr credential show --name $ACR_NAME --query username -o tsv
$ACR_PASS   = az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv

Log "STEP 4/10  Container App Environment $ENV_NAME"
az containerapp env create --name $ENV_NAME --resource-group $RG `
  --location $LOCATION --output none 2>$null

Log "STEP 5/10  Deploy DB node ($APP_WRAPPER)"
az containerapp create --name $APP_WRAPPER --resource-group $RG `
  --environment $ENV_NAME --image "${ACR_SERVER}/${WRAPPER_TAG}" `
  --registry-server $ACR_SERVER --registry-username $ACR_USER `
  --registry-password $ACR_PASS `
  --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1 `
  --ingress external --target-port 8001 `
  --env-vars "PRISM_TENANT_ID=benchmark-db" "PRISM_TARGET_DIM=64" `
  --output none 2>$null
if ($LASTEXITCODE -ne 0) {
  Log "       Updating existing $APP_WRAPPER revision"
  az containerapp update --name $APP_WRAPPER --resource-group $RG `
    --image "${ACR_SERVER}/${WRAPPER_TAG}" --output none
}

$URL_DB = "https://" + (az containerapp show --name $APP_WRAPPER --resource-group $RG `
  --query "properties.configuration.ingress.fqdn" -o tsv)
Log "       DB node URL: $URL_DB"

Log "STEP 6/10  Deploy app node ($APP_BENCH)"
az containerapp create --name $APP_BENCH --resource-group $RG `
  --environment $ENV_NAME --image "${ACR_SERVER}/${BENCH_TAG}" `
  --registry-server $ACR_SERVER --registry-username $ACR_USER `
  --registry-password $ACR_PASS `
  --cpu 1.0 --memory 2.0Gi --min-replicas 1 --max-replicas 1 `
  --ingress external --target-port 8000 `
  --env-vars `
    "PRISM_WRAPPER_URL=$URL_DB" `
    "PRISM_WRAPPER_PORT=443" `
    "PRISM_TENANT_ID=benchmark-app" `
    "PRISM_TARGET_DIM=64" `
    "PRISM_BENCHMARK_SMOKE=1" `
  --output none 2>$null
if ($LASTEXITCODE -ne 0) {
  Log "       Updating existing $APP_BENCH revision"
  az containerapp update --name $APP_BENCH --resource-group $RG `
    --image "${ACR_SERVER}/${BENCH_TAG}" `
    --set-env-vars `
      "PRISM_WRAPPER_URL=$URL_DB" `
      "PRISM_WRAPPER_PORT=443" `
      "PRISM_TENANT_ID=benchmark-app" `
      "PRISM_TARGET_DIM=64" `
      "PRISM_BENCHMARK_SMOKE=1" `
    --output none
}

$URL_APP = "https://" + (az containerapp show --name $APP_BENCH --resource-group $RG `
  --query "properties.configuration.ingress.fqdn" -o tsv)
Log "       App node URL: $URL_APP"

@"
APP_URL=$URL_APP
DB_URL=$URL_DB
RG=$RG
ACR_NAME=$ACR_NAME
ENV_NAME=$ENV_NAME
"@ | Set-Content -Encoding utf8 deploy/driver_urls.env

Log "STEP 7/10  Waiting for health checks"
Start-Sleep -Seconds 25
foreach ($pair in @(@{Label="DB"; Url=$URL_DB}, @{Label="App"; Url=$URL_APP})) {
  $ok = $false
  for ($i = 1; $i -le 30; $i++) {
    try {
      $r = Invoke-WebRequest -Uri "$($pair.Url)/health" -UseBasicParsing -TimeoutSec 8
      if ($r.StatusCode -eq 200) { Log "       $($pair.Label) healthy"; $ok = $true; break }
    } catch {}
    Start-Sleep -Seconds 4
  }
  if (-not $ok) { throw "$($pair.Label) node not healthy at $($pair.Url)" }
}

Log "STEP 8/10  Driver status check"
$status = Invoke-RestMethod -Uri "$URL_APP/driver/status" -TimeoutSec 15
$status | ConvertTo-Json -Depth 5 | Write-Host

Log "STEP 9/10  Running driver benchmark ($USERS users x ${DURATION}s per phase)"
python benchmark/load/run_driver_benchmark.py `
  --app-url $URL_APP `
  --db-url $URL_DB `
  --users $USERS `
  --duration $DURATION `
  --warmup-rows $WARMUP `
  --capture-logs `
  --resource-group $RG `
  --app-name $APP_BENCH `
  --wrapper-name $APP_WRAPPER `
  --log-dir $LOG_DIR

Log "STEP 10/10  Saving Azure copy of results"
$latest = Get-ChildItem benchmark/results/driver_benchmark_*.json |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) {
  Copy-Item $latest.FullName benchmark/results/driver_benchmark_azure.json -Force
  Log "DONE. Results -> benchmark/results/driver_benchmark_azure.json"
  Log "Logs      -> $LOG_DIR/"
  Get-Content $latest.FullName
}
Log "Tear down: az group delete --name $RG --yes --no-wait"
Write-Host "AZURE_DRIVER_E2E_COMPLETE"
