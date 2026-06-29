#!/usr/bin/env bash
# deploy/azure_driver_run.sh — 2-container PrismDriver e2e on Azure Container Apps
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 AZURE_CORE_NO_COLOR=true

RG="${RG:-rg-prism-driver-e2e}"
LOCATION="${LOCATION:-westus2}"
ACR_NAME="${ACR_NAME:-prismdriverbench$RANDOM}"
ENV_NAME="cae-prism-driver"
APP_WRAPPER="prism-wrapper-sim"
APP_BENCH="prism-benchmark"
WRAPPER_TAG="prism-wrapper-sim:latest"
BENCH_TAG="prism-benchmark:latest"
USERS="${BENCH_USERS:-20}"
DURATION="${BENCH_DURATION:-45}"
WARMUP="${BENCH_WARMUP:-1000}"
LOG_DIR="benchmark/results/azure_e2e_logs"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "STEP 1/10  Resource group $RG ($LOCATION)"
az group create --name "$RG" --location "$LOCATION" --output none

log "STEP 2/10  Container Registry $ACR_NAME"
az acr create --resource-group "$RG" --name "$ACR_NAME" \
  --sku Basic --admin-enabled true --output none 2>/dev/null || true

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  log "STEP 3/10  SKIP_BUILD=1 — using existing images"
else
  log "STEP 3/10  Building wrapper-sim + benchmark images"
az acr build --registry "$ACR_NAME" --image "$WRAPPER_TAG" \
    --file benchmark/Dockerfile.wrapper . --no-logs --output none
  az acr build --registry "$ACR_NAME" --image "$BENCH_TAG" \
    --file benchmark/Dockerfile . --no-logs --output none
fi

ACR_SERVER="${ACR_NAME}.azurecr.io"
ACR_USER=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

log "STEP 4/10  Container App Environment"
az containerapp env create --name "$ENV_NAME" --resource-group "$RG" \
  --location "$LOCATION" --output none 2>/dev/null || true

log "STEP 5/10  Deploy DB node ($APP_WRAPPER)"
az containerapp create --name "$APP_WRAPPER" --resource-group "$RG" \
  --environment "$ENV_NAME" --image "${ACR_SERVER}/${WRAPPER_TAG}" \
  --registry-server "$ACR_SERVER" --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1 \
  --ingress external --target-port 8001 \
  --env-vars "PRISM_TENANT_ID=benchmark-db" "PRISM_TARGET_DIM=64" \
  --output none 2>/dev/null || \
az containerapp update --name "$APP_WRAPPER" --resource-group "$RG" \
  --image "${ACR_SERVER}/${WRAPPER_TAG}" --output none

URL_DB="https://$(az containerapp show --name "$APP_WRAPPER" --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv)"
log "       DB node: $URL_DB"

log "STEP 6/10  Deploy app node ($APP_BENCH)"
az containerapp create --name "$APP_BENCH" --resource-group "$RG" \
  --environment "$ENV_NAME" --image "${ACR_SERVER}/${BENCH_TAG}" \
  --registry-server "$ACR_SERVER" --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --cpu 1.0 --memory 2.0Gi --min-replicas 1 --max-replicas 1 \
  --ingress external --target-port 8000 \
  --env-vars \
    "PRISM_WRAPPER_URL=${URL_DB}" \
    "PRISM_WRAPPER_PORT=443" \
    "PRISM_TENANT_ID=benchmark-app" \
    "PRISM_TARGET_DIM=64" \
    "PRISM_BENCHMARK_SMOKE=1" \
  --output none 2>/dev/null || \
az containerapp update --name "$APP_BENCH" --resource-group "$RG" \
  --image "${ACR_SERVER}/${BENCH_TAG}" \
  --set-env-vars \
    "PRISM_WRAPPER_URL=${URL_DB}" \
    "PRISM_WRAPPER_PORT=443" \
    "PRISM_TENANT_ID=benchmark-app" \
    "PRISM_TARGET_DIM=64" \
    "PRISM_BENCHMARK_SMOKE=1" \
  --output none

URL_APP="https://$(az containerapp show --name "$APP_BENCH" --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv)"
log "       App node: $URL_APP"

cat > deploy/driver_urls.env <<EOF
APP_URL=${URL_APP}
DB_URL=${URL_DB}
RG=${RG}
ACR_NAME=${ACR_NAME}
ENV_NAME=${ENV_NAME}
EOF

log "STEP 7/10  Health checks"
sleep 25
for label_url in "DB:${URL_DB}" "App:${URL_APP}"; do
  label="${label_url%%:*}"; url="${label_url#*:}"
  for i in $(seq 1 30); do
    if curl -sf "${url}/health" >/dev/null 2>&1; then log "       ${label} healthy"; break; fi
    sleep 4
  done
done

log "STEP 8/10  Driver status"
curl -sf "${URL_APP}/driver/status" | python -m json.tool || true

log "STEP 9/10  Running benchmark"
python benchmark/load/run_driver_benchmark.py \
  --app-url "$URL_APP" \
  --db-url "$URL_DB" \
  --users "$USERS" \
  --duration "$DURATION" \
  --warmup-rows "$WARMUP" \
  --capture-logs \
  --resource-group "$RG" \
  --app-name "$APP_BENCH" \
  --wrapper-name "$APP_WRAPPER" \
  --log-dir "$LOG_DIR"

log "STEP 10/10  Saving Azure results copy"
cp "$(ls -t benchmark/results/driver_benchmark_*.json | head -1)" \
   benchmark/results/driver_benchmark_azure.json
log "DONE -> benchmark/results/driver_benchmark_azure.json"
log "Tear down: az group delete --name ${RG} --yes --no-wait"
echo "AZURE_DRIVER_E2E_COMPLETE"
