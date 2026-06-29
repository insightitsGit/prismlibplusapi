#!/usr/bin/env bash
# =============================================================================
# deploy/azure_cluster_run.sh
# Deploy the 4-container PrismLib cluster benchmark to Azure Container Apps
# and RUN the benchmark, capturing real cross-VNet numbers.
#
# Difference from azure_cluster_benchmark.sh: builds the image with
# `az acr build` (server-side) so no local Docker daemon is required, and
# runs the benchmark at the end, saving results to an Azure-specific file.
#
# Topology:
#   cae-prism-ab (westus2) — GREEN + BLUE  — same VNet / same environment
#   cae-prism-c  (westus2) — ORANGE        — separate VNet / environment
#   Benchmark runner runs on THIS machine (cross-network to all three).
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# Force UTF-8 so the Windows az CLI doesn't crash streaming build logs (cp1252)
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export AZURE_CORE_NO_COLOR=true

RG="${RG:-rg-prism-cluster-bench}"
LOCATION="${LOCATION:-westus2}"
ACR_NAME="${ACR_NAME:-prismclbenchreg24312}"   # reuse the ACR already created
IMAGE_TAG="${IMAGE_TAG:-cluster-bench:latest}"
ENV_AB="cae-prism-ab"
ENV_C="cae-prism-c"
APP_GREEN="node-green"; APP_BLUE="node-blue"; APP_ORANGE="node-orange"
ADMIN_EMAIL="${ADMIN_EMAIL:-insightits.info@gmail.com}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "STEP 1/9  Resource group $RG ($LOCATION)"
az group create --name "$RG" --location "$LOCATION" --output none

log "STEP 2/9  Container Registry $ACR_NAME"
az acr create --resource-group "$RG" --name "$ACR_NAME" \
  --sku Basic --admin-enabled true --output none

if [ "${SKIP_BUILD:-0}" = "1" ]; then
  log "STEP 3/9  SKIP_BUILD=1 — using existing image ${ACR_NAME}.azurecr.io/${IMAGE_TAG}"
else
  log "STEP 3/9  Building image server-side with az acr build (no local docker)"
  az acr build \
    --registry "$ACR_NAME" \
    --image "$IMAGE_TAG" \
    --file benchmark/cluster/Dockerfile \
    benchmark/cluster/ \
    --output none
  log "       image built: ${ACR_NAME}.azurecr.io/${IMAGE_TAG}"
fi

ACR_SERVER="${ACR_NAME}.azurecr.io"
ACR_USER=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

log "STEP 4/9  Container App Environment A (GREEN+BLUE, same VNet)"
az containerapp env create --name "$ENV_AB" --resource-group "$RG" \
  --location "$LOCATION" --output none

log "STEP 5/9  Container App Environment B (ORANGE, separate VNet)"
az containerapp env create --name "$ENV_C" --resource-group "$RG" \
  --location "$LOCATION" --output none

deploy_node() {
  local app="$1" env="$2" nid="$3" role="$4" net="$5"
  az containerapp create --name "$app" --resource-group "$RG" \
    --environment "$env" --image "${ACR_SERVER}/${IMAGE_TAG}" \
    --registry-server "$ACR_SERVER" --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --cpu 0.5 --memory 1.0Gi --min-replicas 1 --max-replicas 1 \
    --ingress external --target-port 8080 \
    --env-vars "NODE_ID=$nid" "NODE_ROLE=$role" "NETWORK_LABEL=$net" \
               "ADMIN_EMAIL=${ADMIN_EMAIL}" "TOKEN_BUDGET=100000" \
    --output none
}

log "STEP 6/9  Deploying GREEN, BLUE, ORANGE"
deploy_node "$APP_GREEN"  "$ENV_AB" "node-green"  "green"  "same-pod"
deploy_node "$APP_BLUE"   "$ENV_AB" "node-blue"   "blue"   "same-pod"
deploy_node "$APP_ORANGE" "$ENV_C"  "node-orange" "orange" "cross-network"

log "STEP 7/9  Collecting URLs and wiring PEERS"
get_fqdn() { az containerapp show --name "$1" --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv; }
URL_GREEN="https://$(get_fqdn "$APP_GREEN")"
URL_BLUE="https://$(get_fqdn "$APP_BLUE")"
URL_ORANGE="https://$(get_fqdn "$APP_ORANGE")"
log "       GREEN=$URL_GREEN"
log "       BLUE=$URL_BLUE"
log "       ORANGE=$URL_ORANGE"

az containerapp update --name "$APP_GREEN" --resource-group "$RG" \
  --set-env-vars "PEERS={\"blue\":\"${URL_BLUE}\",\"orange\":\"${URL_ORANGE}\"}" \
  --revision-suffix v2 --output none
az containerapp update --name "$APP_BLUE" --resource-group "$RG" \
  --set-env-vars "PEERS={\"green\":\"${URL_GREEN}\",\"orange\":\"${URL_ORANGE}\"}" \
  --revision-suffix v2 --output none
az containerapp update --name "$APP_ORANGE" --resource-group "$RG" \
  --set-env-vars "PEERS={\"green\":\"${URL_GREEN}\",\"blue\":\"${URL_BLUE}\"}" \
  --revision-suffix v2 --output none

cat > deploy/cluster_urls.env <<EOF
GREEN_URL=${URL_GREEN}
BLUE_URL=${URL_BLUE}
ORANGE_URL=${URL_ORANGE}
RG=${RG}
ACR_NAME=${ACR_NAME}
EOF

log "STEP 8/9  Waiting for nodes to become healthy"
sleep 30
for lu in "GREEN:${URL_GREEN}" "BLUE:${URL_BLUE}" "ORANGE:${URL_ORANGE}"; do
  label="${lu%%:*}"; url="${lu#*:}"
  for i in $(seq 1 30); do
    if curl -sf "${url}/health" >/dev/null 2>&1; then log "       ${label} healthy"; break; fi
    sleep 4
  done
done

log "STEP 9/9  Running cluster benchmark (cross-VNet, real Azure)"
python benchmark/cluster/run_cluster_benchmark.py \
  --green "${URL_GREEN}" --blue "${URL_BLUE}" --orange "${URL_ORANGE}"

# Preserve the Azure result under a distinct name (runner writes the default file)
cp benchmark/cluster/cluster_benchmark_results.json \
   benchmark/cluster/cluster_benchmark_results_azure.json
log "DONE. Azure results -> benchmark/cluster/cluster_benchmark_results_azure.json"
log "Resources LEFT RUNNING. Tear down with: az group delete --name ${RG} --yes --no-wait"
echo "AZURE_CLUSTER_BENCHMARK_COMPLETE"
