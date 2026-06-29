#!/usr/bin/env bash
# =============================================================================
# deploy/azure_cluster_benchmark.sh
# Deploy 4-container PrismLib cluster benchmark to Azure Container Apps
#
# Topology:
#   Environment A  (westus2) — GREEN + BLUE  — same VNet, same CAE
#   Environment B  (westus2) — ORANGE        — separate VNet, separate CAE
#   Benchmark runner runs locally (this machine)
#
# Prerequisites:
#   az login
#   docker (for building the image)
#   ACR must be accessible
# =============================================================================

set -euo pipefail

# ── Config — override with env vars if needed ──────────────────────────────
RG="${RG:-rg-prism-cluster-bench}"
LOCATION="${LOCATION:-westus2}"
ACR_NAME="${ACR_NAME:-prismbenchregistry}"
IMAGE_TAG="${IMAGE_TAG:-cluster-bench:latest}"

ENV_AB="cae-prism-ab"      # same VNet (GREEN + BLUE)
ENV_C="cae-prism-c"        # separate VNet (ORANGE)

APP_GREEN="node-green"
APP_BLUE="node-blue"
APP_ORANGE="node-orange"

ADMIN_EMAIL="${ADMIN_EMAIL:-insightits.info@gmail.com}"

# ── Log helpers ────────────────────────────────────────────────────────────
log()  { echo "[$(date +%H:%M:%S)] $*"; }
ok()   { echo "[$(date +%H:%M:%S)] ✓ $*"; }
fail() { echo "[$(date +%H:%M:%S)] ✗ $*" >&2; exit 1; }

# =============================================================================
# 1. Resource group
# =============================================================================
log "Creating resource group $RG in $LOCATION ..."
az group create --name "$RG" --location "$LOCATION" --output none
ok "Resource group ready"

# =============================================================================
# 2. Container Registry
# =============================================================================
log "Creating ACR $ACR_NAME ..."
az acr create \
  --resource-group "$RG" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  --output none
ok "ACR ready"

# Login and build
log "Building and pushing image ..."
az acr login --name "$ACR_NAME"

docker build \
  -f benchmark/cluster/Dockerfile \
  -t "${ACR_NAME}.azurecr.io/${IMAGE_TAG}" \
  benchmark/cluster/

docker push "${ACR_NAME}.azurecr.io/${IMAGE_TAG}"
ok "Image pushed: ${ACR_NAME}.azurecr.io/${IMAGE_TAG}"

# Pull ACR credentials
ACR_SERVER="${ACR_NAME}.azurecr.io"
ACR_USER=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)
ACR_PASS=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)

# =============================================================================
# 3. Container App Environments
# =============================================================================
log "Creating Container App Environment A (GREEN + BLUE, same VNet) ..."
az containerapp env create \
  --name "$ENV_AB" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --output none
ok "Environment A ready"

log "Creating Container App Environment B (ORANGE, separate VNet) ..."
az containerapp env create \
  --name "$ENV_C" \
  --resource-group "$RG" \
  --location "$LOCATION" \
  --output none
ok "Environment B ready"

# =============================================================================
# 4. Deploy GREEN (no peers yet — will update after all URLs are known)
# =============================================================================
log "Deploying GREEN node ..."
az containerapp create \
  --name "$APP_GREEN" \
  --resource-group "$RG" \
  --environment "$ENV_AB" \
  --image "${ACR_SERVER}/${IMAGE_TAG}" \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 1 --max-replicas 1 \
  --ingress external --target-port 8080 \
  --env-vars \
    "NODE_ID=node-green" \
    "NODE_ROLE=green" \
    "NETWORK_LABEL=same-pod" \
    "ADMIN_EMAIL=${ADMIN_EMAIL}" \
    "TOKEN_BUDGET=100000" \
  --output none
ok "GREEN deployed"

# =============================================================================
# 5. Deploy BLUE
# =============================================================================
log "Deploying BLUE node ..."
az containerapp create \
  --name "$APP_BLUE" \
  --resource-group "$RG" \
  --environment "$ENV_AB" \
  --image "${ACR_SERVER}/${IMAGE_TAG}" \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 1 --max-replicas 1 \
  --ingress external --target-port 8080 \
  --env-vars \
    "NODE_ID=node-blue" \
    "NODE_ROLE=blue" \
    "NETWORK_LABEL=same-pod" \
    "ADMIN_EMAIL=${ADMIN_EMAIL}" \
    "TOKEN_BUDGET=100000" \
  --output none
ok "BLUE deployed"

# =============================================================================
# 6. Deploy ORANGE (separate env)
# =============================================================================
log "Deploying ORANGE node ..."
az containerapp create \
  --name "$APP_ORANGE" \
  --resource-group "$RG" \
  --environment "$ENV_C" \
  --image "${ACR_SERVER}/${IMAGE_TAG}" \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --cpu 0.5 --memory 1.0Gi \
  --min-replicas 1 --max-replicas 1 \
  --ingress external --target-port 8080 \
  --env-vars \
    "NODE_ID=node-orange" \
    "NODE_ROLE=orange" \
    "NETWORK_LABEL=cross-network" \
    "ADMIN_EMAIL=${ADMIN_EMAIL}" \
    "TOKEN_BUDGET=100000" \
  --output none
ok "ORANGE deployed"

# =============================================================================
# 7. Collect URLs
# =============================================================================
log "Collecting node URLs ..."
URL_GREEN=$(az containerapp show \
  --name "$APP_GREEN" --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv)
URL_BLUE=$(az containerapp show \
  --name "$APP_BLUE" --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv)
URL_ORANGE=$(az containerapp show \
  --name "$APP_ORANGE" --resource-group "$RG" \
  --query "properties.configuration.ingress.fqdn" -o tsv)

URL_GREEN="https://${URL_GREEN}"
URL_BLUE="https://${URL_BLUE}"
URL_ORANGE="https://${URL_ORANGE}"

ok "GREEN:  $URL_GREEN"
ok "BLUE:   $URL_BLUE"
ok "ORANGE: $URL_ORANGE"

# Build PEERS JSON for each node
PEERS_GREEN="{\"blue\":\"${URL_BLUE}\",\"orange\":\"${URL_ORANGE}\"}"
PEERS_BLUE="{\"green\":\"${URL_GREEN}\",\"orange\":\"${URL_ORANGE}\"}"
PEERS_ORANGE="{\"green\":\"${URL_GREEN}\",\"blue\":\"${URL_BLUE}\"}"

# =============================================================================
# 8. Update PEERS env vars (now that all URLs are known)
# =============================================================================
log "Injecting PEERS into all nodes ..."

az containerapp update \
  --name "$APP_GREEN" --resource-group "$RG" \
  --set-env-vars "PEERS=${PEERS_GREEN}" \
  --revision-suffix v2 --output none

az containerapp update \
  --name "$APP_BLUE" --resource-group "$RG" \
  --set-env-vars "PEERS=${PEERS_BLUE}" \
  --revision-suffix v2 --output none

az containerapp update \
  --name "$APP_ORANGE" --resource-group "$RG" \
  --set-env-vars "PEERS=${PEERS_ORANGE}" \
  --revision-suffix v2 --output none

ok "PEERS injected — nodes will restart"

# =============================================================================
# 9. Wait for nodes to be healthy
# =============================================================================
log "Waiting 30s for nodes to restart with peer config ..."
sleep 30

for label_url in "GREEN:${URL_GREEN}" "BLUE:${URL_BLUE}" "ORANGE:${URL_ORANGE}"; do
  label="${label_url%%:*}"
  url="${label_url#*:}"
  for i in $(seq 1 20); do
    if curl -sf "${url}/health" > /dev/null 2>&1; then
      ok "${label} is healthy"
      break
    fi
    sleep 3
  done
done

# =============================================================================
# 10. Print benchmark command
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All nodes deployed. Run the benchmark:"
echo ""
echo "  cd benchmark/cluster"
echo "  pip install -r requirements.txt"
echo "  python run_cluster_benchmark.py \\"
echo "    --green  ${URL_GREEN} \\"
echo "    --blue   ${URL_BLUE} \\"
echo "    --orange ${URL_ORANGE}"
echo ""
echo "  To tear down everything:"
echo "  az group delete --name ${RG} --yes --no-wait"
echo "════════════════════════════════════════════════════════════"

# Save URLs for teardown / re-runs
cat > deploy/cluster_urls.env << EOF
GREEN_URL=${URL_GREEN}
BLUE_URL=${URL_BLUE}
ORANGE_URL=${URL_ORANGE}
RG=${RG}
EOF
ok "URLs saved to deploy/cluster_urls.env"
