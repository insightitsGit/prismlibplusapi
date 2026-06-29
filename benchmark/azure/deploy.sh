#!/usr/bin/env bash
# PrismLib Benchmark — Azure deployment script
#
# Prerequisites:
#   brew install azure-cli          # macOS
#   az login
#   az account set --subscription <your-subscription-id>
#
# Usage:
#   ./benchmark/azure/deploy.sh [resource-group] [location]
#
# Example:
#   OPENAI_API_KEY=sk-... ./benchmark/azure/deploy.sh rg-prism-benchmark eastus

set -euo pipefail

RG="${1:-rg-prism-benchmark}"
LOCATION="${2:-eastus}"
PREFIX="prism"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "── PrismLib Benchmark Deployment ──────────────────────────────"
echo "  Resource Group : $RG"
echo "  Location       : $LOCATION"
echo "  Image tag      : $IMAGE_TAG"
echo "───────────────────────────────────────────────────────────────"

# 1. Create resource group
echo "[1/6] Creating resource group..."
az group create --name "$RG" --location "$LOCATION" --output none

# 2. Deploy infrastructure
echo "[2/6] Deploying Azure infra (Bicep)..."
DEPLOY_OUT=$(az deployment group create \
  --resource-group "$RG" \
  --template-file "benchmark/azure/infra.bicep" \
  --parameters "@benchmark/azure/params.json" \
  --parameters openAiApiKey="${OPENAI_API_KEY:-}" \
  --query "properties.outputs" \
  --output json)

APP_URL=$(echo "$DEPLOY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['appUrl']['value'])")
REGISTRY=$(echo "$DEPLOY_OUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['registryServer']['value'])")
AI_CONN=$(echo "$DEPLOY_OUT"  | python3 -c "import sys,json; print(json.load(sys.stdin)['appInsightsConnectionString']['value'])")

echo "  App URL     : $APP_URL"
echo "  Registry    : $REGISTRY"

# 3. Build Docker image
echo "[3/6] Building Docker image..."
docker build -t "prism-benchmark:$IMAGE_TAG" -f benchmark/Dockerfile .

# 4. Push to ACR
echo "[4/6] Pushing to Azure Container Registry..."
az acr login --name "${REGISTRY%%.*}"
docker tag "prism-benchmark:$IMAGE_TAG" "$REGISTRY/prism-benchmark:$IMAGE_TAG"
docker push "$REGISTRY/prism-benchmark:$IMAGE_TAG"

# 5. Trigger new revision
echo "[5/6] Updating Container App revision..."
az containerapp update \
  --name "${PREFIX}-benchmark" \
  --resource-group "$RG" \
  --image "$REGISTRY/prism-benchmark:$IMAGE_TAG" \
  --output none

# 6. Wait for healthy
echo "[6/6] Waiting for app to become healthy..."
for i in $(seq 1 30); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$APP_URL/health" 2>/dev/null || echo "000")
  if [ "$STATUS" = "200" ]; then
    echo "  ✓ App is healthy at $APP_URL"
    break
  fi
  echo "  waiting... ($i/30)"
  sleep 5
done

# Save connection string for benchmark runner
echo "APPLICATIONINSIGHTS_CONNECTION_STRING=$AI_CONN" >> benchmark/.env
echo "BENCHMARK_APP_URL=$APP_URL"                     >> benchmark/.env

echo ""
echo "── Deployment complete ─────────────────────────────────────────"
echo "  App URL  : $APP_URL"
echo "  Metrics  : $APP_URL/metrics"
echo "  AI Portal: https://portal.azure.com (search: $PREFIX-insights)"
echo ""
echo "  Next steps:"
echo "    python benchmark/load/run_benchmark.py --host $APP_URL --scenario mixed"
echo "───────────────────────────────────────────────────────────────"
