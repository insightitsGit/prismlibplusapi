// PrismLib Benchmark — Azure Infrastructure
// Provisions: Container App, Container Registry, Application Insights, Log Analytics
//
// Deploy:
//   az deployment group create \
//     --resource-group rg-prism-benchmark \
//     --template-file benchmark/azure/infra.bicep \
//     --parameters @benchmark/azure/params.json

@description('Azure region for all resources')
param location string = resourceGroup().location

@description('Short prefix for resource names (e.g. prism)')
param prefix string = 'prism'

@description('Container image tag to deploy')
param imageTag string = 'latest'

@description('LLM model to use (set to empty to use mock LLM)')
param llmModel string = ''

@secure()
@description('OpenAI API key — leave empty to use mock LLM')
param openAiApiKey string = ''

@description('PrismCache similarity threshold')
param similarityThreshold string = '0.92'

// ---------------------------------------------------------------------------
// Log Analytics Workspace
// ---------------------------------------------------------------------------
resource logWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${prefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ---------------------------------------------------------------------------
// Application Insights
// ---------------------------------------------------------------------------
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${prefix}-insights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logWorkspace.id
    RetentionInDays: 30
  }
}

// ---------------------------------------------------------------------------
// Container Registry
// ---------------------------------------------------------------------------
resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${replace(prefix, '-', '')}benchregistry'
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: true
  }
}

// ---------------------------------------------------------------------------
// Container Apps Environment
// ---------------------------------------------------------------------------
resource caEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${prefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logWorkspace.properties.customerId
        sharedKey: logWorkspace.listKeys().primarySharedKey
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Benchmark Container App
// ---------------------------------------------------------------------------
resource benchmarkApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${prefix}-benchmark'
  location: location
  properties: {
    managedEnvironmentId: caEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'OPTIONS']
        }
      }
      registries: [
        {
          server: registry.properties.loginServer
          username: registry.listCredentials().username
          passwordSecretRef: 'registry-password'
        }
      ]
      secrets: [
        {
          name: 'registry-password'
          value: registry.listCredentials().passwords[0].value
        }
        {
          name: 'openai-api-key'
          value: empty(openAiApiKey) ? 'not-set' : openAiApiKey
        }
        {
          name: 'appinsights-connection-string'
          value: appInsights.properties.ConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'benchmark'
          image: '${registry.properties.loginServer}/prism-benchmark:${imageTag}'
          env: [
            { name: 'PRISM_TENANT_ID',        value: 'azure-benchmark' }
            { name: 'PRISM_THRESHOLD',         value: similarityThreshold }
            { name: 'LLM_MODEL',               value: empty(llmModel) ? 'mock' : llmModel }
            { name: 'PRISM_BENCHMARK_SMOKE',   value: empty(openAiApiKey) ? '1' : '0' }
            { name: 'OPENAI_API_KEY',          secretRef: 'openai-api-key' }
            { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights-connection-string' }
          ]
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 20
              periodSeconds: 10
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 5
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scale'
            http: { metadata: { concurrentRequests: '50' } }
          }
        ]
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------
output appUrl string = 'https://${benchmarkApp.properties.configuration.ingress.fqdn}'
output registryServer string = registry.properties.loginServer
output appInsightsKey string = appInsights.properties.InstrumentationKey
output appInsightsConnectionString string = appInsights.properties.ConnectionString
