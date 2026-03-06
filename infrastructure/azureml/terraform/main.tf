# Azure ML Workspace Configuration
# Sentinel AI Engine - ML Infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "prod"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "resource_group_name" {
  description = "Resource group name"
  type        = string
  default     = "rg-sentinel-ai"
}

# Resource Group
resource "azurerm_resource_group" "ml_rg" {
  name     = var.resource_group_name
  location = var.location

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
    ManagedBy   = "terraform"
  }
}

# Azure ML Workspace
resource "azurerm_machine_learning_workspace" "ml_workspace" {
  name                = "mlw-sentinel-${var.environment}"
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  sku                 = "Enterprise"

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Storage Account for ML
resource "azurerm_storage_account" "ml_storage" {
  name                     = "stsaentinel${var.environment}"
  location                 = azurerm_resource_group.ml_rg.location
  resource_group_name      = azurerm_resource_group.ml_rg.name
  account_tier            = "Standard"
  account_replication_type = "GRS"
  account_kind            = "StorageV2"

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Key Vault for ML secrets
resource "azurerm_key_vault" "ml_keyvault" {
  name                = "kv-sentinel-${var.environment}"
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name           = "standard"

  purge_protection_enabled = false

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Container Registry for model images
resource "azurerm_container_registry" "ml_registry" {
  name                   = "crsentinel${var.environment}"
  location               = azurerm_resource_group.ml_rg.location
  resource_group_name    = azurerm_resource_group.ml_rg.name
  sku                    = "Premium"
  admin_enabled         = true

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Application Insights for ML monitoring
resource "azurerm_application_insights" "ml_insights" {
  name                = "appi-sentinel-${var.environment}"
  location            = azurerm_resource_group.ml_rg.location
  resource_group_name = azurerm_resource_group.ml_rg.name
  application_type    = "other"

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Link ML workspace to resources
resource "azurerm_machine_learning_workspace_linked_service" "storage_link" {
  name             = "linked-storage"
  workspace_id      = azurerm_machine_learning_workspace.ml_workspace.id
  linked_service_id = azurerm_storage_account.ml_storage.id
  type             = "AzureStorage"
}

resource "azurerm_machine_learning_workspace_linked_service" "keyvault_link" {
  name             = "linked-keyvault"
  workspace_id      = azurerm_machine_learning_workspace.ml_workspace.id
  linked_service_id = azurerm_key_vault.ml_keyvault.id
  type             = "AzureKeyVault"
}

# Compute Instance for development
resource "azurerm_machine_learning_compute_instance" "dev_instance" {
  name                = "ci-dev-01"
  location            = azurerm_resource_group.ml_rg.location
  workspace_id        = azurerm_machine_learning_workspace.ml_workspace.id
  vm_size            = "Standard_DS11_v2"
  subnet_resource_id  = var.subnet_id

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Compute Cluster for training
resource "azurerm_machine_learning_compute_cluster" "training_cluster" {
  name                = "gpu-cluster"
  location            = azurerm_resource_group.ml_rg.location
  workspace_id        = azurerm_machine_learning_workspace.ml_workspace.id
  vm_size            = "Standard_NC4as_T4_v3"
  node_count         = 2
  min_node_count     = 0
  max_node_count     = 4
  subnet_resource_id  = var.subnet_id

  scale_settings {
    scale_type = "Auto"
    idle_time_before_scale_down = 300
  }

  identity {
    type = "SystemAssigned"
  }

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Inference Cluster (AKS)
resource "azurerm_machine_learning_inferencing_cluster" "aks_cluster" {
  name                = "aks-inference"
  location            = azurerm_resource_group.ml_rg.location
  workspace_id        = azurerm_machine_learning_workspace.ml_workspace.id
  kubernetes_cluster_id = var.aks_cluster_id

  tags = {
    Environment = var.environment
    Project     = "sentinel-ai-engine"
  }
}

# Outputs
output "workspace_id" {
  value = azurerm_machine_learning_workspace.ml_workspace.id
}

output "workspace_name" {
  value = azurerm_machine_learning_workspace.ml_workspace.name
}

output "storage_account_name" {
  value = azurerm_storage_account.ml_storage.name
}

output "container_registry_name" {
  value = azurerm_container_registry.ml_registry.name
}

output "application_insights_id" {
  value = azurerm_application_insights.ml_insights.id
}

data "azurerm_client_config" "current" {}

variable "subnet_id" {
  description = "Subnet ID for ML compute"
  type        = string
  default     = ""
}

variable "aks_cluster_id" {
  description = "AKS cluster ID for inference"
  type        = string
  default     = ""
}
