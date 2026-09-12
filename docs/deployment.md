# Kubernetes Deployment

This document provides instructions on how to deploy the OpenAI-Compatible API using Kubernetes.

The production-ready manifests are located in the `deploy/kubernetes` directory and include a Deployment, a Service, and a Horizontal Pod Autoscaler (HPA).

## Prerequisites

- A running Kubernetes cluster.
- `kubectl` configured to interact with your cluster.
- A metrics server installed in the cluster (required for HPA to work).

## Deployment Manifests

The `deploy/kubernetes` directory contains three manifests:

1. **`deployment.yaml`**: Contains the main `Deployment` object.
    - Sets up the `openai-compatible-api` container running on port 8000.
    - Defines a `readinessProbe` checking the `/ready` endpoint to ensure the application only receives traffic when fully loaded.
    - Defines a `livenessProbe` checking the `/healthz` endpoint to restart pods if they become unresponsive.
    - Specifies CPU and Memory `requests` and `limits` to ensure efficient scheduling and resource management.
    - Includes annotations (`prometheus.io/scrape: "true"`, `prometheus.io/path: "/metrics"`, `prometheus.io/port: "8000"`) to allow Prometheus to automatically scrape metrics.
2. **`service.yaml`**: Exposes the `Deployment` on port 8000 via a `ClusterIP` Service.
3. **`hpa.yaml`**: Contains the `HorizontalPodAutoscaler` configuration.
    - Automatically scales the number of pods between 1 and 5 based on target CPU (75%) and memory (80%) utilization.

## Deploying to Kubernetes

To deploy the application to your cluster, apply the manifests using `kubectl`:

```bash
kubectl apply -f deploy/kubernetes/deployment.yaml
kubectl apply -f deploy/kubernetes/service.yaml
kubectl apply -f deploy/kubernetes/hpa.yaml
```

Alternatively, you can apply the entire directory at once:

```bash
kubectl apply -f deploy/kubernetes/
```

## Monitoring

- **Status**: Check the status of your pods to ensure they are running successfully:
  ```bash
  kubectl get pods -l app=openai-compatible-api
  ```
- **Autoscaling**: Verify the Horizontal Pod Autoscaler is correctly fetching metrics:
  ```bash
  kubectl get hpa openai-compatible-api
  ```
  *(Note: It may take a few minutes for the HPA to collect metrics after initial deployment.)*
- **Metrics**: If Prometheus is installed in your cluster and configured to honor scrape annotations, it will automatically begin scraping the `/metrics` endpoint on port 8000 of the deployed pods.
