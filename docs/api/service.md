# Service

The FastAPI service exposes the preprocessing, generation, training, evaluation, and virtualisation workflows as HTTP endpoints.

Long-running operations run as background jobs. Clients submit a request and poll the job endpoints for its status and result.

The running service provides complete request and response documentation at `/docs`. Request models that add semantics beyond their corresponding CLI are documented here. Request bodies that mirror CLI arguments are documented by the CLI entry points.

::: _api

## Endpoints

::: _api.root
::: _api.healthz

## Job status

::: _api.JobStatus

## Generation requests

::: _utils._api.generation.DirectGenerationRequest
::: _utils._api.generation.MakePromptsRequest
::: _utils._api.generation.GenerateCIFsRequest

## Training requests

::: _utils._api.training.TrainRequest

## Metrics requests

::: _utils._api.metrics.VUNMetricsRequest
::: _utils._api.metrics.EHullMetricsRequest
::: _utils._api.metrics.XRDMetricsRequest
::: _utils._api.metrics.PropertyMetricsRequest

## Virtualiser requests

::: _utils._api.virtualiser.VirtualiseRequest