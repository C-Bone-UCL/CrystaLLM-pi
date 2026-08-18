# Service

The FastAPI service wraps the command-line utilities. Long-running work is dispatched as a
background job and polled through the jobs endpoints rather than held open on the request.

The live endpoint documentation, with every field description rendered, is served by the running
container at `/docs`. This page lists the request models behind it.

::: _api

## Job status

::: _api.JobStatus

## Endpoint registration

One module per endpoint group. Each registers its routes onto the root app.

::: _utils._api.generation.register_generation_routes
::: _utils._api.training.register_training_routes
::: _utils._api.metrics.register_metrics_routes
::: _utils._api.preprocessing.register_preprocessing_routes
::: _utils._api.virtualiser.register_virtualiser_routes
::: _utils._api.jobs.register_jobs_routes

## Generation requests

::: _utils._api.generation.DirectGenerationRequest
::: _utils._api.generation.MakePromptsRequest
::: _utils._api.generation.GenerateCIFsRequest
::: _utils._api.generation.EvaluateCIFsRequest
::: _utils._api.generation.PostprocessRequest

## Training requests

::: _utils._api.training.TrainRequest

## Metrics requests

::: _utils._api.metrics.VUNMetricsRequest
::: _utils._api.metrics.EHullMetricsRequest
::: _utils._api.metrics.XRDMetricsRequest
::: _utils._api.metrics.PropertyMetricsRequest

## Preprocessing requests

::: _utils._api.preprocessing.DeduplicateRequest
::: _utils._api.preprocessing.CleaningRequest
::: _utils._api.preprocessing.SaveDatasetRequest
::: _utils._api.preprocessing.XRDPreprocessRequest
::: _utils._api.preprocessing.CalcTheorXRDRequest
::: _utils._api.preprocessing.CifsZipToParquetRequest

## Virtualiser requests

::: _utils._api.virtualiser.VirtualiseRequest
