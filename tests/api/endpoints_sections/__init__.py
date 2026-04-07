"""Sectioned API endpoint tests."""
from .root import RootEndpointTests
from .jobs import JobManagementTests
from .preprocessing import PreprocessingEndpointTests
from .training import TrainingEndpointTests
from .generation import GenerationEndpointTests
from .metrics import MetricsEndpointTests
from .command_construction import CommandConstructionTests
from .api_gaps import APIGapTests
from .pipeline import PipelineIntegrationTests
from .virtualiser import VirtualiserEndpointTests

__all__ = [
    "RootEndpointTests",
    "JobManagementTests",
    "PreprocessingEndpointTests",
    "TrainingEndpointTests",
    "GenerationEndpointTests",
    "MetricsEndpointTests",
    "CommandConstructionTests",
    "APIGapTests",
    "PipelineIntegrationTests",
    "VirtualiserEndpointTests",
]
