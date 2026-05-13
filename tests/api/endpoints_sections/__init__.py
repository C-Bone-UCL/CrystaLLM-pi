from tests.api.endpoints_sections.root import RootEndpointTests
from tests.api.endpoints_sections.jobs import JobManagementTests
from tests.api.endpoints_sections.preprocessing import PreprocessingEndpointTests
from tests.api.endpoints_sections.training import TrainingEndpointTests
from tests.api.endpoints_sections.generation import GenerationEndpointTests
from tests.api.endpoints_sections.metrics import MetricsEndpointTests
from tests.api.endpoints_sections.command_construction import CommandConstructionTests
from tests.api.endpoints_sections.api_gaps import APIGapTests
from tests.api.endpoints_sections.pipeline import PipelineIntegrationTests
from tests.api.endpoints_sections.virtualiser import VirtualiserEndpointTests

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
