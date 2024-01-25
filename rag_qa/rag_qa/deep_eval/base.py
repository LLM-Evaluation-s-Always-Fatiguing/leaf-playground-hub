import os
from typing import Literal, Optional, TypeVar, Generic, Union
from pydantic import Field
from leaf_playground.eval_tools.base import EvalToolConfig
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from abc import ABC


class DeepEvalOpenAIModelConfig(EvalToolConfig):
    model: Literal[
        "gpt-4-1106-preview",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ] = Field(json_schema_extra={"default": "gpt-4-1106-preview"})
    api_key: Optional[str] = Field(default=None)


class DeepEvalAzureOpenAIModelConfig(EvalToolConfig):
    endpoint: str
    api_key: str
    deployment_name: str
    deployment_model_version: str
    api_version: str


class DeepEvalEvalToolConfig(EvalToolConfig):
    backend_config: Union[DeepEvalOpenAIModelConfig, DeepEvalAzureOpenAIModelConfig] = Field(union_mode="smart")


C = TypeVar('C', bound=DeepEvalEvalToolConfig)


class DeepEvalBasicEvalTool(ABC, Generic[C]):

    def _create_model_based_on_config(self, config: C):
        if isinstance(config.backend_config, DeepEvalOpenAIModelConfig):
            return ChatOpenAI(
                model_name=config.backend_config.model,
                openai_api_key=os.environ.get("OPENAI_API_KEY", config.backend_config.api_key)
            )
        elif isinstance(config.backend_config, DeepEvalAzureOpenAIModelConfig):
            return AzureChatOpenAI(
                openai_api_version=config.backend_config.api_version,
                azure_deployment=config.backend_config.deployment_name,
                azure_endpoint=config.backend_config.endpoint,
                openai_api_key=config.backend_config.api_key,
                model_version=config.backend_config.deployment_model_version,
            )
        else:
            raise ValueError("Unsupported configuration type")


__all__ = ["DeepEvalEvalToolConfig", "DeepEvalBasicEvalTool"]
