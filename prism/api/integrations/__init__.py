"""
prism.api.integrations — Framework adapters for PrismAPI
=========================================================

Available integrations:

    from prism.api.integrations.langgraph import (
        PrismRetrieverNode,
        MultiProviderRetrieverNode,
        create_retriever_node,
    )

    from prism.api.integrations.langchain import PrismRetriever

Each integration is an optional import — the framework dependency is not
required at install time.  Missing framework raises ImportError with a
clear install instruction.
"""

from prism.api.integrations.langgraph import (
    MultiProviderRetrieverNode,
    PrismRetrieverNode,
    create_retriever_node,
)

__all__ = [
    "PrismRetrieverNode",
    "MultiProviderRetrieverNode",
    "create_retriever_node",
]
