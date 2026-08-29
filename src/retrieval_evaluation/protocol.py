from typing import Protocol

from .results import RetrievalResult


class RetrieverProtocol(Protocol):
    """
    Common interface required by the retrieval evaluation framework.

    Any retrieval implementation that provides a compatible `retrieve`
    method can be evaluated without inheriting from a specific base class.
    """

    def retrieve(self,query: str,top_k: int = 5,) -> RetrievalResult:
        """
        Retrieve the most relevant pages for a query.

        Parameters
        ----------
        query:
            Natural-language retrieval query.

        top_k:
            Maximum number of pages to return.

        Returns
        -------
        RetrievalResult
            Standardized retrieval output containing ranked pages
            and optional timing information.
        """
        ...