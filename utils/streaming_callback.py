from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from queue import SimpleQueue
from typing import Any, Dict, List, Union

class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler - works with LLMs that support streaming."""

    def __init__(self, q: SimpleQueue, job_done):
        self.q = q
        self.job_done = job_done

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        while not self.q.empty():
            try:
                self.q.get(block=False)
            except SimpleQueue.empty:
                continue

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        print("----new token:", token)
        self.q.put(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        self.q.put(self.job_done)

    # def on_llm_error(
    #     self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    # ) -> None:
    #     """Run when LLM errors."""
    #     print("----- LLM errors!!!")
    #     self.q.put(self.job_done)
