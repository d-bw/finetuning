### define llm ###

from typing import List, Optional, Mapping, Any
from functools import partial

from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from openai import OpenAI
from os import getenv



class OpenRouterTools(LLM):

    max_length: int = 2048
    temperature: float = 0.1
    top_p: float = 0.7
    history: List = []
    streaming: bool = True
    model: str = None
    client:object =None
    messages:dict=None


           
    @property
    def _llm_type(self) -> str:
        if "gpt-3.5" in self.model:
            return "gpt-3.5-turbo"
        elif "claude-3-haiku" in self.model:
            return "claude-3-haiku"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "history": [],
            "streaming": self.streaming
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        add_history: bool = False
    ) -> str:
        self.client=OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=getenv("OPENROUTER_API_KEY"),
    )
        self.messages={
            "role": "system",
            "content": "You are very powerful assistant, but don't know current events.",
            }
        if self.model is None :
            raise RuntimeError("Must call `load_model()` to load model!")

        if self.streaming:
            text_callback = partial(StreamingStdOutCallbackHandler().on_llm_new_token, verbose=True)
            resp = self.generate_resp(prompt, text_callback, add_history)
        else:
            resp = self.generate_resp(prompt, add_history=add_history)
        

        return resp

    def generate_resp(self, prompt, text_callback=None, add_history=True):
        resp = ""
        index = 0
        completion=None
        user_message=[]
        user_message.append(self.messages)
        user_message.append({
            "role": "user",
            "content": prompt
        })
        if text_callback:
            if self.model=="gpt-3.5-turbo":

                completion = self.client.chat.completions.create(
                        model="openai/gpt-3.5-turbo",
                        messages=user_message,
                        top_p=self.top_p,
                        temperature=self.temperature
                        )
            elif self.model=="claude-3-haiku":
                completion = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=user_message,
                top_p=self.top_p,
                temperature=self.temperature
                )
            text_callback(completion.choices[0].message.content)
            if add_history:
                self.history.append({
                    "role":"assistant",
                    "content":completion.choices[0].message.content
                })

        else:
            if self.model=="gpt-3.5-turbo":
                completion = self.client.chat.completions.create(
                        model="openai/gpt-3.5-turbo",
                        messages=user_message,
                        top_p=self.top_p,
                        temperature=self.temperature
                        )
            elif self.model=="claude-3-haiku":
                completion = self.client.chat.completions.create(
                model="anthropic/claude-3-haiku",
                messages=user_message,
                top_p=self.top_p,
                temperature=self.temperature
                )
            if add_history:
                self.history.append({
                    "role":"assistant",
                    "content":completion.choices[0].message.content
                })
        return completion.choices[0].message.content

    def load_model(self,model_str):
        if self.model is not None :
            return
        print(model_str)
        if "gpt-3.5" in model_str:
            self.model = "gpt-3.5-turbo"
        elif "claude-3-haiku" in model_str:
            self.model = "claude-3-haiku"
