from __future__ import annotations
from enum import Enum
from os import PathLike
import copy
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    MistralForCausalLM,
    MixtralForCausalLM,
    LlamaForCausalLM
)
from typing import Final

class TokenError(Exception):
    """Base class for token-related exceptions in the model."""
    pass

class NegativeTokenCountError(TokenError):
    """Raised when the token count becomes negative."""
    def __init__(self, message="Negative token count: context window exceeded."):
        super().__init__(message)

class InsufficientAllowedTokensError(TokenError):
    """Raised when there are insufficient tokens for response generation."""
    def __init__(self, allowed_tokens: int):
        message = f"Insufficient allowed tokens. Minimum required is 1000. Allowed tokens: {allowed_tokens}."
        super().__init__(message)

class ModelLoadingException(Exception):
    def __init__(self, model_name_or_path: str | PathLike):
        super().__init__(f"{model_name_or_path} is being loaded.")

class UnsupportedModelException(Exception):
    def __init__(self, model_type: str):
        super().__init__(f"{model_type} is not supported right now.")

class _ModelType(Enum):
    MISTRAL = {
        "inst": ("[INST]", "[/INST]"),
        "bos": "<s>",
        "eos": "</s>"
    }
    LLAMA = {
        "inst": ("user", "assistant"),
        "bos": "",
        "eos": ""
    }

class Model:
    _PLACEHOLDER: Model = None
    _SYSTEM_PROMPT: Final[str] = "You are a helpful AI assistant."
    _MODELS: Final[dict[str | PathLike, Model]] = {}

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = None,
        model: PreTrainedModel = None,
        max_new_tokens: int = 2048,
        context_window: int = 4096,
        temperature: float = 0.1
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.max_new_tokens = max_new_tokens
        self._context_window = context_window
        self.temperature = temperature

        if isinstance(model, (MistralForCausalLM, MixtralForCausalLM)):
            self.type = _ModelType.MISTRAL
        elif isinstance(model, LlamaForCausalLM):
            self.type = _ModelType.LLAMA
        elif tokenizer is None and model is None:
            pass  # Placeholder for later model loading
        else:
            raise UnsupportedModelException(type(model).__name__)

    @staticmethod
    def _get_placeholder() -> Model:
        """Returns a placeholder model."""
        if not Model._PLACEHOLDER:
            Model._PLACEHOLDER = Model()
        return Model._PLACEHOLDER

    @staticmethod
    def _get(model_name_or_path: str | PathLike) -> Model | None:
        """Retrieves a model if already loaded, otherwise returns None."""
        return Model._MODELS.get(model_name_or_path)

    @staticmethod
    def get(
        model_name_or_path: str | PathLike,
        max_new_tokens: int = None,
        context_window: int = 4096,
        temperature: float = 0.1
    ) -> Model | None:
        model = Model._get(model_name_or_path)

        if model is Model._get_placeholder():
            return None
        if model:
            return model

        if max_new_tokens is None:
            raise ValueError("Cannot load model without max_new_tokens.")

        Model._MODELS[model_name_or_path] = Model._get_placeholder()

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        model.eval()

        loaded_model = Model(tokenizer, model, max_new_tokens, context_window, temperature)
        Model._MODELS[model_name_or_path] = loaded_model

        return loaded_model

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> BatchEncoding:
        """Applies a chat template based on the model type and tokenizes the messages."""
        if self.type == _ModelType.MISTRAL:
            return self._apply_chat_template_mistral(messages)
        elif self.type == _ModelType.LLAMA:
            return self._apply_chat_template_llama(messages)

    def _apply_chat_template_llama(self, messages: list[dict[str, str]]) -> BatchEncoding:
        """Applies the Llama chat template and tokenizes the messages."""
        for message in messages:
            message["content"] = message["content"].strip()
            if not message["content"]:
                raise ValueError("Messages can't be empty!")

        return self.tokenizer(
            self.tokenizer.apply_chat_template(messages, tokenize=False),
            return_tensors="pt",
            return_attention_mask=True
        )

    def _apply_chat_template_mistral(self, messages: list[dict[str, str]]) -> BatchEncoding:
        """Applies the Mistral chat template and tokenizes the messages."""
        chat = self.tokenizer.bos_token
        sys_message = Model._SYSTEM_PROMPT

        if messages[0]["role"] == "system":
            sys_message = messages.pop(0)["content"]

        for i, message in enumerate(messages):
            if (message["role"] == "user") != (i % 2 == 0):
                raise Exception("Conversation roles must alternate user/assistant.")

            message["content"] = message["content"].strip()
            if not message["content"]:
                raise ValueError("Messages can't be empty!")

            if message["role"] == "user":
                chat += f"[INST]\n<system>\n{sys_message}\n</system>\n\n{message['content']}\n[/INST]"
            elif message["role"] == "assistant":
                chat += f"{message['content']}{self.tokenizer.eos_token}"
            else:
                raise Exception("Invalid role!")

        return self.tokenizer(chat, return_tensors="pt", return_attention_mask=True)

    def get_response(
        self,
        history: list[dict[str, str]],
        prompt: str,
        dynamic_max_tokens: bool = False
    ) -> str:
        """Generates a response based on the conversation history and prompt."""
        history.append({"role": "user", "content": prompt})

        input_ids = self._apply_chat_template(history).to("cuda")
        total_input_tokens = input_ids.input_ids.size(1)

        if dynamic_max_tokens:
            max_allowed_tokens = self.context_window - total_input_tokens - 100
            if max_allowed_tokens < 0:
                raise NegativeTokenCountError()
            elif max_allowed_tokens < 700:
                raise InsufficientAllowedTokensError(max_allowed_tokens)

            dynamic_max_new_tokens = max(0, max_allowed_tokens)
        else:
            dynamic_max_new_tokens = self.max_new_tokens

        outputs = self.model.generate(
            **input_ids,
            max_new_tokens=dynamic_max_new_tokens,
            do_sample=True,
            temperature=self.temperature
        )

        raw_res = self.tokenizer.decode(outputs[0])
        inst_suffix = self.type.value["inst"][1]
        res = raw_res[
            (raw_res.rfind(inst_suffix) + len(inst_suffix)):raw_res.rfind(self.tokenizer.eos_token)
        ].strip()

        history.append({"role": "assistant", "content": res})
        return res

    def count_tokens(self, history: list[dict[str, str]], prompt: str) -> int:
        """Counts the number of tokens in the conversation history and prompt."""
        history_copy = copy.deepcopy(history)
        history_copy.append({"role": "user", "content": prompt})
        tokenized_input = self._apply_chat_template(history_copy)
        return tokenized_input.input_ids.size(1)

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int) -> None:
        if value <= 0:
            raise ValueError("max_new_tokens must be positive.")
        self._max_new_tokens = value

    @property
    def context_window(self) -> int:
        return self._context_window

    @context_window.setter
    def context_window(self, value: int) -> None:
        if value <= 0:
            raise ValueError("context_window must be positive.")
        self._context_window = value


class Session:
    _SESSIONS: dict[str, Session] = {}

    def __init__(
        self,
        name: str,
        model_name_or_path: str | PathLike,
        max_new_tokens: int,
        system_prompt: str,
        temperature: float = 0.1
    ):
        self.name = name
        self.temperature = temperature
        self.model = Model.get(model_name_or_path, max_new_tokens, temperature=self.temperature)

        if not self.model:
            raise ModelLoadingException(model_name_or_path)

        self._system_prompt = system_prompt
        self._history = [{"role": "system", "content": system_prompt}]

    @staticmethod
    def create(
        name: str,
        model_name_or_path: str | PathLike,
        max_new_tokens: int,
        system_prompt: str = Model._SYSTEM_PROMPT,
        temperature: float = 0.1
    ) -> Session:
        session = Session.get(name)
        if session:
            return session

        session = Session(name, model_name_or_path, max_new_tokens, system_prompt, temperature)
        Session._SESSIONS[name] = session
        return session

    @staticmethod
    def get(name: str) -> Session | None:
        return Session._SESSIONS.get(name)

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, system_prompt: str) -> None:
        if not system_prompt:
            system_prompt = Model._SYSTEM_PROMPT
        self._system_prompt = system_prompt
        self._history[0] = {"role": "system", "content": self._system_prompt}

    def prompt(self, prompt: str) -> str:
        return self.model.generate_response(self._history, prompt, dynamic_max_tokens=True)

    def prompt_with_history(self, prompt: str) -> str:
        return self.model.generate_response(self._history, prompt, dynamic_max_tokens=True)

    def clear(self) -> None:
        self._history = [{"role": "system", "content": self._system_prompt}]

    @property
    def history(self) -> list[dict[str, str]]:
        return copy.deepcopy(self._history)

    def delete(self) -> None:
        del Session._SESSIONS[self.name]
