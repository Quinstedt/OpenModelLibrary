#Needs to be updated


from os import PathLike
from typing import Union, List, Dict
from transformers import Conversation


class ModelLoadingException(Exception):
    """
    An exception raised if trying to create a Session using a model that's being loaded.
    """
    ...


class UnsupportedModelException(Exception):
    """
    An exception raised if trying to load a model that REST-at doesn't support.
    """
    ...


class Model:
    """
    A class abstracting pretrained models, providing functions to load, retrieve, and prompt a model.

    Properties:
    -----------
    max_new_tokens: int
        The maximum number of new tokens the model will generate.
    
    context_window: int
        The total token limit that the model's context can handle.

    Methods:
    --------
    get(model_name_or_path: Union[str, PathLike], max_new_tokens: int, temperature: float = 0.1, context_window: int = 4096) -> Union[Model, None]:
        Loads a given model if not already loaded and returns the model. Returns None if the model is being loaded.

    generate_response_dynamic_max_tokens(history: List[Dict[str, str]], prompt: str) -> str:
        Prompts the model and returns the response, dynamically adjusting the number of tokens based on the history.

    generate_response_fixed_max_tokens(history: List[Dict[str, str]], prompt: str) -> str:
        Prompts the model and returns the response with a fixed number of max tokens.

    count_tokens(history: List[Dict[str, str]], prompt: str) -> int:
        Returns the number of tokens used in a given history and prompt.
    """

    @staticmethod
    def get(
        model_name_or_path: Union[str, PathLike],
        max_new_tokens: int,
        temperature: float = 0.1,
        context_window: int = 4096
    ) -> Union[Model, None]:
        """
        Loads a specified model if not already loaded.

        Parameters:
        -----------
        model_name_or_path: Union[str, PathLike]
            The model to get. Can be either a model name from Hugging Face Hub or a path to a local model.
        
        max_new_tokens: int
            The max_new_tokens parameter used when generating. Required when loading.
        
        temperature: float
            The temperature for randomness in text generation.
        
        context_window: int
            The context window size.

        Returns:
        --------
        Union[Model, None]
            The model that was loaded. None if the specified model is being loaded.

        Raises:
        -------
        ValueError
            When trying to load without max_new_tokens.
        """
        ...

    def generate_response_dynamic_max_tokens(
        self,
        history: List[Dict[str, str]],
        prompt: str
    ) -> str:
        """
        Generates a response using dynamic max tokens based on the context window and history size.

        Parameters:
        -----------
        history: List[Dict[str, str]]
            The conversation history.
        
        prompt: str
            The prompt to send to the model.

        Returns:
        --------
        str
            The response from the model.
        """
        ...

    def generate_response_fixed_max_tokens(
        self,
        history: List[Dict[str, str]],
        prompt: str
    ) -> str:
        """
        Generates a response using fixed max tokens.

        Parameters:
        -----------
        history: List[Dict[str, str]]
            The conversation history.
        
        prompt: str
            The prompt to send to the model.

        Returns:
        --------
        str
            The response from the model.
        """
        ...

    def count_tokens(
        self,
        history: List[Dict[str, str]],
        prompt: str
    ) -> int:
        """
        Counts the number of tokens used in the conversation history plus the current prompt.

        Parameters:
        -----------
        history: List[Dict[str, str]]
            The conversation history.
        
        prompt: str
            The prompt to count tokens for.

        Returns:
        --------
        int
            The number of tokens used.
        """
        ...

    @property
    def max_new_tokens(self) -> int:
        """
        Getter for max_new_tokens.

        Returns:
        --------
        int
            The maximum number of new tokens the model will generate.
        """
        ...

    @max_new_tokens.setter
    def max_new_tokens(self, value: int) -> None:
        """
        Setter for max_new_tokens. Ensures that the value is positive.

        Parameters:
        -----------
        value: int
            The new value for max_new_tokens.

        Raises:
        -------
        ValueError
            If value is not positive.
        """
        ...

    @property
    def context_window(self) -> int:
        """
        Getter for the context window size.

        Returns:
        --------
        int
            The context window size of the model.
        """
        ...

    @context_window.setter
    def context_window(self, value: int) -> None:
        """
        Setter for context_window. Ensures that the value is positive.

        Parameters:
        -----------
        value: int
            The new value for context_window.

        Raises:
        -------
        ValueError
            If value is not positive.
        """
        ...


class Session:
    """
    A class that maintains a conversation session with a model, preserving the conversation history and system prompt.

    Properties:
    -----------
    name: str
        The name of the session.
    
    temperature: float
        The temperature used for text generation randomness.

    system_prompt: str
        The system prompt for the session.

    history: List[Dict[str, str]]
        A copy of the message history of the session.

    Methods:
    --------
    create(name: str, model_name_or_path: Union[str, PathLike], max_new_tokens: int, system_prompt: Optional[str] = None, temperature: float = 0.1) -> Session:
        Creates a new session or retrieves an existing one with the given name.

    get(name: str) -> Optional[Session]:
        Retrieves an existing session by its name.

    prompt(prompt: str) -> str:
        Prompts the model and returns the response. Removes the last user and assistant message from the history.

    prompt_with_history(prompt: str) -> str:
        Prompts the model and returns the response, keeping the full conversation history intact.

    clear() -> None:
        Clears the conversation history, keeping only the system prompt.

    delete() -> None:
        Deletes the session from the session cache.
    """

    def __init__(
        self,
        name: str,
        model_name_or_path: Union[str, PathLike],
        max_new_tokens: int,
        system_prompt: str,
        temperature: float = 0.1
    ) -> None:
        """
        Initializes a new Session with a given model and system prompt.

        Parameters:
        -----------
        name: str
            The name of the session.
        
        model_name_or_path: Union[str, PathLike]
            The path or name of the model to be used.
        
        max_new_tokens: int
            The maximum number of new tokens the model can generate.
        
        system_prompt: str
            The system prompt for the session.

        temperature: float
            The temperature for text generation randomness.
        """
        ...

    @staticmethod
    def create(
        name: str,
        model_name_or_path: Union[str, PathLike],
        max_new_tokens: int,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> Session:
        """
        Creates a new session or retrieves an existing one with the given name.

        Parameters:
        -----------
        name: str
            The name of the session.
        
        model_name_or_path: Union[str, PathLike]
            The path or name of the model to be used.

        max_new_tokens: int
            The maximum number of new tokens the model can generate.

        system_prompt: Optional[str]
            The system prompt for the session. Defaults to the model's default system prompt.

        temperature: float
            The temperature for text generation randomness.

        Returns:
        --------
        Session
            A new or existing session.
        """
        ...

    @staticmethod
    def get(name: str) -> Optional[Session]:
        """
        Retrieves an existing session by its name.

        Parameters:
        -----------
        name: str
            The name of the session.

        Returns:
        --------
        Optional[Session]
            The session if found, otherwise None.
        """
        ...

    @property
    def system_prompt(self) -> str:
        """
        Getter for the session's system prompt.

        Returns:
        --------
        str
            The system prompt used in the session.
        """
        ...

    @system_prompt.setter
    def system_prompt(self, system_prompt: str) -> None:
        """
        Setter for the session's system prompt. Updates the system prompt in the session history.

        Parameters:
        -----------
        system_prompt: str
            The new system prompt to be set for the session.
        """
        ...

    def prompt(self, prompt: str) -> str:
        """
        Prompts the model and returns the response. Removes the last user and assistant message from the history.

        Parameters:
        -----------
        prompt: str
            The user prompt to send to the model.

        Returns:
        --------
        str
            The response from the model.
        """
        ...

    def prompt_with_history(self, prompt: str) -> str:
        """
        Prompts the model and returns the response, keeping the full conversation history intact.

        Parameters:
        -----------
        prompt: str
            The user prompt to send to the model.

        Returns:
        --------
        str
            The response from the model.
        """
        ...

    def clear(self) -> None:
        """
        Clears the conversation history, keeping only the system prompt.
        """
        ...

    @property
    def history(self) -> List[Dict[str, str]]:
        """
        Getter for a copy of the message history in the session.

        Returns:
        --------
        List[Dict[str, str]]
            A deep copy of the conversation history.
        """
        ...

    def delete(self) -> None:
        """
        Deletes the session from the session cache.
        """
        ...
