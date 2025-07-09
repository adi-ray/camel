# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from warnings import warn

from camel.types.enums import JinaReturnFormat

from .base_loader import BaseLoader

JINA_ENDPOINT = "https://r.jina.ai/"


class JinaURLLoader(BaseLoader):
    r"""URL Loader provided by Jina AI. The output is cleaner and more
    LLM-friendly than the URL Reader of UnstructuredIO. Can be configured to
    replace the UnstructuredIO URL Reader in the pipeline.

    Args:
        api_key (Optional[str], optional): The API key for Jina AI. If not
            provided, the reader will have a lower rate limit. Defaults to
            None. (default: :obj:`None`)
        return_format (ReturnFormat, optional): The level of detail
            of the returned content, which is optimized for LLMs. For
            now screenshots are not supported. Defaults to
            ReturnFormat.DEFAULT. (default: :obj:`ReturnFormat.DEFAULT`)
        json_response (bool, optional): Whether to return the response
            in JSON format. Defaults to False. (default: :obj:`False`)
        timeout (int, optional): The maximum time in seconds to wait for
            the page to be rendered. Defaults to 30. (default: :obj:`30`)
        **kwargs (Any): Additional keyword arguments, including proxies,
            cookies, etc. It should align with the HTTP Header field and
            value pairs listed in the reference. (default: :obj:`None`)

    References:
        https://jina.ai/reader
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        return_format: JinaReturnFormat = JinaReturnFormat.DEFAULT,
        json_response: bool = False,
        timeout: int = 30,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        api_key = api_key or os.getenv('JINA_API_KEY')
        if not api_key:
            warn(
                "JINA_API_KEY not set. This will result in a low rate limit "
                "of Jina URL Reader. Get API key here: https://jina.ai/reader."
            )

        # if the following field not provided, it will be None
        api_field = f"Bearer {api_key}" if api_key else None
        json_field = "application/json" if json_response else None

        raw_headers = {
            "Authorization": api_field,
            "X-Return-Format": return_format.value,
            "Accept": json_field,
            "X-Timeout": str(timeout),
            **kwargs,
        }

        # eliminate None values
        self._headers = {k: v for k, v in raw_headers.items() if v}
        self.timeout = timeout

    def _load_single(
        self,
        source: Union[str, Path],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        r"""Load content from a single URL.

        Args:
            source (Union[str, Path]): The URL to load content from. While Path
                objects are accepted for compatibility with BaseLoader, only
                string URLs starting with 'http://' or 'https://' are
                supported.
            **kwargs (Any): Additional arguments to pass to the request.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded content with
                'content' and 'source' keys.

        Raises:
            ValueError: If the source is a Path object, not a valid URL string,
                or if the request fails.
        """
        import requests

        # JinaURLLoader only supports URL strings
        if isinstance(source, Path):
            raise ValueError(
                f"JinaURLLoader only supports URL strings, not file paths. "
                f"Got Path object: {source}"
            )

        if not isinstance(source, str):
            raise ValueError(
                f"Expected string URL, got {type(source).__name__}: {source}"
            )

        if not source.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {source}")

        try:
            response = requests.get(
                JINA_ENDPOINT,
                params={"url": source},
                headers=self._headers,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()
            return {"content": response.text, "source": source}
        except Exception as e:
            raise ValueError(
                f"Failed to read content from {source}: {e}"
            ) from e

    def load(
        self,
        source: Union[str, Path, List[Union[str, Path]]],
        **kwargs: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        r"""Load content from one or more URLs.

        Args:
            source (Union[str, Path, List[Union[str, Path]]]): A single URL or
                a list of URLs to load. While Path objects are accepted for
                compatibility with BaseLoader, only string URLs starting with
                'http://' or 'https://' are supported.
            **kwargs (Any): Additional arguments to pass to the request.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A dictionary with a single key
                "contents" containing a list of loaded data. Each item contains
                'content' and 'source' keys. If a single source is provided,
                the list will contain a single item.

        Raises:
            ValueError: If any source is a Path object, not a valid URL string,
                or if any request fails.
        """
        if isinstance(source, (str, Path)):
            return {"contents": [self._load_single(source, **kwargs)]}
        return {
            "contents": [self._load_single(url, **kwargs) for url in source]
        }

    @property
    def supported_formats(self) -> set[str]:
        r"""Get the set of supported URL schemes.

        Returns:
            set[str]: The set of supported URL schemes.
        """
        return {"http", "https"}
