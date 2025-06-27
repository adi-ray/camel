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

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.toolkits import MCPToolkit
from camel.types import ModelPlatformType, ModelType


async def main():
    # Load config from JSON file
    config_dict = "examples/toolkits/mcp_pptx_html/mcp_config.json"
    async with MCPToolkit(config_path=str(config_dict)) as mcp_toolkit:
        await mcp_toolkit.connect()
        tools = mcp_toolkit.get_tools()
        model = ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
        )
        agent = ChatAgent(
            system_message="""you are a helpful assistant that can convert a html file to pptx file.
            """,
            model=model,
            tools=tools,
        )

        slide_path = os.path.join(
            "examples", "toolkits", "mcp_pptx_html", "slide.html"
        )
        with open(slide_path, 'r') as f:
            slide_html = f.read()
        print(slide_html)
        response = await agent.astep(
            f"convert the following HTML and tailwind css to powerpoint:\n{slide_html}"
        )
        print(response)


import asyncio

asyncio.run(main())
