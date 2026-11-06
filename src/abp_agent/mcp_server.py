"""MCP server: the agent's read tools for any MCP client, at /mcp on the API. It wraps the same
functions the agent calls, so the two can't drift apart. They need no LLM.

Every call needs `Authorization: Bearer <MCP_TOKEN>`. With no token configured nobody gets in. One
shared token suits a few trusted clients; a public server needs OAuth, and that means replacing
only TokenCheck (D31). The SDK's DNS rebinding protection also only accepts localhost.
"""

import functools
import secrets

import anyio.to_thread
from langchain_core.tools import BaseTool
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from abp_agent.agent.tools import TOOLS
from abp_agent.config import get_settings


class TokenCheck(TokenVerifier):
    async def verify_token(self, token: str) -> AccessToken | None:
        expected = get_settings().mcp_token
        if expected and secrets.compare_digest(token.encode(), expected.encode()):
            return AccessToken(token=token, client_id="mcp-client", scopes=[])
        return None


_url = get_settings().mcp_public_url
mcp = FastMCP(
    "abp-agent",
    instructions="ABP Framework docs, C# source and code graph. Results carry [ids]: cite them, "
                 "and answer only from what they say.",
    stateless_http=True,    # each call stands alone: any API worker can serve it
    json_response=True,
    token_verifier=TokenCheck(),
    auth=AuthSettings(issuer_url=_url, resource_server_url=f"{_url}/mcp",
                      validate_token_resource=False),    # TokenCheck is the whole check
)


def with_sources(text: str, citations: list[dict]) -> str:
    """The tool's text, plus where each [id] points: the agent keeps its citations in its state,
    but an MCP client only gets what's in the result."""
    sources = [f"[{c['chunk_id']}] {c.get('url') or ''}".rstrip() for c in citations]
    return "\n\n".join([text, "Sources:\n" + "\n".join(sources)]) if sources else text


def expose(tool: BaseTool) -> None:
    @functools.wraps(tool.func)
    async def call(**kwargs) -> str:
        # The tools block on Postgres and Neo4j; the event loop must not
        result = await anyio.to_thread.run_sync(functools.partial(tool.func, **kwargs))
        return with_sources(*result)

    mcp.add_tool(call, name=tool.name, description=tool.description, structured_output=False,
                 annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))


for _tool in TOOLS:
    expose(_tool)
