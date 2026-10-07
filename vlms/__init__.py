"""The model this project asks for motions: one client, three servers, four documented traps."""

from .endpoints import (CONTEXT_TOKENS, IMAGE_SIDES, PROMPT_SHAPES,  # noqa: F401
                        ROLES, client_for, image_side_for, image_tokens, url_for)
from .qwen import QwenClient, VlmReply  # noqa: F401
