"""Generic NIM chat + tool-calling loop with Pydantic validation."""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .client import get_nim_client, get_nim_model, is_nim_available

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

ToolHandler = Callable[[Dict[str, Any]], Any]


def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract first JSON object from model text (handles markdown fences)."""
    if not text:
        raise ValueError("Empty model response")
    cleaned = text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        # drop first fence and optional last fence
        if lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith('```'):
            lines = lines[:-1]
        cleaned = '\n'.join(lines).strip()
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start < 0 or end < 0 or end <= start:
        raise ValueError("No JSON object found in response")
    return json.loads(cleaned[start : end + 1])


def chat_structured(
    *,
    system: str,
    user: str,
    schema: Type[T],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    retry: bool = True,
) -> Optional[T]:
    """Single-shot structured JSON completion validated by Pydantic."""
    if not is_nim_available():
        return None
    client = get_nim_client()
    model = get_nim_model()
    messages = [
        {'role': 'system', 'content': system},
        {
            'role': 'user',
            'content': (
                f"{user}\n\n"
                "Responda APENAS com um único objeto JSON válido "
                f"compatível com este schema: {schema.model_json_schema()}"
            ),
        },
    ]

    def _call() -> T:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content or ''
        data = _extract_json_object(content)
        return schema.model_validate(data)

    try:
        return _call()
    except (ValidationError, ValueError, json.JSONDecodeError, Exception) as exc:
        logger.warning("NIM structured chat failed: %s", exc)
        if not retry:
            return None
        try:
            messages.append(
                {
                    'role': 'user',
                    'content': (
                        f"A resposta anterior era inválida ({exc}). "
                        "Retorne somente JSON válido conforme o schema."
                    ),
                }
            )
            return _call()
        except Exception as exc2:
            logger.warning("NIM structured chat retry failed: %s", exc2)
            return None


def tool_calling_loop(
    *,
    system: str,
    user: str,
    tools: List[Dict[str, Any]],
    tool_handlers: Dict[str, ToolHandler],
    schema: Type[T],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    max_rounds: int = 3,
) -> Optional[T]:
    """
    Light agentic loop: model may call tools up to max_rounds, then must
    return final JSON matching schema.
    """
    if not is_nim_available():
        return None
    client = get_nim_client()
    model = get_nim_model()
    messages: List[Dict[str, Any]] = [
        {'role': 'system', 'content': system},
        {
            'role': 'user',
            'content': (
                f"{user}\n\n"
                "Quando tiver informação suficiente, responda APENAS com JSON "
                f"válido para: {schema.model_json_schema()}\n"
                "Todo campo textual voltado ao usuário deve estar em português (pt-BR)."
            ),
        },
    ]

    try:
        for _ in range(max_rounds):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools or None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            msg = resp.choices[0].message
            tool_calls = getattr(msg, 'tool_calls', None) or []
            if tool_calls:
                messages.append(
                    {
                        'role': 'assistant',
                        'content': msg.content or '',
                        'tool_calls': [
                            {
                                'id': tc.id,
                                'type': 'function',
                                'function': {
                                    'name': tc.function.name,
                                    'arguments': tc.function.arguments,
                                },
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or '{}')
                    except json.JSONDecodeError:
                        args = {}
                    handler = tool_handlers.get(name)
                    if handler is None:
                        result = {'error': f'Unknown tool: {name}'}
                    else:
                        try:
                            result = handler(args)
                        except Exception as tool_exc:
                            result = {'error': str(tool_exc)}
                    messages.append(
                        {
                            'role': 'tool',
                            'tool_call_id': tc.id,
                            'content': json.dumps(result, default=str),
                        }
                    )
                continue

            content = msg.content or ''
            try:
                data = _extract_json_object(content)
                return schema.model_validate(data)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                messages.append(
                    {
                        'role': 'user',
                        'content': (
                            f"JSON inválido ({exc}). "
                            "Retorne somente o objeto JSON final, "
                            "com rationale e textos em português (pt-BR)."
                        ),
                    }
                )

        # Final forced attempt without tools
        return chat_structured(
            system=system,
            user=user + "\n\nContexto das tools já foi coletado. Emita o JSON final.",
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
            retry=True,
        )
    except Exception as exc:
        logger.warning("NIM tool loop failed: %s", exc)
        return None
