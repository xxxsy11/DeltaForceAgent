"""
LLM 调用工具（纯 LangChain v1 Runnable）
"""

from typing import Any


def extract_text_content(content: Any) -> str:
    """将 LLM 返回内容统一提取为纯文本。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    return str(content)


def invoke_llm_text(
    llm_client: Any,
    prompt: str,
    model: str = "",
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    """
    统一调用 LLM 并返回文本。
    仅支持 LangChain v1（bind + invoke）。
    """
    _ = model  # 兼容旧调用签名；纯 LangChain 路径不使用 model 参数
    if hasattr(llm_client, "bind") and hasattr(llm_client, "invoke"):
        runnable = llm_client.bind(temperature=temperature, max_tokens=max_tokens)
        response = runnable.invoke(prompt)
        return extract_text_content(getattr(response, "content", response)).strip()
    raise TypeError("纯 LangChain 模式下 llm_client 必须支持 bind/invoke")
