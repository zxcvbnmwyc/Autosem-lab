"""Versioned local retrieval for AutoSEM's bounded one-click editor.

The catalog deliberately contains only trusted, repository-owned capability
cards.  Retrieval helps Qwen understand the product vocabulary, but it never
authorizes execution: ``grounding.parse_one_click_edit_plan`` remains the
server-side allow-list before any SAM2 or image-editing work can begin.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KNOWLEDGE_PATH = Path(__file__).with_name("knowledge") / "one_click_editing.json"
CATALOG_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,47}$")
CARD_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")
MAX_CARD_RULE_CHARS = 480
MAX_ALIAS_CHARS = 80
MAX_ALIASES_PER_CARD = 32
MAX_RETRIEVAL_LIMIT = 6

# These are intentionally hard-coded.  A catalog can describe an existing
# capability or narrow its wording, but it cannot grant a new executable tool.
CORE_CARD_IDS = frozenset(
    {
        "subject.single_visible",
        "policy.non_generative",
        "response.contract",
    }
)
OPERATION_CARD_IDS = frozenset(
    {
        "selection.edge_feather",
        "selection.manual_strokes",
        "background.original",
        "background.transparent",
        "background.color",
        "background.blur",
        "subject.brightness",
        "subject.saturation",
        "subject.blur",
    }
)
ALLOWED_CARD_IDS = CORE_CARD_IDS | OPERATION_CARD_IDS
ALLOWED_SCOPES = frozenset({"automatic", "manual_only", "policy"})


class KnowledgeError(RuntimeError):
    """A repository configuration error that should fail closed at startup."""


@dataclass(frozen=True)
class CapabilityCard:
    """One trusted capability card from the local editing catalog."""

    card_id: str
    aliases: tuple[str, ...]
    rule: str
    scope: str

    def as_prompt_data(self) -> dict[str, str]:
        # Aliases are retrieval implementation detail, not instructions for Qwen.
        return {"id": self.card_id, "scope": self.scope, "rule": self.rule}


@dataclass(frozen=True)
class EditingKnowledge:
    """A validated immutable catalog and deterministic retrieval settings."""

    catalog_version: str
    retrieval_limit: int
    always_include: tuple[CapabilityCard, ...]
    operations: tuple[CapabilityCard, ...]

    @property
    def cards_by_id(self) -> dict[str, CapabilityCard]:
        return {
            card.card_id: card
            for card in (*self.always_include, *self.operations)
        }


@dataclass(frozen=True)
class CapabilityRetrieval:
    """The small trusted context supplied to a single Qwen planning call."""

    catalog_version: str
    cards: tuple[CapabilityCard, ...]
    matched_operation_ids: tuple[str, ...]

    @property
    def available_ids(self) -> frozenset[str]:
        return frozenset(card.card_id for card in self.cards)

    def as_prompt_data(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "retrieved_capabilities": [card.as_prompt_data() for card in self.cards],
        }


def _normalise(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _text(value: Any, field: str, maximum: int, *, minimum: int = 1) -> str:
    if not isinstance(value, str):
        raise KnowledgeError(f"知识库中的 {field} 必须是文字。")
    text = value.strip()
    if not minimum <= len(text) <= maximum or any(ord(char) < 32 for char in text):
        raise KnowledgeError(f"知识库中的 {field} 长度或字符不合法。")
    return text


def _load_card(value: Any, *, expected_ids: frozenset[str], section: str) -> CapabilityCard:
    if not isinstance(value, dict):
        raise KnowledgeError(f"知识库中的 {section} 条目必须是对象。")
    unknown = set(value).difference({"id", "aliases", "rule", "scope"})
    if unknown:
        raise KnowledgeError(f"知识库中的 {section} 条目包含未知字段。")
    card_id = _text(value.get("id"), f"{section}.id", 64, minimum=2)
    if not CARD_ID_RE.fullmatch(card_id) or card_id not in expected_ids:
        raise KnowledgeError(f"知识库中的 {section}.id 不在受支持能力列表中。")
    scope = _text(value.get("scope"), f"{section}.scope", 24)
    if scope not in ALLOWED_SCOPES:
        raise KnowledgeError(f"知识库中的 {section}.scope 不受支持。")
    rule = _text(value.get("rule"), f"{section}.rule", MAX_CARD_RULE_CHARS)
    raw_aliases = value.get("aliases")
    if not isinstance(raw_aliases, list) or len(raw_aliases) > MAX_ALIASES_PER_CARD:
        raise KnowledgeError(f"知识库中的 {section}.aliases 格式不合法。")
    aliases: list[str] = []
    for alias in raw_aliases:
        cleaned = _text(alias, f"{section}.aliases", MAX_ALIAS_CHARS, minimum=2)
        if cleaned not in aliases:
            aliases.append(cleaned)
    if section == "operations" and not aliases:
        raise KnowledgeError("知识库中的操作条目至少需要一个检索别名。")
    return CapabilityCard(card_id, tuple(aliases), rule, scope)


def _load_cards(value: Any, *, expected_ids: frozenset[str], section: str) -> tuple[CapabilityCard, ...]:
    if not isinstance(value, list) or len(value) != len(expected_ids):
        raise KnowledgeError(f"知识库中的 {section} 条目数量不正确。")
    cards = tuple(_load_card(item, expected_ids=expected_ids, section=section) for item in value)
    ids = {card.card_id for card in cards}
    if ids != expected_ids or len(ids) != len(cards):
        raise KnowledgeError(f"知识库中的 {section} 存在缺失或重复能力 ID。")
    return cards


def load_editing_knowledge(path: Path = KNOWLEDGE_PATH) -> EditingKnowledge:
    """Load one catalog and reject malformed or capability-expanding content."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KnowledgeError("无法读取一键剪辑能力知识库。") from error
    if not isinstance(value, dict):
        raise KnowledgeError("一键剪辑能力知识库必须是 JSON 对象。")
    unknown = set(value).difference(
        {"schema_version", "catalog_version", "retrieval_limit", "always_include", "operations"}
    )
    if unknown or value.get("schema_version") != 1:
        raise KnowledgeError("一键剪辑能力知识库版本或字段不受支持。")
    catalog_version = _text(value.get("catalog_version"), "catalog_version", 48)
    if not CATALOG_VERSION_RE.fullmatch(catalog_version):
        raise KnowledgeError("一键剪辑能力知识库版本格式不合法。")
    retrieval_limit = value.get("retrieval_limit")
    if isinstance(retrieval_limit, bool) or not isinstance(retrieval_limit, int):
        raise KnowledgeError("一键剪辑能力知识库检索数量必须是整数。")
    if not 1 <= retrieval_limit <= MAX_RETRIEVAL_LIMIT:
        raise KnowledgeError("一键剪辑能力知识库检索数量超出范围。")
    return EditingKnowledge(
        catalog_version=catalog_version,
        retrieval_limit=retrieval_limit,
        always_include=_load_cards(
            value.get("always_include"), expected_ids=CORE_CARD_IDS, section="always_include"
        ),
        operations=_load_cards(
            value.get("operations"), expected_ids=OPERATION_CARD_IDS, section="operations"
        ),
    )


def retrieve_editing_knowledge(
    instruction: str, knowledge: EditingKnowledge | None = None
) -> CapabilityRetrieval:
    """Return core policies plus the relevant static capability cards.

    This is intentionally lexical instead of embedding-based: the catalog is
    small, fully trusted and needs deterministic behavior that can be tested.
    """
    catalog = knowledge or EDITING_KNOWLEDGE
    query = _normalise(instruction)
    ranked: list[tuple[int, int, CapabilityCard]] = []
    for index, card in enumerate(catalog.operations):
        matches = [alias for alias in card.aliases if _normalise(alias) in query]
        if matches:
            # Longer matching phrases beat generic vocabulary, while catalog
            # order gives stable ties.
            ranked.append((sum(len(_normalise(alias)) for alias in matches), index, card))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = tuple(card for _score, _index, card in ranked[: catalog.retrieval_limit])
    return CapabilityRetrieval(
        catalog_version=catalog.catalog_version,
        cards=(*catalog.always_include, *selected),
        matched_operation_ids=tuple(card.card_id for card in selected),
    )


EDITING_KNOWLEDGE = load_editing_knowledge()
