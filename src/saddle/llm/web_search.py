"""Web search for the RayXI pipeline.

Gives the pipeline the ability to research game mechanics, systems, properties,
and balance data for any game type — critical for the from-scratch (no template)
pipeline path.

Primary: GLM with native web_search tool (no extra API key needed).
Fallback: Tavily API if TAVILY_API_KEY is set.

The research phase runs BEFORE HLR when no template exists. Results are saved
to a temp game KB so all downstream stages have real data to work from.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

_log = logging.getLogger("saddle.llm.web_search")


async def research_game_concept_glm(
    user_prompt: str,
    genre: str,
    glm_caller,
) -> dict:
    """Research a game concept using GLM's native web search.

    GLM searches the web inline during generation, so we give it one
    comprehensive prompt and let it search + synthesize in a single call.
    """
    research_prompt = (
        f"Research the following game concept thoroughly using web search.\n\n"
        f"## User's Game Concept\n{user_prompt}\n\n"
        f"## Detected Genre: {genre}\n\n"
        f"Search the web for:\n"
        f"1. What runtime systems a {genre} game needs (physics, AI, camera, HUD, "
        f"collision, input, scoring, items, etc.) — be EXHAUSTIVE, list ALL systems\n"
        f"2. Standard entity types and their properties (vehicles, items, projectiles, "
        f"track elements, UI elements, etc.)\n"
        f"3. Typical balance values (speeds, damage ranges, health pools, timers, "
        f"cooldowns, distances)\n"
        f"4. Camera perspective and visual orientation conventions for this genre\n"
        f"5. Common game mechanics, interactions, and item/ability systems\n"
        f"6. Collision detection approaches (what shapes collide with what)\n"
        f"7. HUD elements typically shown to the player\n"
        f"8. AI behavior patterns for CPU opponents\n\n"
        f"Then synthesize ALL your findings into a structured JSON game reference.\n\n"
        f"Output a JSON object with this structure:\n"
        f'{{\n'
        f'  "genre": "{genre}",\n'
        f'  "source": "glm_web_research",\n'
        f'  "core_systems": [\n'
        f'    {{"name": "system_name", "description": "what it does", "properties": [\n'
        f'      {{"name": "prop_name", "type": "int|float|bool|string|Vector2", '
        f'"role": "entity_owner", "typical_range": "min-max", "purpose": "why it exists"}}\n'
        f'    ], "interactions": ["description of key interactions"]}}\n'
        f'  ],\n'
        f'  "entity_roles": [\n'
        f'    {{"name": "role_name", "description": "...", "standard_properties": '
        f'["prop1", "prop2"]}}\n'
        f'  ],\n'
        f'  "balance_reference": {{\n'
        f'    "speed_range": "...", "health_range": "...", "timer_range": "...",\n'
        f'    "notes": "key balance considerations"\n'
        f'  }},\n'
        f'  "camera_and_perspective": {{\n'
        f'    "view_type": "...", "entity_facing": "...", "orientation_notes": "..."\n'
        f'  }},\n'
        f'  "hud_elements": ["element1", "element2"],\n'
        f'  "collision_model": {{\n'
        f'    "shapes": ["entity.shape_name: purpose"],\n'
        f'    "pairs": ["shapeA vs shapeB: response"]\n'
        f'  }},\n'
        f'  "ai_behaviors": ["behavior1", "behavior2"],\n'
        f'  "common_mechanics": ["mechanic1", "mechanic2"],\n'
        f'  "items_and_interactions": [\n'
        f'    {{"name": "...", "trigger": "...", "effect": "...", "properties": []}}\n'
        f'  ]\n'
        f'}}\n\n'
        f"Be THOROUGH — this becomes the ONLY reference for all downstream pipeline "
        f"stages. Missing systems here means missing systems in the final game.\n"
        f"Output ONLY the JSON."
    )

    _log.info("Researching game concept via GLM web search")
    try:
        raw = await glm_caller(
            "You are a game design researcher with web search access. "
            "Search the web thoroughly and synthesize findings into a structured "
            "game knowledge base. Output only JSON.",
            research_prompt,
            json_mode=True,
            label="glm_web_research",
            web_search=True,
        )
        research_kb = json.loads(raw)
        _log.info("GLM web research: %d top-level keys, %d systems found",
                  len(research_kb),
                  len(research_kb.get("core_systems", [])))
        return research_kb
    except Exception as exc:
        _log.warning("GLM web research failed: %s", exc)
        return {}


async def _research_via_llm_knowledge(
    user_prompt: str,
    genre: str,
    caller,
) -> dict:
    """Research using LLM's training knowledge (no web search tool).

    Fallback when web search tool is unavailable. The LLM still has extensive
    game design knowledge from training — just not live web results.
    """
    _log.info("Researching game concept via LLM training knowledge (no web search)")
    # Reuse the same prompt but without web_search=True
    research_prompt = (
        f"You are an expert game designer. Using your knowledge of game design, "
        f"provide a comprehensive analysis of what systems and mechanics are needed "
        f"for the following game concept.\n\n"
        f"## User's Game Concept\n{user_prompt}\n\n"
        f"## Detected Genre: {genre}\n\n"
        f"Identify ALL of these:\n"
        f"1. Every runtime system this game needs (physics, AI, camera, HUD, "
        f"collision, input, scoring, items, etc.) — be EXHAUSTIVE\n"
        f"2. Standard entity types and their properties\n"
        f"3. Typical balance values (speeds, damage, health, timers, cooldowns)\n"
        f"4. Camera perspective and visual orientation conventions\n"
        f"5. Common game mechanics and interactions\n"
        f"6. Collision detection model (what shapes collide with what)\n"
        f"7. HUD elements for the player\n"
        f"8. AI behavior patterns\n\n"
        f"Output a JSON object with this structure:\n"
        f'{{\n'
        f'  "genre": "{genre}",\n'
        f'  "source": "llm_knowledge",\n'
        f'  "core_systems": [\n'
        f'    {{"name": "system_name", "description": "what it does", "properties": [\n'
        f'      {{"name": "prop_name", "type": "int|float|bool|string|Vector2", '
        f'"role": "entity_owner", "typical_range": "min-max", "purpose": "why"}}\n'
        f'    ], "interactions": ["key interactions"]}}\n'
        f'  ],\n'
        f'  "entity_roles": [{{"name": "role", "description": "...", "standard_properties": ["..."]}}\n],\n'
        f'  "balance_reference": {{"speed_range": "...", "health_range": "...", "notes": "..."}},\n'
        f'  "camera_and_perspective": {{"view_type": "...", "entity_facing": "...", "orientation_notes": "..."}},\n'
        f'  "hud_elements": ["element1", "element2"],\n'
        f'  "collision_model": {{"shapes": ["..."], "pairs": ["..."]}},\n'
        f'  "ai_behaviors": ["behavior1", "behavior2"],\n'
        f'  "common_mechanics": ["mechanic1", "mechanic2"],\n'
        f'  "items_and_interactions": [{{"name": "...", "trigger": "...", "effect": "...", "properties": []}}]\n'
        f'}}\n\n'
        f"Be THOROUGH. Output ONLY JSON."
    )
    try:
        raw = await caller(
            "You are a game design expert. Output structured JSON game knowledge bases.",
            research_prompt,
            json_mode=True,
            label="llm_knowledge_research",
        )
        result = json.loads(raw)
        _log.info("LLM knowledge research: %d systems found",
                  len(result.get("core_systems", [])))
        return result
    except Exception as exc:
        _log.warning("LLM knowledge research failed: %s", exc)
        return {}


async def research_game_concept(
    user_prompt: str,
    genre: str,
    caller,
) -> dict:
    """Research a game concept via web search.

    Uses LLM training knowledge as primary (GLM web search disabled —
    billing issues with the OpenAI-compat endpoint).
    Falls back to Tavily if TAVILY_API_KEY is set.

    Args:
        caller: an LLMCaller (from `LLMPool.caller_for("default")` in
            the migrated pipeline path). Used for both knowledge-based
            research and Tavily synthesis. May be None if no LLM is
            available — the function returns {} or raw Tavily results.
    """
    # Primary: LLM training knowledge.  GLM web search disabled —
    # billing issues with OpenAI-compat endpoint.
    if caller is not None:
        result = await _research_via_llm_knowledge(user_prompt, genre, caller)
        if result:
            return result

    # Fallback: Tavily (if API key available)
    import os
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        for candidate in [
            Path.home() / ".config" / "rayxi" / "tavily.key",
            Path.home() / ".tavily_api_key",
        ]:
            if candidate.exists():
                tavily_key = candidate.read_text().strip()
                break

    if not tavily_key:
        _log.info("No web search available (no GLM, no TAVILY_API_KEY)")
        return {}

    return await _tavily_research(user_prompt, genre, tavily_key, caller)


async def _tavily_research(
    user_prompt: str,
    genre: str,
    api_key: str,
    caller,
) -> dict:
    """Fallback: Tavily-based research with LLM synthesis."""
    import httpx

    queries = [
        f"{genre} game core systems and mechanics complete list",
        f"{genre} game entity properties attributes and balance data",
        f"{genre} game camera perspective HUD collision detection",
        f"{genre} game AI behavior items interactions",
    ]

    _log.info("Researching via Tavily: %d queries", len(queries))
    all_results: list[dict] = []
    for query in queries:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": 5,
                        "search_depth": "advanced",
                        "include_answer": True,
                    },
                )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("answer"):
                    all_results.append({"content": data["answer"]})
                for r in data.get("results", []):
                    all_results.append({"content": r.get("content", "")})
        except Exception as exc:
            _log.warning("Tavily search failed for '%s': %s", query[:40], exc)

    if not all_results and caller is None:
        return {}

    if not caller:
        return {"genre": genre, "source": "tavily_raw", "raw_results": all_results[:20]}

    # Synthesize with LLM
    context = json.dumps(all_results, indent=2)[:30000]
    try:
        raw = await caller(
            "Synthesize game research into a structured JSON knowledge base. Output only JSON.",
            f"Genre: {genre}\nUser prompt: {user_prompt}\n\nSearch results:\n{context}",
            json_mode=True,
            label="tavily_synthesis",
        )
        return json.loads(raw)
    except Exception as exc:
        _log.warning("Tavily synthesis failed: %s", exc)
        return {"genre": genre, "source": "tavily_raw", "raw_results": all_results[:20]}


def save_research_kb(kb_dir: Path, game_name: str, research: dict) -> Path:
    """Save research results to a temp game KB file."""
    if not research:
        return kb_dir / "games" / f"{game_name}_research.json"

    games_dir = kb_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)
    path = games_dir / f"{game_name}_research.json"
    path.write_text(json.dumps(research, indent=2, ensure_ascii=False), encoding="utf-8")
    _log.info("Saved research KB: %s (%d bytes)", path, path.stat().st_size)
    return path


def load_research_kb(kb_dir: Path, game_name: str) -> dict:
    """Load previously saved research KB if it exists."""
    path = kb_dir / "games" / f"{game_name}_research.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}
