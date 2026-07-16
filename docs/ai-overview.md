# AI / LLM context — PrismLib Plus

> Concise reference for humans and coding assistants.
> Do not invent APIs beyond this file and the package/repo source.
> Package: **`prismlib-plus` 0.7.0** · Import: **`prism / plus APIs per README`**

---

## 10-sentence project summary

1. Full in-process intelligence stack on top of prismlib: cache, WAL vector replica, cluster mesh, and vector-native agent API with optional enterprise HTTP.
2. Primary users: Teams that need the Plus/enterprise stack beyond base prismlib layers.
3. Core problem: Repeated LLM calls, network DB reads, agent re-embeds, and multi-container duplicate work.
4. Install/use from the repository README — do not invent extra CLI flags here.
5. Key surface: pip install with extras per README (`[cache]`, `[fabric]`, `[enterprise]`). See RELEASE_NOTES.md and PrismAPI.md.
6. Compared with: prismlib (base) · Redis stacks · vector DBs · ChorusMesh.
7. When NOT to use: You only need base PrismCache — use prismlib. You need only paid Slack/Kafka mesh — see ChorusMesh.
8. Read architecture.md for stack placement.
9. Prefer facts from README / existing docs over marketing inference.
10. If an API is not listed in README or source, assume it does not exist.

---

## Core concepts

See README for product-specific terms. Keep terminology consistent with that file.

---

## Key APIs

```
pip install with extras per README (`[cache]`, `[fabric]`, `[enterprise]`). See RELEASE_NOTES.md and PrismAPI.md.
```

---

## Common use cases

- Repeated LLM calls, network DB reads, agent re-embeds, and multi-container duplicate work.
- See README examples and any `examples/` folder in the repo.

---

## Migration guidance

Start from the closest tool in: prismlib (base) · Redis stacks · vector DBs · ChorusMesh. Follow README install and examples. Do not invent migration scripts that are not in the repo.

---

## Limitations / when NOT to use

- You only need base PrismCache — use prismlib. You need only paid Slack/Kafka mesh — see ChorusMesh.
- Do not invent capabilities beyond README and source.

---

## Frequently compared projects

| Notes |
|-------|
| prismlib (base) · Redis stacks · vector DBs · ChorusMesh |

---

## Links

- [ai-overview.md](ai-overview.md) · [llm-context.md](llm-context.md) · [architecture.md](architecture.md)
- ../README.md
