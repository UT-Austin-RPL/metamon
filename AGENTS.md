# Metamon — Agent Notes

## Codebase Q&A: use the graphify graph first

This repo has a prebuilt graphify knowledge graph of the `metamon/` package at
`graphify-out/graph.json` (2,840 nodes, 6,741 edges, 151 communities).

**When asked a question about the codebase** (architecture, "how does X work",
"what calls Y", tracing data flow, file relationships, "where is Z defined"),
**run `graphify query "<question>"` before reading files manually.** The graph
is already built — do not rebuild it unless the user explicitly asks
(`--update`, `/graphify <path>`, or "rebuild the graph").

Other useful commands (all read the same `graphify-out/graph.json` by default):
- `graphify path "A" "B"` — shortest path between two concepts
- `graphify explain "NodeName"` — plain-language explanation of a node + neighbors
- `graphify god-nodes` — most connected nodes (architectural hubs)

Outputs to browse:
- `graphify-out/graph.html` — interactive graph, open in a browser
- `graphify-out/GRAPH_REPORT.md` — audit report (god nodes, communities, surprising connections)

If `graphify-out/graph.json` is missing, rebuild with:
`graphify extract metamon --code-only` (no API key needed; code is extracted via AST).
