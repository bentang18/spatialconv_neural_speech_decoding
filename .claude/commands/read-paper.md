Read and analyze the paper at $ARGUMENTS for relevance to the v12 Neural Field Perceiver architecture.

## Paper retrieval (use in order of preference)

1. **Local PDF** — if the argument is a file path, use `Read` directly
2. **arxiv MCP** — if the paper is on arxiv or bioRxiv, use `mcp__arxiv__*` tools to search/fetch. Prefer this over WebFetch for academic papers
3. **exa MCP** — if you need to find the paper (given only a title or author), use `mcp__exa__*` for semantic search. Better than WebSearch for academic content
4. **WebFetch** — fallback for direct URLs that aren't on arxiv/bioRxiv

Read ALL pages/sections, not just the abstract. For PDFs >10 pages, read in chunks.

## Steps

1. **Read the full paper** using the retrieval priority above.

2. **Create a pastwork summary** at `pastwork/summaries/<first_author><year>_<short_name>.md` following the established format. Include:
   - Full citation with DOI/URL
   - Architecture details (exact dimensions, layer counts, training hyperparameters)
   - Key results with specific numbers
   - What we can reuse vs what doesn't apply to our uECOG regime
   - Regime comparison table (their data vs ours)
   - Common mistakes to correct (things the abstract oversimplifies)

3. **Assess v12 relevance** — score each aspect:
   - Per-patient layers: how do they handle cross-subject variation?
   - Spatial processing: grid-based, coordinate-based, or neither?
   - Temporal processing: masking, autoregressive, or other?
   - SSL approach: what objective, what scale?
   - Training protocol: joint, pretrain+finetune, or other?

4. **Update project docs** if the paper is Tier 1-3 relevant:
   - Add to `docs/reading_list.md` at appropriate tier
   - Add to `docs/research_synthesis.md` paper table + any new findings
   - Update `CLAUDE.md` if it changes established findings

5. **Report** key takeaways: what's novel, what confirms v12 choices, what challenges our assumptions.
