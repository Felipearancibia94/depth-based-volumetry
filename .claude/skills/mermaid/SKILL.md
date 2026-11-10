---
name: mermaid
description: Create, read, edit, embed and validate Mermaid diagrams (flowchart, sequence, class, state, ER, gitGraph) kept as .mmd sources in docs/diagrams/mermaid and embedded as mermaid blocks in the Markdown that explains them. Use for formal technical diagrams - call sequences, state machines, data flow between modules, schemas, dependencies, branch flow - or when a mermaid block must be understood, changed or checked. For overviews, roadmaps, hardware, photos or free-form spatial layouts use /excalidraw.
---

# Mermaid diagrams

Mermaid is text; GitHub renders a ```` ```mermaid ```` block in Markdown files, issues and
PRs. Same layout as Excalidraw: the source lives with the other diagrams in
`docs/diagrams/mermaid/<name>.mmd`, and the Markdown that explains it holds a generated copy
of the block, kept in sync by `scripts/sync.py` (GitHub cannot include a file by reference).
Official reference: https://mermaid.js.org/syntax/syntax-reference.html. Examples per type
and a syntax cheat sheet: [reference.md](reference.md) (read it when writing a type you have
not used yet; its inline blocks are the one exception to the `.mmd` rule).

## Mermaid or Excalidraw

| Use Mermaid (this skill) | Use `/excalidraw` |
|---|---|
| Flows, call sequences, state machines, schemas, dependencies, git flow | Overviews, roadmaps, hardware and model maps, free-form architecture |
| Must be reviewed in a text diff and edited by anyone | Layout, colour and photos matter; drawn once and polished |
| Source `.mmd`, shown as a block inside the doc that explains it | Source spec, shipped as PNG plus an editable `.excalidraw` |

Use both only when one page needs a visual overview and a precise flow (for example an
Excalidraw overview and a Mermaid sequence of one step). Never draw the same content in both,
and never convert one into the other.

## Create

1. **Pick the type** from what the reader must learn:
   - steps, decisions, data flow, dependencies, system components: `flowchart` (use
     `subgraph` for boundaries; this is the architecture diagram, because `architecture-beta`
     is beta and GitHub's Mermaid version may not support it);
   - who calls whom, in order: `sequenceDiagram`;
   - types and how they relate: `classDiagram`;
   - the fixed states of one thing: `stateDiagram-v2`;
   - tables, columns, keys: `erDiagram`;
   - branches and merges: `gitGraph`.
2. **Take the facts from the code or the doc**, not from memory: read the modules, use their
   exact names for nodes (`EventStore`, not "database layer"), and draw only what exists. Say
   in one sentence under the diagram what is simplified or illustrative.
3. **Write it small.** One idea per diagram, about 15 nodes at most (split otherwise). `LR` for
   pipelines, `TD` for hierarchies. Quote every label that has spaces or punctuation:
   `A["Label (text)"]`. Use `<br/>` for a second line. English text.
4. **Style only when it carries meaning.** Same meaning as the Excalidraw legend: green done,
   orange open, grey later. Define it once with `classDef`, include `color:#1e1e1e` so text
   stays readable in dark mode, and state the meaning in the sentence under the diagram.
5. **Save the source** as `docs/diagrams/mermaid/<name>.mmd` (kebab-case, no fences, no
   front matter needed). It is the only place you edit.
6. **Embed it** in the document that owns the topic, with one sentence before it saying what
   it shows. Add the marker pair (path relative to that Markdown file), then run `sync.py`,
   which fills the block:

   ```text
   <!-- mermaid: diagrams/mermaid/<name>.mmd -->
   <!-- /mermaid -->
   ```

   Embed the same `.mmd` in several documents if needed; do not copy it by hand or repeat its
   content as prose.
7. **Sync and validate** (below) before reporting it done.

## Read and edit an existing diagram

- Read nodes as components or steps, edges as the relation the arrow style defines in that
  diagram (usually solid = direct call, dotted = optional or asynchronous), subgraphs as
  boundaries. Then compare with the code: diagrams go stale, so report any mismatch.
- Edit the `.mmd`, never the block between the markers (it is overwritten). Edit minimally:
  keep node ids, direction and order stable so the diff shows only the change. Do not
  reformat the whole file. Then sync and re-validate.
- A diagram found as a plain block with no markers is hand-written: move it to a `.mmd` and
  add the markers when you are already changing it, not as a separate cleanup.
- When code changes a module, flow or state that a diagram shows, update the diagram in the
  same change.

## Sync and validate

```bash
S=.claude/skills/mermaid/scripts
uv run --no-project python $S/sync.py --all
uv run --no-project python $S/check.py docs/diagrams/mermaid/*.mmd
```

`sync.py --all` rewrites every marked block in `README.md`, `ARCHITECTURE.md`, `ROADMAP.md`
and `docs/` from its source; with `--check` it writes nothing and exits 1 if any block is
stale or a source is missing. It never touches text outside the markers.

`check.py` prints `open: http://127.0.0.1:<port>/`. Open that URL in the built-in browser and
wait about 5 s: the page parses and renders every diagram with a pinned Mermaid (12.2.0,
loaded from cdn.jsdelivr.net, so it needs network) and the command then prints one
`file:line OK` or `file:line ERROR message` per diagram and exits 1 if any failed. It accepts
`.mmd` files and Markdown files (every ```` ```mermaid ```` block). A pass proves the pinned
version accepts the syntax, not that GitHub's version does: to see GitHub's, put a block
containing only `info` in a PR or issue preview. Stay on classic syntax (no `@{ shape: ... }`,
no `architecture-beta`) unless that version is known.

Common breakages (from the official syntax reference):

- the lowercase word `end` as a label or id (use `End` or quote it);
- an id starting with `o` or `x` right after a link (`A---oB` draws a circle edge; add a
  space or capitalise);
- unquoted parentheses, brackets or quotes in a label (quote the label; `#quot;` for `"`);
- `%%` comments must be on their own line and must not contain `{}`;
- sequence diagrams: `;` ends a statement (write `#59;`) and hex colours are not accepted;
- front matter (`---` ... `---`) must start on the very first line of the block;
- misspelled keywords break the diagram, while misspelled config keys are silently ignored.

If the checker cannot run (no network, no browser pane), say the blocks were not validated;
do not claim they render.
