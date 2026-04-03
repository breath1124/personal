---
name: vibe-lab-oracle
description: Maintain and extend the oracle lab in projects/vibe-lab, including the local bazi and ziwei-inspired rules engine, chart rendering, AI fortune-reading skills, and /lab/oracle UX. Use when changing the oracle input model, chart logic, AI readout behavior, or the presentation of the oracle lab.
---

# Vibe Lab Oracle

Use this skill when working on the oracle lab in `projects/vibe-lab`.

## Open These Files First

- `projects/vibe-lab/src/lib/oracle.ts`
- `projects/vibe-lab/src/lib/ai-skills.ts`
- `projects/vibe-lab/src/components/oracle-studio.tsx`
- `projects/vibe-lab/src/lib/openai.ts`
- `projects/vibe-lab/src/components/settings-panel.tsx`

## Working Rules

- Treat the local oracle engine as an experimental interpretation layer, not as canonical traditional metaphysics software.
- Preserve the split between local deterministic chart generation and optional AI commentary.
- Keep the model grounded: no prophecy tone, no certainty claims, no “fated” language.
- Keep AI output actionable. Translate chart language into observation points, boundaries, and experiments.
- Keep the UI in Simplified Chinese and make uncertainty visible when the birth time is approximate or unknown.

## Implementation Workflow

1. Change chart mechanics in `src/lib/oracle.ts` first.
2. Change model skills in `src/lib/ai-skills.ts` second.
3. Change page flow and payload wiring in `src/components/oracle-studio.tsx`.
4. Keep the AI payload anchored in local output.
   Include the draft input, local chart result, and the user question instead of asking the model to invent a chart.
5. Re-check the shared settings path if model calls or errors change.

## Validation

- Run `npm run build` in `projects/vibe-lab`.
- Manually test `/lab/oracle/` for:
  - generate local chart
  - switch AI skills
  - generate AI readout with a configured API key
  - reset form state
