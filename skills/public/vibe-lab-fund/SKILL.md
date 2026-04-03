---
name: vibe-lab-fund
description: Maintain and extend the fund assistant in projects/vibe-lab, including holdings input, Eastmoney data loading, notice and earnings signal display, AI fund skills, and /lab/fund UX. Use when changing portfolio logic, data fetching, AI prompt behavior, or notice-driven analysis in the fund assistant.
---

# Vibe Lab Fund

Use this skill when working on the fund assistant in `projects/vibe-lab`.

## Open These Files First

- `projects/vibe-lab/src/components/fund-assistant.tsx`
- `projects/vibe-lab/src/lib/ai-skills.ts`
- `projects/vibe-lab/src/lib/finance/browser.ts`
- `projects/vibe-lab/src/lib/finance/rules.ts`
- `projects/vibe-lab/src/generated/market-brief.json`
- `projects/vibe-lab/src/lib/openai.ts`

## Working Rules

- Preserve the current structure: holdings are user-entered, fund snapshots and notices are fetched, and AI sits on top as an optional layer.
- Keep all guidance risk-aware. Do not imply guaranteed returns or pretend the app has tick-level real-time coverage.
- Treat notices and earnings signals as clues to verify, not final conclusions.
- Keep the UI in Simplified Chinese and keep AI output structured with explicit headings.
- When changing AI prompts, keep the selected skill, portfolio brief, holdings, snapshots, notices, and market brief in the payload.

## Implementation Workflow

1. Start from the user action path in `src/components/fund-assistant.tsx`.
2. If the issue is data quality or missing notice content, inspect `src/lib/finance/browser.ts`.
3. If the issue is local interpretation, update `src/lib/finance/rules.ts`.
4. If the issue is model behavior, update `src/lib/ai-skills.ts` and keep the output headings stable.
5. If the issue is market brief generation, inspect the generated JSON and the build scripts before changing UI copy.

## Validation

- Run `npm run build` in `projects/vibe-lab`.
- Manually test `/lab/fund/` for:
  - add holding
  - select holding
  - refresh snapshot
  - open notices
  - generate AI analysis with a configured API key
