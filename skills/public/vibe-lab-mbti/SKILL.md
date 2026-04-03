---
name: vibe-lab-mbti
description: Maintain and extend the MBTI assistant in projects/vibe-lab, including the local question bank, scoring logic, rule-based report, AI skill prompts, and /lab/mbti UX. Use when changing MBTI questions, result interpretation, AI readout behavior, or the page flow for the MBTI assistant.
---

# Vibe Lab MBTI

Use this skill when working on the MBTI assistant in `projects/vibe-lab`.

## Open These Files First

- `projects/vibe-lab/src/lib/mbti.ts`
- `projects/vibe-lab/src/lib/ai-skills.ts`
- `projects/vibe-lab/src/components/mbti-assistant.tsx`
- `projects/vibe-lab/src/components/settings-panel.tsx`
- `projects/vibe-lab/src/lib/openai.ts`

## Working Rules

- Keep the separation between local logic and model output clear.
- Treat the local result as a four-axis self-assessment, not an official MBTI instrument.
- Keep the UI copy in Simplified Chinese unless the task explicitly asks for bilingual output.
- Preserve the current pattern: local deterministic result first, optional AI readout second.
- When editing AI prompts, keep them grounded and non-deterministic. Avoid language that turns a personality type into destiny or fixed ability limits.

## Implementation Workflow

1. Update the local source of truth first.
   For question or scoring changes, edit `src/lib/mbti.ts` before touching UI copy.
2. Update AI skills second.
   Keep skill labels, descriptions, headings, and system prompts aligned in `src/lib/ai-skills.ts`.
3. Update the page flow in `src/components/mbti-assistant.tsx`.
   Keep the selected AI skill, current focus, user context, and local report in the model payload.
4. Verify the settings path still works.
   The MBTI page reads the shared model config from `src/components/settings-panel.tsx` and `src/lib/openai.ts`.

## Validation

- Run `npm run build` in `projects/vibe-lab`.
- If the task changes scoring or reporting, manually sanity-check one completed questionnaire path in `/lab/mbti/`.
