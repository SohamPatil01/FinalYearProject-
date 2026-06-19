# VioLane React frontend (optional)

The production UI is served by FastAPI from `templates/violane.html` with a vanilla JS port of the beams background at `static/js/beams-background.js`.

This folder holds the **shadcn-compatible React** version of the same background for a future Vite/Next migration.

## Why `/components/ui`?

shadcn/ui installs primitives into `components/ui` by convention. Keeping that path makes `npx shadcn@latest add …` work without reconfiguring aliases.

## Setup (new React app)

From the repo root:

```bash
cd frontend
npm install
```

Dependencies include **motion** (Framer Motion v11+), React 19, TypeScript, and Tailwind v4.

### shadcn CLI (if starting fresh elsewhere)

```bash
npx shadcn@latest init
# When prompted:
# - TypeScript: yes
# - Style: New York (or Default)
# - Tailwind CSS: yes
# - Components path: @/components
# - Utils path: @/lib/utils
# - React Server Components: as needed

npm install motion
```

### Use the beams background

```tsx
import { BeamsBackground } from "@/components/ui/beams-background";

export default function Page() {
  return (
    <BeamsBackground intensity="medium">
      {/* your app content */}
    </BeamsBackground>
  );
}
```

Demo: `components/ui/beams-background-demo.tsx`

## Files

| Path | Purpose |
|------|---------|
| `components/ui/beams-background.tsx` | Animated canvas beams (from design spec) |
| `lib/utils.ts` | `cn()` helper for Tailwind class merging |
| `package.json` | `motion`, React, TypeScript, Tailwind |

## FastAPI integration today

No React build is required. Open `http://127.0.0.1:8765` — `violane.html` loads `/static/js/beams-background.js` automatically.
