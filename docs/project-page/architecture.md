# Project Page Structure

The website content for the Ψ₀ landing page is intentionally split into three layers:

1. `src/features/project-page/content/projectContent.js`
   This is the single source of truth for the page content contract. It exports one `projectPageContent` object with section-level data for `hero`, `abstract`, `demos`, `method`, `experiments`, and `footer`.

2. `src/features/project-page/content/siteConfig.js`
   This holds shared layout and site-level configuration such as demo carousel geometry and footer copy. Use this when the page needs to adapt to another project without rewriting component internals.

3. `src/features/project-page/components/*.astro`
   Each major page section renders from the shared data module. Future edits should prefer updating data first, then section markup only if the layout must change.

4. `src/features/project-page/project-page.css`
   Shared styling for the landing page. Keep section-specific class names stable so future agents can modify one section without reworking the whole stylesheet.

## Editing guidance

- Prefer editing values inside `projectPageContent` before changing any Astro markup.
- Replace project-specific strings and asset paths inside `projectPageContent.hero`, `projectPageContent.abstract`, `projectPageContent.demos`, `projectPageContent.method`, and `projectPageContent.experiments`.
- Update shared carousel sizing in `src/features/project-page/content/siteConfig.js`, not by adding per-theme geometry overrides.
- Each demo group should keep a stable `slug` for selection state, but demo card geometry should remain shared across themes.
- Keep reusable layout and footer settings in `siteConfig.js`; keep project-specific text and asset references in `projectContent.js`.
- Use `public/figures/`, `public/media/...`, and `public/paper/...` for website assets referenced by the content contract.

## Component map

- `HeroSection.astro`: renders `projectPageContent.hero`.
- `AbstractSection.astro`: renders `projectPageContent.abstract`.
- `DemosSection.astro`: renders `projectPageContent.demos`.
- `MethodSection.astro`: renders `projectPageContent.method`.
- `ExperimentsSection.astro`: renders `projectPageContent.experiments`.
- `ProjectPage.astro`: top-level composition plus wide-screen table of contents and footer.

## Validation

- Run `npm test` for content-shape checks and public-asset existence checks.
- Run `nix develop --command npm run build` to verify the full Astro build.
- See `docs/project-page/adoption.md` for the reuse workflow aimed at other researchers.
