# AGENTS.md

## Purpose

This repository hosts the Ψ₀ project website built with Astro and deployed as a static site.

## First places to look

- `README.md`
- `docs/project-page/architecture.md`
- `src/features/project-page/content/projectContent.js`
- `src/features/project-page/content/siteConfig.js`
- `src/features/project-page/ProjectPage.astro`
- `src/features/project-page/project-page.css`

## Working model

- Treat `src/features/project-page/content/projectContent.js` as the primary content/config layer.
- Treat `src/features/project-page/content/siteConfig.js` as the shared layout contract layer.
- Treat `src/features/project-page/components/` as section-level rendering only.
- Treat `src/features/project-page/project-page.css` as the page-specific visual system.
- Keep `src/pages/index.astro` minimal.

## Editing guidance

- Prefer changing content in `src/features/project-page/content/projectContent.js` before editing Astro markup.
- Reuse shared demo card styling instead of adding theme-specific one-off rules unless there is a clear structural reason.
- If demo card geometry needs to change, update `src/features/project-page/content/siteConfig.js` first and keep all themes on the same contract.
- Keep the project name consistent as `Ψ₀` in visible UI text.
- If a figure is referenced on the website, prefer website-friendly image assets in `public/figures/` instead of PDF embeds.
- If a video is referenced on the website, use the optimized files in `public/media/psi-0/`.
- Do not reintroduce the old `option-a/option-b/option-c` page structure.

## Validation

Run these after non-trivial changes:

```bash
npm test
nix develop --command npm run build
```

## Deployment notes

- GitHub Pages deployment is handled by `.github/workflows/deploy.yml`.
- The Astro site is configured for static output in `astro.config.mjs`.

## Repo hygiene

- Large raw paper/media artifacts may exist in the repo root for local work; avoid staging them unless the user explicitly wants them versioned.
- Prefer committing website code and optimized public assets only.

## Demo-specific failure modes

The demo carousel has repeatedly failed in a few predictable ways:

- theme-specific card geometry drift
- portrait clips introducing a different visual rhythm from landscape clips
- copy length causing inconsistent card bottoms
- theme switches failing to recenter the active demo

Prefer solving those at the shared contract level rather than by adding more theme-specific CSS.
