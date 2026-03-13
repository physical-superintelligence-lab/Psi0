# Ψ₀ Website

Astro site for the Ψ₀ project page and demo gallery.

## Stack

- Astro 6
- Static site output for GitHub Pages
- Nix flake for local development
- Plain CSS with content-driven Astro components

## Local development

```bash
nix develop
npm install
npm run dev -- --host
```

Astro usually serves the site at `http://localhost:4321`.

## Validation

```bash
npm test
nix develop --command npm run build
```

## Deployment

The repo is configured for GitHub Pages via `.github/workflows/deploy.yml`.

Required GitHub setting:

- `Settings > Pages > Source = GitHub Actions`

To publish under `https://psi-lab.ai/Psi0`, host this code in the
`physical-superintelligence-lab/Psi0` repository and keep the Astro `base` set to
`/Psi0`.

## Project layout

- `src/pages/index.astro`
  Single route entry point.
- `src/features/project-page/ProjectPage.astro`
  Top-level page composition for the Ψ₀ site.
- `src/features/project-page/components/`
  Section components for hero, demos, method, and experiments.
- `src/features/project-page/content/projectContent.js`
  Single project-content contract for hero text, demos, method blocks, tables, and asset references.
- `src/features/project-page/content/siteConfig.js`
  Shared layout contract and site-level configuration such as demo geometry, footer copy, and TOC entries.
- `src/features/project-page/project-page.css`
  Page-specific layout and component styling.
- `src/styles/theme.css`
  Global theme tokens and base typography.
- `public/figures/`
  Website-ready figure assets derived from the paper.
- `public/media/psi-0/`
  Optimized website demo videos.
- `public/paper/psi0-paper.pdf`
  Paper PDF exposed by the site.
- `scripts/transcode-media.sh`
  ffmpeg-based media optimization helper.
- `tests/project-page/`
  Regression tests for content contracts and feature structure.
- `docs/project-page/architecture.md`
  Editing guide for the landing page feature module.

## Content editing

Most site edits should start in `src/features/project-page/content/projectContent.js`.

That file exports one `projectPageContent` object with six top-level sections:

- `hero`
- `abstract`
- `demos`
- `method`
- `experiments`
- `footer`

Only edit the Astro components when the layout or interaction model needs to change.

## Reusing this for another project

To adapt this site for another research/project page:

1. Replace the values in `src/features/project-page/content/projectContent.js`.
2. Put your project assets under `public/figures/`, `public/media/<your-project>/`, and `public/paper/`.
3. Update shared layout knobs in `src/features/project-page/content/siteConfig.js` only if the default carousel/page geometry does not fit your content.
4. Run `npm test` to catch missing assets and malformed content.
5. Run `nix develop --command npm run build` before shipping.

The demos carousel is intentionally driven by one shared geometry contract. Avoid per-theme layout overrides unless you are willing to update both CSS and the regression tests.

See [docs/project-page/adoption.md](/home/zhenyu/src/psi0.github.io/docs/project-page/adoption.md) for a field-by-field adoption checklist.

## Failure modes we now guard against

- theme-to-theme demo card drift caused by hidden layout overrides
- demo copy overflowing the shared card geometry
- missing website source/footer configuration
- accidental divergence between benchmark data and rendered table columns

## Notes

- The repo may contain large raw paper/media artifacts that are not part of the deployed site.
- Prefer using the optimized assets under `public/` for website work.

## TODO

- [ ] verify deployed path is /psi0
