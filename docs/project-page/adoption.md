# Adopting This Project Page

This page is designed so another research team can reuse it by editing one
content file and replacing public assets.

## What to edit

Edit `src/features/project-page/content/projectContent.js`.

That file exports one `projectPageContent` object with these sections:

- `hero`
  Use this for the project title, eyebrow link, authors, affiliations, teaser video, and top-level action links.
- `abstract`
  Use this for the abstract kicker/title/body.
- `demos`
  Use this for themed demo groups, demo labels, titles, summaries, and video asset paths.
- `method`
  Use this for the lead copy, architecture figure, method steps, and the four support blocks.
- `experiments`
  Use this for benchmark copy, figure metadata, benchmark table columns/rows, and ablation columns/rows.
- `footer`
  Use this for the website source label/link.

## What assets to replace

Put your project assets in `public/` and update the matching paths in
`projectPageContent`.

Typical replacements:

- figures: `public/figures/`
- demo videos: `public/media/<project-slug>/`
- paper PDF: `public/paper/`
- teaser video: `public/media/<project-slug>/`

The tests will fail if referenced public assets do not exist.

## What usually does not need edits

You should not normally need to edit:

- `src/features/project-page/components/*.astro`
- `src/features/project-page/project-page.css`
- `src/pages/index.astro`

Only change those when you want a different layout or interaction model, not
just different project content.

## Shared layout knobs

Edit `src/features/project-page/content/siteConfig.js` if you need to adjust:

- demo card width
- demo media height
- demo copy height
- reel edge inset
- reel gap
- footer copyright
- table-of-contents labels

Keep demo geometry shared across all demo groups. Do not add per-group or
per-demo layout overrides unless you also update tests and CSS intentionally.

## Validation workflow

Run these after content or asset changes:

```bash
npm test
nix develop --command npm run build
```

`npm test` checks:

- content contract shape
- benchmark/ablation column consistency
- presence of referenced public assets
- shared carousel layout contract

## Practical adoption checklist

1. Replace `projectPageContent.hero`.
2. Replace `projectPageContent.abstract`.
3. Replace `projectPageContent.demos.groups` and copy your videos into `public/media/...`.
4. Replace `projectPageContent.method`.
5. Replace `projectPageContent.experiments`.
6. Replace the paper/figure/teaser assets in `public/`.
7. Run `npm test`.
8. Run `nix develop --command npm run build`.
