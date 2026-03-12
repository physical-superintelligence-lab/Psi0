import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const repoRoot = process.cwd();

const requiredPaths = [
  "src/features/project-page/ProjectPage.astro",
  "src/features/project-page/components/AbstractSection.astro",
  "src/features/project-page/components/DemosSection.astro",
  "src/features/project-page/components/ExperimentsSection.astro",
  "src/features/project-page/components/HeroSection.astro",
  "src/features/project-page/components/MethodSection.astro",
  "src/features/project-page/content/projectContent.js",
  "src/features/project-page/content/siteConfig.js",
  "src/features/project-page/project-page.css",
  "docs/project-page/architecture.md",
  "docs/project-page/adoption.md",
  "tests/project-page/content.test.mjs",
];

const retiredPaths = [
  "src/components/ProjectPage.astro",
  "src/components/project/ResourcesSection.astro",
  "src/data/projectContent.js",
  "src/data/siteConfig.js",
  "src/styles/project-page.css",
  "tests/content.test.mjs",
  "docs/project-page.md",
];

test("project page feature module files exist in the expected locations", () => {
  for (const relativePath of requiredPaths) {
    assert.ok(
      fs.existsSync(path.join(repoRoot, relativePath)),
      `missing expected file: ${relativePath}`,
    );
  }
});

test("legacy project page paths stay retired after the reorganization", () => {
  for (const relativePath of retiredPaths) {
    assert.ok(
      !fs.existsSync(path.join(repoRoot, relativePath)),
      `legacy path should not exist: ${relativePath}`,
    );
  }
});
