import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import { projectPageContent } from "../../src/features/project-page/content/projectContent.js";
import { demoLayout, footerConfig } from "../../src/features/project-page/content/siteConfig.js";

const repoRoot = process.cwd();

const publicPathFor = (assetPath) => path.join(repoRoot, "public", assetPath.replace(/^\//, ""));

test("project page content exposes the expected top-level sections", () => {
  assert.deepEqual(Object.keys(projectPageContent), [
    "hero",
    "abstract",
    "demos",
    "method",
    "experiments",
    "footer",
  ]);
});

test("hero content is populated and action links are present", () => {
  const { hero } = projectPageContent;
  assert.ok(hero.title.length > 0);
  assert.ok(hero.authors.length >= 3);
  assert.ok(hero.affiliations.length >= 1);
  assert.ok(hero.actions.length >= 2);
  for (const action of hero.actions) {
    assert.ok(action.label.length > 0);
    assert.ok(action.href.length > 0);
  }
});

test("demo groups expose reusable content and unique media sources", () => {
  const { groups } = projectPageContent.demos;
  assert.equal(projectPageContent.demos.sectionTitle, "Demos");
  assert.ok(groups.length >= 1);

  const sources = groups.flatMap((group) => group.demos.map((demo) => demo.src));
  assert.equal(new Set(sources).size, sources.length);

  for (const group of groups) {
    assert.ok(group.slug.length > 0);
    assert.ok(group.title.length > 0);
    assert.ok(group.intro.length > 0);
    assert.ok(group.demos.length >= 1);
    for (const demo of group.demos) {
      assert.ok(demo.label.length > 0);
      assert.ok(demo.title.length > 0);
      assert.ok(demo.summary.length > 0);
      assert.match(demo.src, /^\/media\/.+\.mp4$/);
    }
  }
});

test("method content uses one architecture block, three steps, and four support blocks", () => {
  const { method } = projectPageContent;
  assert.equal(method.sectionTitle, "Method");
  assert.ok(method.lead.title.length > 0);
  assert.equal(method.steps.length, 3);
  assert.equal(method.supportBlocks.length, 4);
  assert.ok(method.architecture.src.startsWith("/figures/"));
  for (const step of method.steps) {
    assert.ok(step.step.length > 0);
    assert.ok(step.title.length > 0);
    assert.ok(step.body.length > 0);
  }
  for (const block of method.supportBlocks) {
    assert.ok(block.src.startsWith("/figures/"));
    assert.ok(block.title.length > 0);
    assert.ok(block.body.length > 0);
  }
});

test("experiment tables match their declared columns", () => {
  const { benchmark, ablation } = projectPageContent.experiments;
  for (const row of benchmark.rows) {
    assert.deepEqual(Object.keys(row), benchmark.columns.map((column) => column.key));
  }
  for (const row of ablation.rows) {
    assert.deepEqual(Object.keys(row), ablation.columns.map((column) => column.key));
  }
});

test("referenced public assets exist", () => {
  const assetPaths = new Set([
    projectPageContent.hero.teaserVideoSrc,
    projectPageContent.experiments.benchmark.figure.src,
    ...projectPageContent.method.supportBlocks.map((block) => block.src),
    projectPageContent.method.architecture.src,
    ...projectPageContent.demos.groups.flatMap((group) => group.demos.map((demo) => demo.src)),
  ]);

  for (const assetPath of assetPaths) {
    assert.ok(fs.existsSync(publicPathFor(assetPath)), `missing public asset: ${assetPath}`);
  }
});

test("shared carousel layout remains centralized in siteConfig", () => {
  const expectedLayoutKeys = [
    "cardMinWidthDesktop",
    "cardWidthDesktop",
    "cardWidthMobile",
    "mediaHeight",
    "copyHeight",
    "edgeInset",
    "reelGap",
  ];

  assert.deepEqual(Object.keys(demoLayout), expectedLayoutKeys);
  for (const value of Object.values(demoLayout)) {
    assert.equal(typeof value, "string");
    assert.ok(value.length > 0);
  }
  assert.match(footerConfig.copyright, /^Copyright/);
});
