/**
 * Shared site-level configuration.
 *
 * Keep layout contracts here instead of scattering magic numbers across CSS
 * and components. This makes it easier to adapt the project page for other
 * projects while preserving known-good geometry.
 */

export const demoLayout = {
  cardMinWidthDesktop: "640px",
  cardWidthDesktop: "96%",
  cardWidthMobile: "min(88vw, 520px)",
  mediaHeight: "clamp(320px, 42vw, 460px)",
  copyHeight: "120px",
  edgeInset: "max(4vw, 28px)",
  reelGap: "18px",
};

export const footerConfig = {
  copyright: "Copyright © 2026 PSI Lab. All rights reserved.",
};

export const tableOfContents = [
  { id: "hero", label: "Overview" },
  { id: "abstract", label: "Abstract" },
  { id: "demos", label: "Demos" },
  { id: "method", label: "Method" },
  { id: "experiments", label: "Experiments" },
];
