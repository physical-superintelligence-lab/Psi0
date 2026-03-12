# Paper Copy README

This note tracks where the project website intentionally differs from the current LaTeX paper, even after aligning the website copy to use paper language as much as possible.

## Goal

The website should prefer paper wording when it reads clearly in a landing-page context.
Differences are allowed when the paper is too dense, too equation-heavy, or too repetitive for the webpage.

## Current differences

- The Method section opens with a dedicated two-image data-source block for `EgoDex` and `Humanoid Everyday`.
  The paper discusses these datasets in the pre-training subsection, but it does not present them as a standalone visual pair at the top of the method narrative.

- The website compresses the pre-training subsection.
  The paper includes details about the shared task-space action representation, FAST tokenization, next-step action prediction, and tokenizer reconstruction loss.
  The website omits those details in the main narrative to keep the page readable.

- The website training cards use shorter headings than the paper subsection titles.
  The card titles are written to scan well in a three-card layout, while the card bodies stay close to the paper text.

- The website architecture summary omits some implementation specifics from the paper.
  The paper names `Qwen3-VL-2B-Instruct`, the MM-DiT parameter count, and the `AMO` controller.
  The website keeps the higher-level architectural description and leaves those lower-level specifics out of the main summary block.

- The website teleoperation block is positioned after architecture and staged training.
  The paper presents teleoperation as its own later subsection as well, but the website placement is chosen to support the landing-page narrative flow.

- The website experiments summary is shortened.
  The paper includes stronger quantitative detail such as the `40%` gap over the second-best baseline and the discussion of why the training recipe works.
  The website summary keeps the main result and the claim about data efficiency, but compresses the supporting analysis.

## Editing guidance

- When updating website copy from the paper, prefer copying the paper text first and simplifying only where the webpage becomes awkward.
- If a website phrase is more interpretive than the paper, document it here.
- If the paper changes dataset statistics, training-stage wording, or benchmark claims, update `src/features/project-page/content/projectContent.js` first and then revise this note if needed.
