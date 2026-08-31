# LateAct local site build notes

## Files created

- `index.html` — semantic single-page project site and Open Graph metadata.
- `style.css` — responsive, framework-free academic visual system.
- `SITE_BUILD_NOTES.md` — build, provenance, preview, and deployment record.

The site directly reuses existing files under `showcase/`; frozen scientific artifacts were neither moved nor deleted.

## Site structure

The page order is:

1. hero and local resource links;
2. representative validated Gate 1 demo;
3. three aggregate headline metrics;
4. asynchronous-control problem and serving-policy baselines;
5. Gate 0 denoising-time commitment observation;
6. commitment-aware rollback mechanism;
7. exact rollback-state system result;
8. concise Gate 0/1/2 results;
9. mandatory scope and limitations;
10. citation and resources.

The implementation is plain HTML/CSS: no backend, build system, framework, analytics, external fonts, or external JavaScript.

## Source asset mapping

| page element | source |
|---|---|
| hero/social metadata image | `showcase/og_cover.png` |
| autoplay comparison demo | `showcase/lateact_demo.mp4` |
| video poster | `showcase/demo_thumbnail.png` |
| headline values | `showcase/headline_metrics.json` |
| commitment observation | `showcase/commitment_curve.svg` |
| rollback method | `showcase/lateact_mechanism.svg` |
| checkpoint system result | `showcase/rollback_state.png` |
| claims and limitations | `GATE0_REPORT.md`, `GATE1_REPORT.md`, `GATE2_REPORT.md`, `PHASE3_REPORT.md`, `PHASE4_MINWM_REPORT.md`, and `showcase/PROJECT_PAGE_COPY.md` |

## Tiny asset edits performed

- The commitment-curve subtitle now states that thin lines are 16 validated pairs and the bold blue line is the median trend.
- The mechanism title now reads `LateAct rolls back to the latest pre-commitment state`.
- Both PNG and SVG variants were regenerated from `showcase/build_showcase.py`; no data or mechanism geometry changed.

## Placeholder links

The `Technical Report` button currently links to local `GATE2_REPORT.md`, and `Code` links to local `README.md`. Replace both with their eventual public URLs before public deployment. The demo link is already a valid relative asset link.

`og:image` intentionally uses the relative path `showcase/og_cover.png` because the public domain is not known. After the final Pages URL exists, replace it with the absolute public image URL for the most reliable LinkedIn/Open Graph crawler behavior.

## Local preview

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/`. The entrypoint is `index.html`.

## Later GitHub Pages deployment

Do not perform these steps until public deployment is approved:

1. Replace the local Technical Report and Code links with their final public URLs.
2. Commit `index.html`, `style.css`, `SITE_BUILD_NOTES.md`, the required `showcase/` assets, and any review screenshots intended for the repository.
3. Push the repository to its approved GitHub remote.
4. In repository **Settings → Pages**, select **Deploy from a branch**, choose the approved branch, and use the repository root (`/`) as the publishing folder.
5. Once the Pages domain is known, set `og:image` to the absolute URL for `showcase/og_cover.png`; optionally add an absolute canonical URL.
6. Recheck the public page at desktop and mobile widths, then run the public URL through LinkedIn Post Inspector to refresh the Featured-card preview.

The branch/root publishing procedure above was checked against GitHub's current official documentation: <https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site>.

No repository was created, nothing was pushed, and GitHub Pages was not enabled during this local build.

## Local verification performed

- Started a localhost-only static server at `127.0.0.1:8000` and requested every unique local page dependency.
- Confirmed HTTP 200 responses for the stylesheet, report/code placeholders, MP4, poster, both SVG figures, rollback PNG, and Open Graph image.
- Decoded the complete H.264/yuv420p demo with FFmpeg without errors.
- Checked semantic landmark structure, figure alt text, captions, video accessibility/playback attributes, relative paths, absence of machine-specific paths, CSS brace integrity, reduced-motion handling, and desktop/mobile breakpoints.
- Confirmed the page contains no external scripts, analytics, fonts, or network dependencies.

This server environment has no installed Chromium, Firefox, Playwright, Selenium, or other graphical browser renderer. Pixel-level browser screenshots and a live browser-console inspection could therefore not be produced, so `site_review/` was not created. The responsive structure passed static checks and is ready for human visual review in a local browser.
