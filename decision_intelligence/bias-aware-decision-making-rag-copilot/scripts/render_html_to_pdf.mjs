/**
 * HTML to PDF renderer.
 *
 * Reads the manifest produced by ``generate_book_dossiers.py`` and renders
 * each generated HTML dossier to PDF using a headless Chromium via Playwright.
 *
 * Use this path when high-fidelity PDF output (CSS layout, web fonts, exact
 * pixel rendering) is preferred over the ReportLab path. Otherwise the
 * ReportLab build inside ``generate_book_dossiers.py`` is sufficient.
 */

import fs from "fs";
import path from "path";
import { createRequire } from "module";

// Resolve Playwright from the bundled Codex runtime when available so this
// script works without an explicit ``npm install`` in the project.
const bundledNodeModules =
  process.env.CODEX_NODE_MODULES ||
  "/Users/romsharm/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const require = createRequire(path.join(bundledNodeModules, "index.js"));
const { chromium } = require("playwright");

// Anchor every relative path to the project root, regardless of where the
// script was invoked from.
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const manifestPath = path.join(
  root,
  "data",
  "metadata",
  "book_dossiers",
  "generated_html",
  "manifest.json"
);

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const browser = await chromium.launch({ headless: true });

// One PDF per manifest entry. ``setContent`` avoids a webserver round-trip by
// loading the HTML payload directly into the page.
for (const item of manifest) {
  const page = await browser.newPage();
  const html = fs.readFileSync(item.html_path, "utf8");
  await page.setContent(html, { waitUntil: "load" });
  await page.pdf({
    path: item.pdf_path,
    format: "A4",
    printBackground: true,
    margin: {
      top: "18mm",
      right: "14mm",
      bottom: "18mm",
      left: "14mm"
    }
  });
  await page.close();
}

await browser.close();
