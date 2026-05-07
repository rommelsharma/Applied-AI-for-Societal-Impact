import fs from "fs";
import path from "path";
import { createRequire } from "module";

const bundledNodeModules =
  process.env.CODEX_NODE_MODULES ||
  "/Users/romsharm/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules";
const require = createRequire(path.join(bundledNodeModules, "index.js"));
const { chromium } = require("playwright");

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
