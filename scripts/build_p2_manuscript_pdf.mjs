import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const repo = path.resolve(import.meta.dirname, "..");
const runtimeModules = process.env.RUNTIME_NODE_MODULES;
if (!runtimeModules) {
  throw new Error("RUNTIME_NODE_MODULES is required");
}

const { marked } = await import(pathToFileURL(path.join(runtimeModules, "marked", "lib", "marked.esm.js")));
const { chromium } = await import(pathToFileURL(path.join(runtimeModules, "playwright", "index.mjs")));

const sourcePath = path.join(repo, "docs", "P2_FINAL_MANUSCRIPT.md");
const outputPath = path.join(repo, "deliverables", "22097191_ZHANG_YUE_P2_Research_Project.pdf");
const qaDir = path.join(repo, "outputs", "docx_qa", "p2_submission_final");
const htmlPath = path.join(qaDir, "22097191_ZHANG_YUE_P2_Research_Project.html");

fs.mkdirSync(qaDir, { recursive: true });

const markdown = fs.readFileSync(sourcePath, "utf8");
const titleMatch = markdown.match(/^#\s+(.+)$/m);
if (!titleMatch) throw new Error("Manuscript title was not found");

const title = titleMatch[1].trim();
const bodyStart = markdown.indexOf("## Abstract");
if (bodyStart < 0) throw new Error("Abstract heading was not found");

const bodyMarkdown = markdown.slice(bodyStart);
const headings = [...bodyMarkdown.matchAll(/^(#{1,3})\s+(.+)$/gm)].map((match) => ({
  level: match[1].length,
  text: match[2].trim(),
}));

const slugCounts = new Map();
const slugger = (value) => {
  const base = value
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-") || "section";
  const count = slugCounts.get(base) || 0;
  slugCounts.set(base, count + 1);
  return count === 0 ? base : `${base}-${count + 1}`;
};

const renderer = new marked.Renderer();
renderer.heading = ({ tokens, depth }) => {
  const text = marked.Parser.parseInline(tokens);
  const plain = text.replace(/<[^>]+>/g, "");
  return `<h${depth} id="${slugger(plain)}">${text}</h${depth}>`;
};
marked.setOptions({ gfm: true, renderer });

const bodyHtml = marked.parse(bodyMarkdown);
slugCounts.clear();
const toc = headings
  .filter((heading) => heading.level <= 2)
  .map((heading) => `<li class="toc-l${heading.level}"><a href="#${slugger(heading.text)}">${heading.text}</a></li>`)
  .join("\n");

const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>${title}</title>
<style>
  @page { size: A4; margin: 19mm 18mm 18mm 22mm; }
  * { box-sizing: border-box; }
  html { font-family: "Times New Roman", Georgia, serif; color: #111827; }
  body { margin: 0; font-size: 10.5pt; line-height: 1.46; }
  .cover { height: 255mm; display: flex; flex-direction: column; justify-content: space-between; text-align: center; page-break-after: always; }
  .cover-kicker { margin-top: 18mm; font: 700 12pt Arial, sans-serif; color: #1f4e79; letter-spacing: 0.8px; }
  .cover h1 { margin: 22mm auto 12mm; max-width: 155mm; font-size: 23pt; line-height: 1.25; color: #17365d; }
  .cover-rule { width: 50mm; height: 2px; margin: 0 auto 14mm; background: #c55a11; }
  .cover-meta { width: 125mm; margin: 0 auto; border-collapse: collapse; text-align: left; }
  .cover-meta th, .cover-meta td { border: 0; padding: 2.3mm 1mm; font-size: 11pt; }
  .cover-meta th { width: 37mm; font-family: Arial, sans-serif; color: #374151; }
  .cover-note { font: 9pt Arial, sans-serif; color: #4b5563; }
  .toc { page-break-after: always; }
  .toc h1 { color: #17365d; border-bottom: 2px solid #c55a11; padding-bottom: 3mm; }
  .toc ul { list-style: none; padding: 0; margin: 8mm 0 0; }
  .toc li { border-bottom: 1px dotted #d1d5db; }
  .toc a { display: block; padding: 2.1mm 0; color: #111827; text-decoration: none; }
  .toc-l1 { font-weight: 700; margin-top: 2mm; }
  .toc-l2 { padding-left: 7mm; font-size: 9.5pt; }
  article > h1 { page-break-before: always; margin-top: 0; padding-bottom: 3mm; border-bottom: 2px solid #c55a11; color: #17365d; font-size: 18pt; }
  article > h1:first-child { page-break-before: auto; }
  h2 { margin: 6mm 0 2mm; color: #1f4e79; font-size: 13pt; page-break-after: avoid; }
  h3 { margin: 4mm 0 1.5mm; color: #374151; font-size: 11pt; page-break-after: avoid; }
  p { margin: 0 0 3.2mm; text-align: justify; orphans: 3; widows: 3; }
  ul, ol { margin: 1mm 0 3mm 6mm; padding-left: 6mm; }
  li { margin-bottom: 1mm; }
  blockquote { margin: 4mm 8mm; padding: 3mm 5mm; border-left: 3px solid #c55a11; background: #f6f8fa; font-style: italic; }
  table { width: 100%; margin: 4mm 0 5mm; border-collapse: collapse; font-size: 8.7pt; page-break-inside: avoid; }
  th { background: #17365d; color: white; font-family: Arial, sans-serif; }
  th, td { border: 1px solid #9ca3af; padding: 1.8mm 2mm; vertical-align: top; }
  tr:nth-child(even) td { background: #f4f6f8; }
  code { font: 8.5pt Consolas, monospace; overflow-wrap: anywhere; }
  strong { color: #111827; }
  a { color: #1f4e79; }
</style>
</head>
<body>
  <section class="cover">
    <div>
      <div class="cover-kicker">WQF7023 ARTIFICIAL INTELLIGENCE RESEARCH PROJECT</div>
      <h1>${title}</h1>
      <div class="cover-rule"></div>
      <table class="cover-meta">
        <tr><th>Name</th><td>ZHANG YUE</td></tr>
        <tr><th>Matric No.</th><td>22097191</td></tr>
        <tr><th>Programme</th><td>Master of Artificial Intelligence</td></tr>
        <tr><th>Supervisor</th><td>Dr. Uzair Ishtiaq</td></tr>
        <tr><th>Submission</th><td>Final Research Manuscript</td></tr>
        <tr><th>Date</th><td>16 August 2026</td></tr>
      </table>
    </div>
    <div class="cover-note">Automated results frozen. Human evaluation was not conducted; no human or clinical validation is claimed.</div>
  </section>
  <section class="toc">
    <h1>Table of Contents</h1>
    <ul>${toc}</ul>
  </section>
  <article>${bodyHtml}</article>
</body>
</html>`;

fs.writeFileSync(htmlPath, html, "utf8");

const browser = await chromium.launch({
  executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  headless: true,
});
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "networkidle" });
  await page.pdf({
    path: outputPath,
    format: "A4",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    headerTemplate: "<div></div>",
    footerTemplate: '<div style="width:100%;font:8px Arial;color:#6b7280;text-align:center"><span class="pageNumber"></span></div>',
    margin: { top: "19mm", right: "18mm", bottom: "18mm", left: "22mm" },
  });
} finally {
  await browser.close();
}

console.log(`Created ${path.relative(repo, outputPath)}`);
