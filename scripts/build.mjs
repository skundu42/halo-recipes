// Builds the static site into dist/: the recipe index, the raw recipe files at the indexed commit, and
// content-hashed assets, so everything but index.html can be cached as immutable.
import { createHash } from "node:crypto";
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { extractRecipes } from "./extract.mjs";
import { resolveSource } from "./source.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const site = join(root, "site");
const dist = join(root, "dist");
const modules = join(root, "node_modules");

const FONTS = [
  "@fontsource/newsreader/files/newsreader-latin-400-normal.woff2",
  "@fontsource/hanken-grotesk/files/hanken-grotesk-latin-400-normal.woff2",
  "@fontsource/hanken-grotesk/files/hanken-grotesk-latin-500-normal.woff2",
  "@fontsource/hanken-grotesk/files/hanken-grotesk-latin-600-normal.woff2",
];

const digest = (content) => createHash("sha256").update(content).digest("hex").slice(0, 10);

function write(relPath, content) {
  const out = join(dist, relPath);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, content);
  return relPath;
}

function writeHashed(dir, name, content) {
  const dot = name.lastIndexOf(".");
  return write(`${dir}/${name.slice(0, dot)}.${digest(content)}${name.slice(dot)}`, content);
}

function replaceOnce(text, from, to, file) {
  if (!text.includes(from)) throw new Error(`${file} no longer contains "${from}"; update build.mjs`);
  return text.replace(from, to);
}

const source = resolveSource(process.env, join(root, ".cache"));
const { recipes, methods, families, warnings } = extractRecipes(source.dir, source);

rmSync(dist, { recursive: true, force: true });

const short = source.sha.slice(0, 12);
for (const recipe of recipes) {
  recipe.raw = write(`raw/${short}/${recipe.path}`, readFileSync(join(source.dir, recipe.path)));
}

const index = {
  source: { repo: source.repo, ref: source.ref, sha: source.sha, url: source.url },
  builtAt: new Date().toISOString(),
  methods,
  families,
  recipes,
};
const dataPath = writeHashed("data", "recipes.json", JSON.stringify(index));

let css = readFileSync(join(site, "styles.css"), "utf8");
for (const font of FONTS) {
  const file = font.split("/").at(-1);
  const hashed = writeHashed("assets", file, readFileSync(join(modules, font)));
  css = replaceOnce(css, `url("fonts/${file}")`, `url("${hashed.replace(/^assets\//, "")}")`, "site/styles.css");
}
// The SIL Open Font License travels with the fonts it covers.
for (const family of new Set(FONTS.map((font) => font.split("/")[1]))) {
  write(`assets/OFL-${family}.txt`, readFileSync(join(modules, "@fontsource", family, "LICENSE")));
}

let html = readFileSync(join(site, "index.html"), "utf8");
html = replaceOnce(html, 'href="styles.css"', `href="${writeHashed("assets", "styles.css", css)}"`, "site/index.html");
html = replaceOnce(html, 'src="app.js"', `src="${writeHashed("assets", "app.js", readFileSync(join(site, "app.js")))}"`, "site/index.html");
html = replaceOnce(html, 'content="data/recipes.json"', `content="${dataPath}"`, "site/index.html");
write("index.html", html);

for (const file of ["_headers", "favicon.svg", "robots.txt"]) copyFileSync(join(site, file), join(dist, file));

console.log(`Indexed ${recipes.length} recipes from ${source.repo}@${short} (${source.ref}) into dist/`);
for (const warning of [...source.warnings, ...warnings]) console.warn(`  warning: ${warning}`);
