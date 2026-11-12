// Serves dist/ on localhost for a quick look. `npx wrangler dev` previews it exactly as Cloudflare
// serves it, _headers included.
import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const dist = resolve(dirname(fileURLToPath(import.meta.url)), "..", "dist");
const port = Number(process.env.PORT || 8788);
const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".woff2": "font/woff2",
  ".yaml": "text/plain; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

createServer(async (req, res) => {
  let path = normalize(join(dist, decodeURIComponent(new URL(req.url, "http://localhost").pathname)));
  if (path !== dist && !path.startsWith(dist + sep)) return res.writeHead(403).end();
  try {
    if ((await stat(path)).isDirectory()) path = join(path, "index.html");
    const body = await readFile(path);
    res.writeHead(200, { "content-type": TYPES[extname(path)] ?? "application/octet-stream" }).end(body);
  } catch {
    res.writeHead(404, { "content-type": "text/plain" }).end("Not found");
  }
}).listen(port, "127.0.0.1", () => console.log(`Serving dist/ at http://127.0.0.1:${port}`));
