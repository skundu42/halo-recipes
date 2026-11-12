// Renders og/card.html to site/og.png (1200×630) with headless Chrome. The PNG is committed, so the
// Cloudflare build needs no browser; run `npm run og` after changing the card.
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const output = join(root, "site", "og.png");
const CHROMES = [
  process.env.CHROME_PATH,
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
];

const chrome = CHROMES.find((path) => path && existsSync(path));
if (!chrome) throw new Error("No Chrome or Chromium found: set CHROME_PATH to the browser binary");

// A throwaway profile, so the render never touches a running browser's profile.
const profile = mkdtempSync(join(tmpdir(), "og-chrome-"));
try {
  execFileSync(
    chrome,
    [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--force-device-scale-factor=1",
      "--window-size=1200,630",
      "--virtual-time-budget=3000",
      `--user-data-dir=${profile}`,
      `--screenshot=${output}`,
      pathToFileURL(join(root, "og", "card.html")).href,
    ],
    { stdio: ["ignore", "ignore", "inherit"] },
  );
} finally {
  rmSync(profile, { recursive: true, force: true });
}
console.log(`Wrote ${output}`);
