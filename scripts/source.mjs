// Resolves the Halo checkout a build indexes: a local directory (HALO_REPO_DIR) or a shallow clone of a GitHub ref.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, rmSync } from "node:fs";
import { join, resolve } from "node:path";

const git = (cwd, ...args) =>
  execFileSync("git", ["-c", "advice.detachedHead=false", ...args], { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "inherit"] }).trim();

export function resolveSource(env, cacheDir) {
  const repo = env.HALO_GITHUB_REPO || "whitecircle/halo";
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) throw new Error(`HALO_GITHUB_REPO must look like owner/name, got "${repo}"`);

  let dir;
  const warnings = [];
  if (env.HALO_REPO_DIR) {
    dir = resolve(env.HALO_REPO_DIR);
    if (!existsSync(join(dir, "examples"))) throw new Error(`HALO_REPO_DIR=${dir} has no examples/ directory: point it at a Halo checkout`);
    if (git(dir, "status", "--porcelain")) warnings.push(`${dir} has uncommitted changes: GitHub links point at HEAD, which does not include them`);
  } else {
    // A branch or tag; `git clone --branch` does not take a bare commit SHA.
    const ref = env.HALO_REF || "main";
    dir = join(cacheDir, "halo");
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(cacheDir, { recursive: true });
    git(cacheDir, "clone", "--quiet", "--depth", "1", "--branch", ref, `https://github.com/${repo}.git`, dir);
  }

  const sha = git(dir, "rev-parse", "HEAD");
  return {
    dir,
    repo,
    ref: env.HALO_REF || (env.HALO_REPO_DIR ? git(dir, "rev-parse", "--abbrev-ref", "HEAD") : "main"),
    sha,
    url: `https://github.com/${repo}`,
    githubBlob: (path) => `https://github.com/${repo}/blob/${sha}/${path}`,
    warnings,
  };
}
