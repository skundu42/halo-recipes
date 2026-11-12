# Halo recipe explorer

A searchable index of every training recipe in [Halo](https://github.com/whitecircle/halo). Filter by method,
model family, tuning, parallelism and the GPUs you have, then copy the launch command. It's a static site
built for Cloudflare Pages.

This is a community project and is not affiliated with White Circle. Where a recipe disagrees with Halo's
supported matrix, the matrix wins.

## How the index is built

`npm run build` reads a Halo checkout and writes a static site to `dist/`. Nothing is copied by hand; every
field comes from the repo at build time:

| Field | Source |
| --- | --- |
| Model, dataset, tuning, context, parallelism | each `examples/**/*.yaml` |
| Launch command and trainer GPU count | the recipe's header comment (`# Launch: …`), plus `launcher-configs/accelerate/*.yaml` for accelerate launches |
| `halo launch` equivalent | `scripts/training/` indexed the way `halo/cli.py` does; multi-node launches get none, because `halo launch` is single-node |
| Model family and its axis support | the compatibility matrix in `agent-docs/models/README.md` |
| Cookbook links | `human-docs/cookbooks/`: a cookbook covers every recipe of a family it cites |

GitHub links and the raw recipe files are pinned to the indexed commit.

## Run it locally

Requires Node 20 or newer and git.

```bash
npm ci
npm run build                         # shallow-clones whitecircle/halo main into .cache/
HALO_REPO_DIR=../halo npm run build   # or index a local checkout instead
npm run preview                       # http://127.0.0.1:8788
npm test                              # extractor tests; HALO_REPO_DIR=../halo also checks a real checkout
```

| Variable | Default | Effect |
| --- | --- | --- |
| `HALO_REF` | `main` | Branch or tag to clone (`git clone --branch` does not accept a commit SHA) |
| `HALO_GITHUB_REPO` | `whitecircle/halo` | Repository to clone and link to |
| `HALO_REPO_DIR` | unset | Index this local checkout instead of cloning |

The build fails if a recipe doesn't parse or if the docs it reads change shape. It prints a warning for any
recipe it can't match to a family.

## Deploy to Cloudflare Pages

**From Git (recommended).** Push this folder to its own GitHub repo, then in the Cloudflare dashboard go to
Workers & Pages, choose Create, then Pages, then Connect to Git, and use these settings:

| Setting | Value |
| --- | --- |
| Build command | `npm run build` |
| Build output directory | `dist` |
| Environment variables | `NODE_VERSION=22`, and optionally `HALO_REF=v1.3.0` to pin a release |

Cloudflare runs the build, which clones Halo, so each deploy indexes the latest `main` (or your pinned ref).

**Direct upload.** Build locally and push the output: `npm run deploy` runs the build and then
`npx wrangler pages deploy dist`. The project name comes from `wrangler.toml`.

`site/_headers` sets a strict Content-Security-Policy: the page loads nothing from other origins, and fonts
are self-hosted. It also marks hashed assets, the index and the raw files as immutable, and makes the HTML
revalidate on every request.

## Keep it current

Halo's recipes change often. `.github/workflows/rebuild.yml` triggers a Cloudflare Pages deploy hook every
day:

1. In the Pages project, go to Settings, then Builds, then Deploy hooks, and create a hook for your
   production branch.
2. In this repo's GitHub settings, add the hook URL as the Actions secret `CLOUDFLARE_DEPLOY_HOOK_URL`.

## Layout

```
scripts/extract.mjs   repo -> recipe index (pure functions, unit tested)
scripts/source.mjs    which Halo checkout to read: local dir or shallow clone
scripts/build.mjs     writes dist/: hashed assets, data/recipes.<hash>.json, raw/<sha>/…
scripts/preview.mjs   local static server
site/                 index.html, app.js (no dependencies), styles.css, _headers
test/                 node:test suite
```

Newsreader and Hanken Grotesk are used under the SIL Open Font License; the build ships each license under
`assets/`.
