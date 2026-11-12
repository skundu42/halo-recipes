# Halo recipe explorer

A searchable index of every training recipe in [Halo](https://github.com/whitecircle/halo). Filter by method,
model family, tuning, parallelism and the GPUs you have, then copy the launch command. It's a static site
deployed as a Cloudflare Worker that serves only static assets.

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

## Deploy to Cloudflare

`wrangler.toml` deploys the site as a Worker that serves `dist/` as static assets. Its `[build]` step runs
`npm run build` before every `wrangler deploy`, so the deploy needs no separate build command.

**From Git (recommended).** In the Cloudflare dashboard, go to Workers & Pages, choose Create, then import
this GitHub repo as a Worker. The defaults work:

| Setting | Value |
| --- | --- |
| Build command | leave empty |
| Deploy command | `npx wrangler deploy` |
| Build variables | optional: `HALO_REF=v1.3.0` to pin a Halo release |

Cloudflare reads the Node version from `.nvmrc`. Each build clones Halo, so every deploy indexes the latest
`main` (or your pinned ref).

**From your machine.** `npm run deploy` builds and deploys with wrangler; run `npx wrangler login` first.

`site/_headers` sets a strict Content-Security-Policy: the page loads nothing from other origins, and fonts
are self-hosted. It also marks hashed assets, the index and the raw files as immutable, and makes the HTML
revalidate on every request.

## Keep it current

Halo's recipes change often. `.github/workflows/rebuild.yml` triggers a Cloudflare deploy hook every day:

1. In the Worker, go to Settings, then Builds, then Deploy Hooks, and create a hook for your production
   branch.
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
