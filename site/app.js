// Halo recipe explorer: loads the build's recipe index and filters it in the browser. Filter state lives in the
// URL, so any view, an open recipe included, is a shareable link.

const DATA_URL = document.querySelector('meta[name="halo-recipes-data"]').content;
const SLOTS_PER_NODE = 8;
const NO_FAMILY = "Not family-specific";
const DATA_PARALLEL = "Data parallel only";
const HAS_COOKBOOK = "Has a cookbook";
const NO_COOKBOOK = "No cookbook yet";
const AXIS_NAMES = {
  EP: "Expert parallel (EP)",
  ETP: "Expert tensor parallel (ETP)",
  CP: "Context parallel (CP)",
  TP: "Tensor parallel (TP)",
  PP: "Pipeline parallel (PP)",
};
const TUNING_NAMES = { Full: "Full fine-tune", LoRA: "LoRA", QLoRA: "QLoRA" };
const ENGINE_NAMES = { vllm: "vLLM", sglang: "SGLang" };
const SORT_KEYS = ["method", "gpus-asc", "gpus-desc", "context", "name"];
const BASE_TITLE = document.title;

const FACETS = [
  { key: "method", legend: "Method", values: (r) => [r.method.label] },
  { key: "family", legend: "Model family", values: (r) => [r.family?.name ?? NO_FAMILY], last: [NO_FAMILY] },
  { key: "tuning", legend: "Tuning", values: (r) => [TUNING_NAMES[r.tuning] ?? r.tuning], first: Object.values(TUNING_NAMES) },
  {
    key: "parallelism",
    legend: "Parallelism",
    values: (r) => (r.axes.length ? r.axes.map((a) => AXIS_NAMES[a.label] ?? a.label) : [DATA_PARALLEL]),
    first: Object.values(AXIS_NAMES),
  },
  { key: "engine", legend: "Rollout engine", values: (r) => (r.rolloutEngine ? [ENGINE_NAMES[r.rolloutEngine] ?? r.rolloutEngine] : []) },
  { key: "cookbook", legend: "Cookbook", values: (r) => [r.cookbooks.length ? HAS_COOKBOOK : NO_COOKBOOK], first: [HAS_COOKBOOK, NO_COOKBOOK] },
];

const state = { gpus: null, q: "", sort: "method", selected: {}, open: null };
const $ = (selector) => document.querySelector(selector);

let index;
let recipes = [];
let byId = new Map();
let rack;
let facetControls = [];
let openedByPush = false;
let lastTrigger = null;

function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
  el.append(...children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false && c !== ""));
  return el;
}

const plural = (n, word) => `${n.toLocaleString("en-US")} ${word}${n === 1 ? "" : "s"}`;
const repoFile = (path) => `${index.source.url}/blob/${index.source.sha}/${path}`;

const announcer = h("p", { class: "visually-hidden", "aria-live": "polite" });
document.body.append(announcer);

// URL state

function readUrl() {
  const params = new URLSearchParams(location.search);
  const gpus = Number.parseInt(params.get("gpus") ?? "", 10);
  state.gpus = Number.isInteger(gpus) && gpus > 0 ? gpus : null;
  state.q = params.get("q") ?? "";
  state.sort = SORT_KEYS.includes(params.get("sort")) ? params.get("sort") : "method";
  for (const facet of FACETS) state.selected[facet.key] = new Set(params.getAll(facet.key));
  state.open = params.get("recipe");
}

function urlFor({ recipe = state.open } = {}) {
  const params = new URLSearchParams();
  if (state.gpus) params.set("gpus", state.gpus);
  if (state.q) params.set("q", state.q);
  if (state.sort !== "method") params.set("sort", state.sort);
  for (const facet of FACETS) for (const value of state.selected[facet.key]) params.append(facet.key, value);
  if (recipe) params.set("recipe", recipe);
  const query = params.toString();
  return `${location.pathname}${query ? `?${query}` : ""}`;
}

function writeUrl({ push = false } = {}) {
  history[push ? "pushState" : "replaceState"](null, "", urlFor());
}

// Filtering and sorting

function searchTerms() {
  return state.q.toLowerCase().split(/\s+/).filter(Boolean);
}

function passes(recipe, skipFacet, terms) {
  if (state.gpus !== null && (recipe.trainerGpus === null || recipe.trainerGpus > state.gpus)) return false;
  if (terms.length && !terms.every((t) => recipe.haystack.includes(t))) return false;
  for (const facet of FACETS) {
    if (facet.key === skipFacet) continue;
    const selected = state.selected[facet.key];
    if (selected.size && !facet.values(recipe).some((v) => selected.has(v))) return false;
  }
  return true;
}

function comparator() {
  const methodRank = new Map(index.methods.map((m, i) => [m.label, i]));
  const byName = (a, b) => a.name.localeCompare(b.name);
  const gpus = (r, missing) => r.trainerGpus ?? missing;
  const family = (r) => r.family?.name ?? "￿";
  return {
    method: (a, b) =>
      methodRank.get(a.method.label) - methodRank.get(b.method.label) ||
      family(a).localeCompare(family(b)) ||
      gpus(a, Infinity) - gpus(b, Infinity) ||
      byName(a, b),
    "gpus-asc": (a, b) => gpus(a, Infinity) - gpus(b, Infinity) || byName(a, b),
    "gpus-desc": (a, b) => gpus(b, -1) - gpus(a, -1) || byName(a, b),
    context: (a, b) => (b.maxLength ?? -1) - (a.maxLength ?? -1) || byName(a, b),
    name: byName,
  }[state.sort];
}

// Labels

function parallelismLabel(r) {
  if (r.axes.length) return r.axes.map((a) => `${a.label}${a.size}`).join(" + ");
  return r.trainerGpus === 1 ? "Single GPU" : "Data parallel";
}

const contextLabel = (r) => (r.maxLength ? r.maxLength.toLocaleString("en-US") : "Default");

function gpuSentence(r) {
  if (r.trainerGpus === null) return "Not stated in the launch command";
  if (r.nodes > 1) return `${r.trainerGpus} across ${r.nodes} nodes, ${r.gpusPerNode} on each`;
  return r.trainerGpus === 1 ? "1, a single process" : `${r.trainerGpus} on one node`;
}

function gpuCell(r) {
  if (r.trainerGpus === null) return [h("span", { class: "gpu-count", title: "The launch command does not state a GPU count" }, "?")];
  const perNode = r.gpusPerNode;
  const width = Math.ceil(perNode / SLOTS_PER_NODE) * SLOTS_PER_NODE;
  const cells = [];
  for (let node = 0; node < r.nodes; node++) {
    for (let i = 0; i < width; i++) cells.push(h("span", { class: i < perNode ? "glyph-cell is-on" : "glyph-cell" }));
  }
  const text = r.nodes > 1 ? `${r.nodes} × ${perNode}` : String(perNode);
  return [
    h("span", { class: "glyph", "aria-hidden": "true" }, cells),
    h("span", { class: "gpu-count", "aria-hidden": "true" }, text),
    h("span", { class: "visually-hidden" }, gpuSentence(r)),
  ];
}

// GPU rack

function buildRack() {
  const maxGpus = Math.max(0, ...recipes.map((r) => r.trainerGpus ?? 0));
  const nodes = Math.max(2, Math.ceil(maxGpus / SLOTS_PER_NODE));
  const root = $("#rack");
  const slots = [];
  for (let node = 0; node < nodes; node++) {
    const cells = [];
    for (let i = 0; i < SLOTS_PER_NODE; i++) {
      const count = node * SLOTS_PER_NODE + i + 1;
      const slot = h(
        "label",
        { class: "slot", dataset: { count } },
        h("input", { class: "slot-input", type: "radio", name: "gpus", value: count, "aria-label": `Up to ${plural(count, "GPU")}` }),
        h("span", { class: "pad", "aria-hidden": "true" }),
        h("span", { class: "slot-number", "aria-hidden": "true" }, String(i)),
      );
      slot.style.setProperty("--i", String(count - 1)); // staggers the light-up sweep
      slots.push(slot);
      cells.push(slot);
    }
    root.append(h("div", { class: "board" }, h("p", { class: "board-label" }, `Node ${node + 1}`), h("div", { class: "slots" }, cells)));
  }
  const anyInput = h("input", { class: "slot-input", type: "radio", name: "gpus", value: "any" });
  const any = h("label", { class: "rack-any pill" }, anyInput, "Any number");
  root.append(any);

  root.addEventListener("change", (event) => {
    state.gpus = event.target.value === "any" ? null : Number(event.target.value);
    update();
  });
  root.addEventListener("pointerover", (event) => previewRack(Number(event.target.closest(".slot")?.dataset.count) || null));
  root.addEventListener("pointerleave", () => previewRack(null));
  return { slots, any, anyInput };
}

function previewRack(count) {
  for (const slot of rack.slots) {
    const n = Number(slot.dataset.count);
    slot.classList.toggle("is-preview", count !== null && n <= count && (state.gpus === null || n > state.gpus));
  }
}

function paintRack() {
  for (const slot of rack.slots) {
    const n = Number(slot.dataset.count);
    slot.classList.toggle("is-on", state.gpus !== null && n <= state.gpus);
    slot.querySelector("input").checked = n === state.gpus;
  }
  rack.any.classList.toggle("is-on", state.gpus === null);
  rack.anyInput.checked = state.gpus === null;
  const nodes = Math.ceil((state.gpus ?? 0) / SLOTS_PER_NODE);
  $("#rack-readout").textContent =
    state.gpus === null ? "Pick a slot to filter by GPU count" : `Up to ${plural(state.gpus, "GPU")}${nodes > 1 ? ` across ${nodes} nodes` : ""}`;
}

// Facets

function orderedValues(facet) {
  const seen = new Set(recipes.flatMap((r) => facet.values(r)));
  const ranked = facet.key === "method" ? index.methods.map((m) => m.label) : (facet.first ?? []);
  const head = ranked.filter((v) => seen.has(v));
  const tail = (facet.last ?? []).filter((v) => seen.has(v));
  const rest = [...seen].filter((v) => !head.includes(v) && !tail.includes(v)).sort((a, b) => a.localeCompare(b));
  return [...head, ...rest, ...tail];
}

function buildFacets() {
  const form = $("#filters");
  facetControls = FACETS.flatMap((facet) => {
    const values = orderedValues(facet);
    if (!values.length) return [];
    const options = values.map((value) => {
      const input = h("input", { class: "option-input", type: "checkbox", name: facet.key, value });
      const count = h("span", { class: "option-count" });
      const label = h("label", { class: "option" }, input, h("span", {}, value), count);
      return { value, input, count, label };
    });
    form.append(h("fieldset", { class: "facet" }, h("legend", { class: "facet-legend" }, facet.legend), options.map((o) => o.label)));
    return [{ facet, options }];
  });
  form.addEventListener("change", (event) => {
    const { name, value, checked } = event.target;
    if (checked) state.selected[name].add(value);
    else state.selected[name].delete(value);
    update();
  });
  // Values in a shared link that this index no longer has would hide every recipe.
  for (const { facet, options } of facetControls) {
    const known = new Set(options.map((o) => o.value));
    state.selected[facet.key] = new Set([...state.selected[facet.key]].filter((v) => known.has(v)));
  }
}

function paintFacets(terms) {
  let active = state.gpus === null ? 0 : 1;
  for (const { facet, options } of facetControls) {
    const counts = new Map();
    for (const recipe of recipes) {
      if (!passes(recipe, facet.key, terms)) continue;
      for (const value of new Set(facet.values(recipe))) counts.set(value, (counts.get(value) ?? 0) + 1);
    }
    for (const option of options) {
      const n = counts.get(option.value) ?? 0;
      option.input.checked = state.selected[facet.key].has(option.value);
      option.count.textContent = n;
      option.label.classList.toggle("is-empty", n === 0 && !option.input.checked);
    }
    active += state.selected[facet.key].size;
  }
  $("#filters-active").textContent = active ? `(${active} on)` : "";
}

// Results table

function recipeRow(r) {
  return h(
    "tr",
    { class: "recipe-row", dataset: { id: r.id } },
    h("td", { class: "cell-gpus" }, gpuCell(r)),
    h(
      "td",
      { class: "cell-recipe" },
      h("a", { class: "recipe-link", href: urlFor({ recipe: r.id }), dataset: { open: r.id } }, r.name),
      h("span", { class: "recipe-model" }, r.model.id),
    ),
    h("td", { "data-label": "Method" }, r.method.label),
    h("td", { "data-label": "Family" }, r.family?.name ?? NO_FAMILY),
    h("td", { class: "cell-tight", "data-label": "Parallelism" }, parallelismLabel(r)),
    h("td", { class: "cell-tight", "data-label": "Tuning" }, TUNING_NAMES[r.tuning] ?? r.tuning),
    h("td", { class: "cell-context", "data-label": "Context" }, contextLabel(r)),
  );
}

function paintRows(terms) {
  const shown = recipes.filter((r) => passes(r, null, terms)).sort(comparator());
  $("#recipe-rows").replaceChildren(...shown.map(recipeRow));
  $("#recipes").hidden = shown.length === 0;
  $("#empty").hidden = shown.length !== 0;
  const fit = state.gpus === null ? "" : ` fit up to ${plural(state.gpus, "GPU")}`;
  $("#count").textContent =
    shown.length === recipes.length ? `${plural(recipes.length, "recipe")}${fit}` : `${shown.length} of ${plural(recipes.length, "recipe")}${fit}`;
}

function update() {
  const terms = searchTerms();
  writeUrl();
  paintRack();
  paintFacets(terms);
  paintRows(terms);
}

// Recipe detail

function link(href, text) {
  return href ? h("a", { href, rel: "noopener" }, text) : text;
}

function section(title, ...children) {
  return h("section", { class: "detail-section" }, h("h3", { class: "detail-heading" }, title), children);
}

function commandBlock(text) {
  const code = h("code", {}, text);
  const button = h("button", { type: "button", class: "button" }, "Copy");
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
      announcer.textContent = "Command copied";
    } catch {
      getSelection().selectAllChildren(code);
      button.textContent = "Press Ctrl+C";
    }
    setTimeout(() => (button.textContent = "Copy"), 2000);
  });
  return h("div", { class: "command" }, h("pre", { class: "command-text" }, code), button);
}

function launchSection(r) {
  const [primary, ...alternatives] = r.launch.commands;
  const engine = ENGINE_NAMES[r.rolloutEngine] ?? r.rolloutEngine ?? "rollout";
  return section(
    "Launch",
    h(
      "p",
      { class: "detail-text" },
      "Run from the repository root inside the ",
      link(`${repoFile("human-docs/cookbooks/README.md")}#start-the-training-container`, "Halo training container"),
      ".",
    ),
    r.needsRolloutServer &&
      h(
        "p",
        { class: "detail-text" },
        `Start a ${engine} server on GPUs the trainer does not use first. The recipe notes below give its flags; `,
        link(repoFile("human-docs/rollout-servers.md"), "Rollout servers"),
        " covers the setup.",
      ),
    r.launch.halo && [h("p", { class: "detail-text" }, "With the halo CLI:"), commandBlock(r.launch.halo)],
    primary && [h("p", { class: "detail-text" }, r.launch.halo ? "Or with the launcher the recipe names:" : "As the recipe launches it:"), commandBlock(primary)],
    r.nodes > 1 && h("p", { class: "detail-text" }, "halo launch runs on one node, so this recipe starts with torchrun on every node, each with its own NODE_RANK."),
    alternatives.length > 0 && [h("p", { class: "detail-text" }, "The recipe also lists:"), alternatives.map(commandBlock)],
    !primary && h("p", { class: "detail-text" }, "The recipe states no launch command. Its notes below say how to run it."),
  );
}

function supportSection(family) {
  const axes = Object.keys(family.support);
  return section(
    "What this family supports",
    h(
      "table",
      { class: "support" },
      h("thead", {}, h("tr", {}, axes.map((axis) => h("th", { scope: "col" }, axis)))),
      h("tbody", {}, h("tr", {}, axes.map((axis) => h("td", { class: /^yes/i.test(family.support[axis]) ? "is-yes" : "is-no" }, family.support[axis] || "—")))),
    ),
    h(
      "p",
      { class: "detail-text" },
      "From the repo's compatibility matrix. The footnotes, and why a family drops an axis, are on ",
      link(family.page && repoFile(family.page), "its model page"),
      ".",
    ),
  );
}

function highlightYaml(text) {
  return text.split("\n").flatMap((line, i, lines) => {
    const parts = [];
    const comment = line.match(/^(\s*)(#.*)$/);
    const entry = line.match(/^(\s*(?:-\s+)?)([\w.-]+)(:)(.*)$/);
    if (comment) parts.push(comment[1], h("span", { class: "y-comment" }, comment[2]));
    else if (entry) {
      const [, lead, key, colon, rest] = entry;
      const at = rest.search(/\s#/);
      parts.push(lead, h("span", { class: "y-key" }, key), colon, at >= 0 ? rest.slice(0, at) : rest);
      if (at >= 0) parts.push(h("span", { class: "y-comment" }, rest.slice(at)));
    } else parts.push(line);
    if (i < lines.length - 1) parts.push("\n");
    return parts;
  });
}

function yamlSection(r) {
  const pre = h("pre", { class: "yaml" }, "Loading the recipe file…");
  fetch(r.raw)
    .then((response) => {
      if (!response.ok) throw new Error(String(response.status));
      return response.text();
    })
    .then((text) => pre.replaceChildren(...highlightYaml(text)))
    .catch(() => pre.replaceChildren("This file did not load. Open it on GitHub instead."));
  return section(
    "Full recipe",
    h(
      "p",
      { class: "yaml-actions" },
      h("a", { class: "pill", href: r.github, rel: "noopener" }, "Open on GitHub"),
      h("a", { class: "pill", href: r.raw, download: `${r.name}.yaml` }, "Download YAML"),
    ),
    pre,
  );
}

function renderDetail(r) {
  const guides = [
    ...r.cookbooks.map((c) => h("li", {}, link(repoFile(c.path), `${c.name} cookbook`))),
    r.family?.page && h("li", {}, link(repoFile(r.family.page), `${r.family.name} model page`)),
    h("li", {}, link(repoFile("human-docs/supported-matrix.md"), "Supported matrix")),
  ];
  $("#detail-body").replaceChildren(
    h("p", { class: "detail-method" }, r.family ? `${r.method.label}, ${r.family.name}` : r.method.label),
    h("h2", { class: "detail-title", id: "detail-title" }, r.name),
    r.summary && h("p", { class: "detail-summary" }, r.summary),
    h(
      "dl",
      { class: "facts" },
      h("dt", {}, "Model"),
      h("dd", {}, link(r.model.url, r.model.id)),
      r.datasets.length > 0 && [h("dt", {}, r.datasets.length === 1 ? "Dataset" : "Datasets"), h("dd", {}, r.datasets.map((d, i) => [i ? ", " : "", link(d.url, d.name)]))],
      h("dt", {}, "GPUs"),
      h("dd", {}, gpuSentence(r)),
      h("dt", {}, "Parallelism"),
      h("dd", {}, parallelismLabel(r)),
      h("dt", {}, "Tuning"),
      h("dd", {}, TUNING_NAMES[r.tuning] ?? r.tuning),
      h("dt", {}, "Context length"),
      h("dd", {}, r.maxLength ? `${r.maxLength.toLocaleString("en-US")} tokens` : "The method's default"),
      r.rolloutEngine && [h("dt", {}, "Rollout engine"), h("dd", {}, ENGINE_NAMES[r.rolloutEngine] ?? r.rolloutEngine)],
    ),
    launchSection(r),
    r.notes && section("Notes from the recipe", h("pre", { class: "notes" }, r.notes)),
    r.settings.length > 0 && section("Key settings", h("dl", { class: "facts settings" }, r.settings.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]))),
    r.family && supportSection(r.family),
    section("Guides", h("ul", { class: "link-list" }, guides)),
    yamlSection(r),
  );
}

function showRecipe(id, { push }) {
  const recipe = byId.get(id);
  if (!recipe) return;
  state.open = id;
  renderDetail(recipe);
  document.title = `${recipe.name} | ${BASE_TITLE}`;
  const dialog = $("#detail");
  if (!dialog.open) {
    lastTrigger = document.activeElement;
    dialog.showModal();
    dialog.scrollTop = 0;
  }
  if (push) {
    openedByPush = true;
    writeUrl({ push: true });
  }
}

function wireDetail() {
  const dialog = $("#detail");
  $("#detail-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("close", () => {
    document.title = BASE_TITLE;
    if (lastTrigger?.isConnected) lastTrigger.focus();
    if (!state.open) return; // closed by a history step, which already set the URL
    state.open = null;
    if (openedByPush) {
      openedByPush = false;
      history.back();
    } else writeUrl();
  });

  $("#recipe-rows").addEventListener("click", (event) => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const row = event.target.closest(".recipe-row");
    if (!row || (!event.target.closest("a") && getSelection().toString())) return;
    event.preventDefault();
    showRecipe(row.dataset.id, { push: true });
  });

  window.addEventListener("popstate", () => {
    readUrl();
    openedByPush = false;
    syncControls();
    update();
    if (state.open && byId.has(state.open)) showRecipe(state.open, { push: false });
    else if (dialog.open) dialog.close();
  });
}

// Wiring

function syncControls() {
  $("#q").value = state.q;
  $("#sort").value = state.sort;
}

function wireToolbar() {
  $("#q").addEventListener("input", (event) => {
    state.q = event.target.value;
    update();
  });
  $("#sort").addEventListener("change", (event) => {
    state.sort = event.target.value;
    update();
  });
  $("#clear-all").addEventListener("click", () => {
    state.gpus = null;
    state.q = "";
    for (const facet of FACETS) state.selected[facet.key].clear();
    syncControls();
    update();
  });
  const panel = $("#filters-panel");
  const wide = matchMedia("(min-width: 900px)");
  const openOnWide = () => {
    if (wide.matches) panel.open = true;
  };
  wide.addEventListener("change", openOnWide);
  openOnWide();
}

async function main() {
  readUrl();
  let response;
  try {
    response = await fetch(DATA_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    index = await response.json();
  } catch (error) {
    $("#recipes").hidden = true;
    $("#count").textContent = `The recipe index did not load (${error.message}). Reload the page to try again.`;
    return;
  }
  recipes = index.recipes.map((r) => ({
    ...r,
    haystack: [r.name, r.path, r.model.id, r.method.label, r.family?.name, r.summary, ...r.datasets.map((d) => d.name)]
      .filter(Boolean)
      .join(" ")
      .toLowerCase(),
  }));
  byId = new Map(recipes.map((r) => [r.id, r]));
  rack = buildRack();
  buildFacets();
  syncControls();
  wireToolbar();
  wireDetail();
  update();
  if (state.open) {
    if (byId.has(state.open)) showRecipe(state.open, { push: false });
    else {
      state.open = null;
      writeUrl();
    }
  }
}

main();
