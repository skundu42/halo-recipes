// Builds the recipe index from a Halo checkout. Every field is read or derived from the repo itself:
// recipe YAMLs and their header comments, scripts/training, the compatibility matrix and the cookbooks.
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join, posix, relative, sep } from "node:path";
import YAML from "yaml";

const AXES = [
  ["expert_parallel_size", "EP"],
  ["expert_tensor_parallel_size", "ETP"],
  ["context_parallel_size", "CP"],
  ["tensor_parallel_size", "TP"],
  ["pipeline_parallel_size", "PP"],
];

const METHOD_LABELS = {
  sft: "SFT",
  "preference/dpo": "DPO",
  "preference/kto": "KTO",
  "preference/smpo": "SMPO",
  "reward/reward-model": "Reward model",
  classification: "Classification",
  embedding: "Embedding",
  "distillation/self-distill": "Self-distillation",
  "distillation/teacher-distill": "Teacher distillation",
  "grpo/offline": "GRPO, offline",
  "grpo/online": "GRPO, online",
  "grpo/environmental": "GRPO with environments",
};

const ROLLOUT_METHODS = new Set(["grpo/online", "grpo/environmental"]);

// Recipe fields worth surfacing next to the full YAML, in display order.
const KEY_SETTINGS = [
  "dataset",
  "max_length",
  "max_prompt_length",
  "max_completion_length",
  "learning_rate",
  "num_train_epochs",
  "max_steps",
  "per_device_train_batch_size",
  "gradient_accumulation_steps",
  "attn_implementation",
  "packing",
  "gradient_checkpointing",
  "lora_r",
  "lora_alpha",
  "lora_target_modules",
  "loss_type",
  "beta",
  "num_generations",
  "environment_type",
  "rollout_backend",
  "moe_balancing",
  "teacher_model",
];

const FOOTNOTES = /[⁰¹²³⁴⁵⁶⁷⁸⁹]+/g;

const toPosix = (p) => p.split(sep).join(posix.sep);

function walk(dir, keep) {
  return readdirSync(dir)
    .sort()
    .flatMap((name) => {
      const path = join(dir, name);
      return statSync(path).isDirectory() ? walk(path, keep) : keep(path) ? [path] : [];
    });
}

/** The recipe's leading comment block, up to the first `# --- Section ---` divider or config line. */
export function headerComment(text) {
  const lines = [];
  for (const line of text.split(/\r?\n/)) {
    if (/^#\s*---/.test(line)) break;
    if (line.startsWith("#")) {
      lines.push(line.replace(/^# ?/, ""));
      continue;
    }
    if (line.trim() === "" && lines.length === 0) continue;
    break;
  }
  return lines;
}

const COMMAND = new RegExp(
  String.raw`((?:[A-Z][A-Z0-9_]*=\S+\s+)*)` + // env prefix
    String.raw`(torchrun|python3?|accelerate launch)\s+` +
    String.raw`((?:--?[\w-]+(?:=\S+)?(?:\s+(?!-|scripts/)[^\s<]+)?\s+)*?)` + // launcher flags
    String.raw`(scripts/training/\S+\.py)\s+<this config>` +
    String.raw`((?:\s+--\S+)*)`, // config overrides
  "g",
);

const words = (s) => s.trim().split(/\s+/).filter(Boolean);

/** Every launch command the header spells out, first one primary; backslash continuations are joined. */
export function parseLaunchCommands(headerLines) {
  const logical = [];
  for (const line of headerLines) {
    const prev = logical.at(-1);
    if (prev !== undefined && prev.endsWith("\\")) logical[logical.length - 1] = `${prev.slice(0, -1).trimEnd()} ${line.trim()}`;
    else logical.push(line.trim());
  }
  return logical.flatMap((line) =>
    [...line.matchAll(COMMAND)].map(([, env, launcher, flags, script, extra]) => ({
      env: words(env),
      launcher,
      flags: words(flags),
      script,
      extra: words(extra),
    })),
  );
}

// torchrun and accelerate take `--nproc-per-node` and `--nproc_per_node` alike.
const flagName = (flag) => flag.split("=", 1)[0].replace(/^--/, "").replaceAll("-", "_");

function flagValue(flags, name) {
  const want = flagName(name);
  for (let i = 0; i < flags.length; i++) {
    if (flagName(flags[i]) !== want) continue;
    return flags[i].includes("=") ? flags[i].slice(flags[i].indexOf("=") + 1) : flags[i + 1];
  }
  return undefined;
}

function positiveInt(value) {
  const n = Number.parseInt(value, 10);
  return Number.isInteger(n) && n > 0 && String(n) === String(value) ? n : null;
}

/** Trainer processes per node and node count; `null` where the command leaves them to the shell. */
export function trainerLayout(command, readAccelerateConfig) {
  if (command.launcher.startsWith("python")) return { gpusPerNode: 1, nodes: 1 };
  if (command.launcher === "torchrun") {
    const nnodes = flagValue(command.flags, "--nnodes");
    return {
      gpusPerNode: positiveInt(flagValue(command.flags, "--nproc_per_node")),
      nodes: nnodes === undefined ? 1 : positiveInt(nnodes),
    };
  }
  const config = readAccelerateConfig(flagValue(command.flags, "--config_file"));
  const explicit = flagValue(command.flags, "--num_processes");
  return {
    gpusPerNode: explicit === undefined ? positiveInt(config?.num_processes) : positiveInt(explicit),
    nodes: positiveInt(config?.num_machines ?? 1),
  };
}

export function renderCommand(command, configPath) {
  return [...command.env, command.launcher, ...command.flags, command.script, configPath, ...command.extra].join(" ");
}

/** The `halo launch` method name for a training script, as the CLI indexes it (path under scripts/training, hyphenated). */
export function haloMethodName(script) {
  return script.replace(/^scripts\/training\//, "").replace(/\.py$/, "").replaceAll("_", "-");
}

/** The equivalent `halo launch` line, or `null` when the command uses a launcher flag the CLI does not take. */
export function haloLaunchCommand(command, configPath, methods) {
  const method = haloMethodName(command.script);
  if (!methods.has(method)) return null;
  const options = [];
  if (command.launcher === "torchrun") {
    for (const flag of command.flags) {
      const value = flag.slice(flag.indexOf("=") + 1);
      if (!flag.includes("=") || !positiveInt(value)) return null;
      if (flagName(flag) === "nproc_per_node") options.push("-n", value);
      else if (flagName(flag) === "master_port") options.push("-p", value);
      else return null;
    }
  } else if (command.launcher === "accelerate launch") {
    for (let i = 0; i < command.flags.length; i++) {
      const name = flagName(command.flags[i]);
      const value = command.flags[i].includes("=") ? command.flags[i].slice(command.flags[i].indexOf("=") + 1) : command.flags[++i];
      if (name === "config_file") options.push("-a", value);
      else if (name === "num_processes") options.push("-n", value);
      else return null;
    }
  } else if (command.flags.length) return null;
  return [...command.env, "halo launch", method, configPath, ...options, ...command.extra].join(" ");
}

/** Training methods `halo launch` indexes: scripts under scripts/training with a `__main__` guard. */
export function trainingMethods(repoDir) {
  const root = join(repoDir, "scripts", "training");
  return new Set(
    walk(root, (p) => p.endsWith(".py"))
      .filter((p) => readFileSync(p, "utf8").includes("__main__"))
      .map((p) => haloMethodName(`scripts/training/${toPosix(relative(root, p))}`)),
  );
}

export function globToRegExp(glob) {
  const body = glob
    .split("*")
    .map((part) => part.replace(/[.+?^${}()|[\]\\]/g, "\\$&"))
    .join("[^/]*");
  return new RegExp(`^${body}$`);
}

function parseTable(lines) {
  const rows = lines.filter((l) => l.trim().startsWith("|")).map((l) => l.trim().slice(1, -1).split("|").map((c) => c.trim()));
  const [header, , ...body] = rows;
  return body.map((cells) => Object.fromEntries(header.map((h, i) => [h, cells[i] ?? ""])));
}

const backticked = (cell) => [...cell.matchAll(/`([^`]+)`/g)].map((m) => m[1]);

function markdownLink(cell) {
  const m = cell.match(/^\[([^\]]+)\]\(([^)]+)\)/);
  return m ? { text: m[1], href: m[2] } : { text: cell, href: null };
}

/** Families from agent-docs/models/README.md's compatibility matrix, each with its recipe globs and axis verdicts. */
export function parseCompatibilityMatrix(markdown) {
  const section = markdown.split(/^## Compatibility matrix\s*$/m)[1]?.split(/^## /m)[0];
  if (!section) throw new Error("agent-docs/models/README.md lost its '## Compatibility matrix' section; update parseCompatibilityMatrix");
  return parseTable(section.split("\n")).flatMap((row) => {
    const { text, href } = markdownLink(row["Model Family"]);
    const globs = backticked(row["Training Configs"] ?? "").filter((g) => g.startsWith("examples/"));
    if (!globs.length) return [];
    const support = Object.fromEntries(AXES.map(([, label]) => [label, (row[label] ?? "").replace(FOOTNOTES, "").trim()]));
    return [{ name: text, page: href && `agent-docs/models/${href}`, modelType: backticked(row["`model_type`"] ?? "")[0] ?? null, support, globs }];
  });
}

const specificity = (glob) => glob.replaceAll("*", "").length;

/** The family whose glob matches the recipe path; else one whose folder and file pattern match under any method. */
export function matchFamily(path, families) {
  const candidates = (test) =>
    families
      .flatMap((family) => family.globs.filter(test).map((glob) => ({ family, score: specificity(glob) })))
      .sort((a, b) => b.score - a.score);
  const exact = candidates((glob) => globToRegExp(glob).test(path));
  if (exact.length) return exact[0].family;
  const dirs = path.split("/").slice(1, -1);
  const file = path.split("/").at(-1);
  const anyMethod = candidates((glob) => {
    const [, , folder, pattern] = glob.split("/");
    return pattern !== undefined && !folder.includes("*") && dirs.includes(folder) && globToRegExp(pattern).test(file);
  });
  return anyMethod[0]?.family ?? null;
}

/** Cookbooks from human-docs/cookbooks/README.md, with the recipe models they name and the recipe paths they cite. */
export function parseCookbooks(readme, readCookbook) {
  const table = readme.split("\n").filter((l) => l.trim().startsWith("|"));
  return parseTable(table)
    .map((row) => {
      const { text, href } = markdownLink(row.Cookbook ?? "");
      if (!href) return null;
      const body = readCookbook(href);
      return {
        name: text,
        path: `human-docs/cookbooks/${href}`,
        models: backticked(row["Recipe model"] ?? ""),
        cites: [...new Set(body.match(/examples\/[\w./*-]+\.yaml/g) ?? [])],
      };
    })
    .filter(Boolean);
}

const HUB_ID = /^[\w.-]+\/[\w.-]+$/;

function hubLink(id, kind) {
  const bare = id.replace(/[@:].*$/, "");
  if (!HUB_ID.test(bare) || bare.startsWith(".")) return null;
  return kind === "dataset" ? `https://huggingface.co/datasets/${bare}` : `https://huggingface.co/${bare}`;
}

function datasetsOf(value) {
  const list = value === undefined || value === null ? [] : Array.isArray(value) ? value : [value];
  return list.map((d) => {
    const name = typeof d === "string" ? d : (d.path ?? d.name ?? JSON.stringify(d));
    return { name, url: hubLink(name, "dataset") };
  });
}

function settingValue(value) {
  if (Array.isArray(value)) return value.map((v) => (typeof v === "string" ? v : JSON.stringify(v))).join(", ");
  if (value !== null && typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function methodOf(path, command) {
  const fromScript = command && haloMethodName(command.script);
  if (fromScript) return fromScript;
  const [, top, variant] = path.split("/");
  return top === "grpo" ? `grpo/${variant}` : top;
}

function rolloutEngine(method, path, config) {
  if (!ROLLOUT_METHODS.has(method)) return null;
  if (config.rollout_backend) return config.rollout_backend;
  const segment = path.split("/").find((s) => s === "vllm" || s === "sglang");
  if (segment) return segment;
  // rollout_backend defaults to vllm; online GRPO generates through TRL's vLLM client.
  return method === "grpo/environmental" || config.use_vllm ? "vllm" : null;
}

/** Index every recipe YAML under examples/. Throws on a recipe that does not parse: a broken index is worse than none. */
export function extractRecipes(repoDir, { githubBlob }) {
  const cliMethods = trainingMethods(repoDir);
  const families = parseCompatibilityMatrix(readFileSync(join(repoDir, "agent-docs/models/README.md"), "utf8"));
  const cookbooks = parseCookbooks(readFileSync(join(repoDir, "human-docs/cookbooks/README.md"), "utf8"), (file) =>
    readFileSync(join(repoDir, "human-docs/cookbooks", file), "utf8"),
  );
  const accelerateConfigs = new Map();
  const readAccelerateConfig = (file) => {
    if (!file || !existsSync(join(repoDir, file))) return null;
    if (!accelerateConfigs.has(file)) accelerateConfigs.set(file, YAML.parse(readFileSync(join(repoDir, file), "utf8")));
    return accelerateConfigs.get(file);
  };

  const warnings = [];
  const recipes = walk(join(repoDir, "examples"), (p) => p.endsWith(".yaml")).map((file) => {
    const path = toPosix(relative(repoDir, file));
    const text = readFileSync(file, "utf8");
    let config;
    try {
      config = YAML.parse(text) ?? {};
    } catch (error) {
      throw new Error(`${path} does not parse as YAML: ${error.message}`);
    }
    const header = headerComment(text);
    const commands = parseLaunchCommands(header);
    const [primary] = commands;
    if (!primary) warnings.push(`${path}: no launch command in its header comment`);
    const method = methodOf(path, primary);
    const layout = primary ? trainerLayout(primary, readAccelerateConfig) : { gpusPerNode: null, nodes: null };
    const family = matchFamily(path, families);
    if (!family) warnings.push(`${path}: no compatibility-matrix family matches`);
    const axes = AXES.flatMap(([key, label]) => (Number(config[key]) > 1 ? [{ label, size: Number(config[key]) }] : []));
    const tuning = config.load_in_4bit ? "QLoRA" : config.use_peft ? "LoRA" : "Full";

    return {
      id: path.replace(/^examples\//, "").replace(/\.yaml$/, ""),
      path,
      name: path.split("/").at(-1).replace(/\.yaml$/, ""),
      summary: header[0] ?? "",
      notes: header.join("\n"),
      method: { slug: method, label: METHOD_LABELS[method] ?? method },
      family: family && { name: family.name, page: family.page, modelType: family.modelType, support: family.support },
      model: { id: String(config.model_name_or_path ?? ""), url: hubLink(String(config.model_name_or_path ?? ""), "model") },
      datasets: datasetsOf(config.dataset),
      tuning,
      axes,
      nodes: layout.nodes,
      gpusPerNode: layout.gpusPerNode,
      trainerGpus: layout.nodes && layout.gpusPerNode ? layout.nodes * layout.gpusPerNode : null,
      rolloutEngine: rolloutEngine(method, path, config),
      needsRolloutServer: ROLLOUT_METHODS.has(method),
      maxLength: Number.isInteger(config.max_length) ? config.max_length : null,
      launch: {
        commands: commands.map((c) => renderCommand(c, path)),
        halo: primary ? haloLaunchCommand(primary, path, cliMethods) : null,
      },
      settings: KEY_SETTINGS.filter((k) => config[k] !== undefined && config[k] !== null).map((k) => [k, settingValue(config[k])]),
      cookbooks: cookbooks
        .filter((c) => c.cites.includes(path) || c.models.includes(config.model_name_or_path))
        .map(({ name, path: page }) => ({ name, path: page })),
      github: githubBlob(path),
    };
  });

  // A cookbook covers its whole family: extend each family's cookbooks to every recipe of that family.
  const familyCookbooks = new Map();
  for (const r of recipes) {
    if (!r.family) continue;
    const known = familyCookbooks.get(r.family.name) ?? new Map();
    for (const c of r.cookbooks) known.set(c.path, c);
    familyCookbooks.set(r.family.name, known);
  }
  for (const r of recipes) {
    if (!r.family) continue;
    const merged = new Map(r.cookbooks.map((c) => [c.path, c]));
    for (const [p, c] of familyCookbooks.get(r.family.name)) merged.set(p, c);
    r.cookbooks = [...merged.values()];
  }

  if (!recipes.length) throw new Error(`no recipe YAMLs under ${join(repoDir, "examples")}`);
  const present = new Set(recipes.map((r) => r.method.slug));
  const methods = [...Object.keys(METHOD_LABELS), ...present]
    .filter((slug, i, all) => present.has(slug) && all.indexOf(slug) === i)
    .map((slug) => ({ slug, label: METHOD_LABELS[slug] ?? slug }));
  return { recipes, methods, families: families.map(({ name, page }) => ({ name, page })), warnings };
}
