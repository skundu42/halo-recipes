import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { test } from "node:test";
import {
  extractRecipes,
  globToRegExp,
  haloLaunchCommand,
  headerComment,
  matchFamily,
  parseCompatibilityMatrix,
  parseLaunchCommands,
  renderCommand,
  trainerLayout,
} from "../scripts/extract.mjs";

const METHODS = new Set(["sft", "grpo/online", "grpo/environmental", "preference/dpo", "embedding"]);
const noAccelerate = () => null;

test("header stops at the first section divider and skips leading blank lines", () => {
  const text = "\n# Summary line.\n# Launch: python scripts/training/sft.py <this config>\n# --- Model ---\nmodel_name_or_path: x/y\n";
  assert.deepEqual(headerComment(text), ["Summary line.", "Launch: python scripts/training/sft.py <this config>"]);
});

test("backslash continuations join into one multi-node command", () => {
  const [cmd] = parseLaunchCommands([
    "Launch on EACH node:",
    "  torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \\",
    "    --master_addr=$MASTER_ADDR --master_port=29500 scripts/training/sft.py <this config>",
  ]);
  assert.equal(cmd.script, "scripts/training/sft.py");
  assert.deepEqual(trainerLayout(cmd, noAccelerate), { gpusPerNode: 8, nodes: 2 });
  assert.equal(haloLaunchCommand(cmd, "examples/a.yaml", METHODS), null, "halo launch is single-node");
});

test("env prefixes survive and prose before the command is ignored", () => {
  const [cmd] = parseLaunchCommands([
    "Launch: 4 train GPUs + 2 vLLM servers TP=2 on the other 4; CUDA_VISIBLE_DEVICES=4,5,6,7 DIST_NCCL_TIMEOUT_MINUTES=60 torchrun --nproc_per_node=4 scripts/training/grpo/environmental.py <this config>",
  ]);
  assert.deepEqual(cmd.env, ["CUDA_VISIBLE_DEVICES=4,5,6,7", "DIST_NCCL_TIMEOUT_MINUTES=60"]);
  assert.equal(
    haloLaunchCommand(cmd, "examples/r.yaml", METHODS),
    "CUDA_VISIBLE_DEVICES=4,5,6,7 DIST_NCCL_TIMEOUT_MINUTES=60 halo launch grpo/environmental examples/r.yaml -n 4",
  );
});

test("alternatives keep their order and their config overrides", () => {
  const commands = parseLaunchCommands([
    "Launch: python scripts/training/embedding.py <this config>",
    "  or: torchrun --nproc_per_node=2 scripts/training/embedding.py <this config> --tensor_parallel_size=2 --batch_sampler=batch_sampler",
  ]);
  assert.equal(commands.length, 2);
  assert.deepEqual(trainerLayout(commands[0], noAccelerate), { gpusPerNode: 1, nodes: 1 });
  assert.equal(
    renderCommand(commands[1], "examples/e.yaml"),
    "torchrun --nproc_per_node=2 scripts/training/embedding.py examples/e.yaml --tensor_parallel_size=2 --batch_sampler=batch_sampler",
  );
});

test("hyphenated torchrun flags count the same as underscored ones", () => {
  const [cmd] = parseLaunchCommands(["torchrun --nnodes=2 --node-rank=<r> --nproc-per-node=8 --master-addr=<addr> scripts/training/sft.py <this config>"]);
  assert.deepEqual(trainerLayout(cmd, noAccelerate), { gpusPerNode: 8, nodes: 2 });
});

test("accelerate reads its process count from the launcher config", () => {
  const [cmd] = parseLaunchCommands([
    "Launch: accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml scripts/training/preference/dpo.py <this config>",
  ]);
  const config = (file) => (file === "launcher-configs/accelerate/fsdp2_gradop_config.yaml" ? { num_processes: 8, num_machines: 1 } : null);
  assert.deepEqual(trainerLayout(cmd, config), { gpusPerNode: 8, nodes: 1 });
  assert.equal(
    haloLaunchCommand(cmd, "examples/d.yaml", METHODS),
    "halo launch preference/dpo examples/d.yaml -a launcher-configs/accelerate/fsdp2_gradop_config.yaml",
  );
});

test("a script the CLI does not index gets no halo launch line", () => {
  const [cmd] = parseLaunchCommands(["torchrun --nproc_per_node=8 scripts/training/classification.py <this config>"]);
  assert.equal(haloLaunchCommand(cmd, "examples/c.yaml", METHODS), null);
});

test("glob stars stay within one path segment", () => {
  assert.ok(globToRegExp("examples/sft/qwen3/*").test("examples/sft/qwen3/a.yaml"));
  assert.ok(!globToRegExp("examples/sft/qwen3/*").test("examples/sft/qwen3/deep/a.yaml"));
  assert.ok(globToRegExp("examples/*/qwen3_5/*9b*").test("examples/reward/qwen3_5/rm-qwen3.5-9b.yaml"));
});

const MATRIX = `## Compatibility matrix

| Model Family | \`model_type\` | EP | ETP | CP | TP | PP | Training Configs |
|---|---|:--:|:--:|:--:|:--:|:--:|:--|
| [Qwen3.5 dense](qwen3_5.md) | \`qwen3_5_text\` | — | — | No | Yes | Yes ⁴ | \`examples/*/qwen3_5/*9b*\` |
| [Qwen3.5 MoE](qwen3_5.md) | \`qwen3_5_moe\` | Yes | Yes | No | Yes ³ | Yes ⁴ | \`examples/sft/qwen3_5/*-a*b-*\` |
| Any other HF model | — | — | — | No | Yes | No | — |

## MoE knobs
`;

test("the matrix parses names, pages, footnote-free verdicts and drops rows without recipes", () => {
  const families = parseCompatibilityMatrix(MATRIX);
  assert.deepEqual(
    families.map((f) => [f.name, f.page, f.modelType, f.support.PP]),
    [
      ["Qwen3.5 dense", "agent-docs/models/qwen3_5.md", "qwen3_5_text", "Yes"],
      ["Qwen3.5 MoE", "agent-docs/models/qwen3_5.md", "qwen3_5_moe", "Yes"],
    ],
  );
});

test("families match exactly first, then by folder and file pattern under another method", () => {
  const families = parseCompatibilityMatrix(MATRIX);
  assert.equal(matchFamily("examples/sft/qwen3_5/qwen3.5-35b-a3b-ultrachat-ep.yaml", families).name, "Qwen3.5 MoE");
  assert.equal(matchFamily("examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-aime-full-ep4.yaml", families).name, "Qwen3.5 MoE");
  assert.equal(matchFamily("examples/preference/qwen3_5/dpo-qwen3.5-9b-tulu3.yaml", families).name, "Qwen3.5 dense");
  assert.equal(matchFamily("examples/grpo/environmental/environmental-grpo-template.yaml", families), null);
});

test("a missing matrix section fails loudly", () => {
  assert.throws(() => parseCompatibilityMatrix("# Models\n"), /Compatibility matrix/);
});

const checkout = process.env.HALO_REPO_DIR;
test("the real checkout indexes every recipe with a model, a method and a launch command", { skip: !checkout || !existsSync(checkout) }, () => {
  const { recipes, warnings } = extractRecipes(checkout, { githubBlob: (p) => p });
  assert.ok(recipes.length > 0);
  assert.equal(new Set(recipes.map((r) => r.id)).size, recipes.length, "recipe ids are unique");
  for (const r of recipes) {
    assert.ok(r.model.id, `${r.path} names a model`);
    assert.ok(r.launch.commands.length, `${r.path} has a launch command`);
    assert.ok(r.trainerGpus > 0, `${r.path} has a GPU count`);
  }
  assert.ok(warnings.every((w) => w.includes("no compatibility-matrix family")), warnings.join("\n"));
});
