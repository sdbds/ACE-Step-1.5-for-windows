# ACE-Step 1.5 CLI Guide

This guide explains how to use `cli.py`, the interactive wizard and config-driven CLI for ACE-Step inference.

The CLI is **wizard/config only**: you either run the wizard to build a config, or load a `.toml` config and generate.

---

## Quick Start

Generate via wizard (interactive):

```bash
python cli.py
```

Generate from a saved config:

```bash
python cli.py --config config.toml
```

Create or edit a config without generating:

```bash
python cli.py --configure
python cli.py --configure --config config.toml
```

---

## CLI Flags

- `-c` / `--config` — Path to a `.toml` configuration file to load.
- `--configure` — Run wizard to save configuration without generating.
- `--log-level` — Logging level for internal modules. One of `TRACE`, `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`. Default: `INFO`.

---

## Wizard Flow

1. Choose one of 6 tasks.
2. Select a DiT model (from locally available models, or auto-download).
3. Select an LM model (from locally available models, or auto-download).
4. Provide task-specific inputs (source audio, tracks, etc.).
5. For `text2music`: choose between Simple Mode (auto-generate caption/lyrics via LM) or manual input.
6. Provide caption / description.
7. Choose lyrics mode (instrumental / auto-generate / file / paste).
8. Set number of outputs.
9. Optionally configure advanced parameters (metadata, DiT settings, LM settings, output settings).
10. Review summary and confirm generation.
11. Save configuration to a `.toml` file.

If you skip advanced parameters, the wizard fills **all optional parameters** with defaults from `GenerationParams` and `GenerationConfig`.

---

## Configure Mode (`--configure`)

`--configure` runs the wizard **without generation** and always saves a config.

Behavior:
- If `--config` is provided, the file is loaded and used as the wizard's starting values.
- After the wizard, you choose a filename to save (overwriting or new).
- The program exits without generation.

---

## Configuration File (`.toml`)

The wizard saves a `.toml` file containing all parameters. These keys map directly to the fields used in `cli.py`.

When you load a config with `--config`, all keys are applied to the runtime settings.

---

## Prompt Editing (`instruction.txt`)

When `thinking=True` and a config file is loaded via `--config`, the CLI looks for an `instruction.txt` file in the project root. If found, its contents are used as the pre-loaded formatted prompt for LM audio-token generation, bypassing the interactive editing step.

When running without a config file (wizard mode), the CLI writes the LM's formatted prompt to `instruction.txt` and pauses so you can edit it before audio-token generation proceeds.

This allows fine-tuning the exact prompt (caption, lyrics, metadata) that the LM sees before generating audio codes.
