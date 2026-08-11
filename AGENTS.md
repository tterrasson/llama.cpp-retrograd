# Agent notes — Retroback fork of llama.cpp

This checkout is **not** upstream llama.cpp. It is the long-lived fork
`tterrasson/llama.cpp-retroback`, vendored as a git submodule of the Retroback
workspace at `crates/retroback-ffi/runtime/vendor/llama.cpp`.

Upstream's own `AGENTS.md` was replaced by this file on purpose: its
instructions (contributing guide, PR conventions, CI expectations) describe a
repository this one is not.

**Read [`RETRO_FORK.md`](RETRO_FORK.md) before changing anything here.** It
says where the fork is allowed to put code, which upstream files it touches and
why, and what the commit history is supposed to look like. The short version:

- one commit per patch family, named `retro(<family>): <what>`;
- never delete an upstream file;
- additive code goes in a fork-owned file, reached from upstream's dispatcher
  by one line;
- every hunk in an upstream file carries a `// retro delta:` comment saying why.

Build, test and rebase instructions live in the Retroback workspace, not here:
`CLAUDE.md`, `docs/tests/notice.md` and `docs/LLAMA_CPP_UPSTREAM_REBASE.md` at
the workspace root. Do not run raw `cmake`/`ctest` invocations to validate a
change — use the lane scripts under `scripts/`.
