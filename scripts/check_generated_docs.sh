#!/usr/bin/env bash
# Fail if a generated doc is stale. EXAMPLES.md and SCRIPTS.md index things that exist on disk;
# hand-editing them means they drift the moment someone adds a probe.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 scripts/gen_env_inventory.py >/dev/null
if ! git diff --quiet -- docs/ENV_INVENTORY.md; then
  echo "docs/ENV_INVENTORY.md is stale — run: python3 scripts/gen_env_inventory.py"
  git checkout -- docs/ENV_INVENTORY.md
  exit 1
fi
python3 scripts/gen_indexes.py --check
python3 scripts/gen_kernel_map.py --check
echo "generated docs are current"
