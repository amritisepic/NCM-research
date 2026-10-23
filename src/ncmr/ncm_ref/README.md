# Vendored HazyResearch/ncm (MIT)

Unmodified copy of https://github.com/HazyResearch/ncm at the commit in `UPSTREAM_COMMIT`
(see `LICENSE`). Kept byte-identical so reproduction numbers come from the released code.
Our own code imports it through `ncmr.ncm_ref_loader`, which puts this directory on `sys.path`
(the upstream modules use top-level imports such as `from models.common import ...`).

Known upstream issues we work around in our own code, never by editing these files:
- `quant.py: idx_to_zhats` appends aliases of one tensor (upstream issue #11); use `z_levels` or our snapshot helper.
- `eval_shard.py` / `eval_merge.py` hard-code `.cuda()`; our scripts re-implement the same logic device-agnostically.
