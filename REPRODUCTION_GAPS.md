# Reproduction gaps and engineering substitutions

This repository intentionally distinguishes paper-defined behavior from lightweight substitutions.

## Paper-defined and retained

- Seven metadata fields: `Name`, `Summary`, `Home-page`, `Author`, `Author-Email`, `License`, `Description`.
- Offline sensitive-API workflow: API call graphs -> six centralities -> corpus aggregation/ranking -> Top-500 -> LLM verification.
- The six centralities are Degree, Closeness, Harmonic, Katz, Eigenvector, and Betweenness. The offline implementation adds 1 to every centrality value before corpus aggregation, matching the method text.
- Suspicious code contexts are centered on sensitive API occurrences and capped at 4096 tokens; identical snippets are deduplicated.
- Package code representations are mean-pooled over snippet representations.
- Stage-2 contract: `Vm[768]`, `Vc[768]`, `Vg[1024]` -> zero-pad the first two to 1024 -> per-view L2 normalize -> reshape each to 32x32 -> stack to `F[3,32,32]`.
- Stage-3 dimensions: 3 convolution layers with 64/128/256 channels and 2x2 max pooling, then channel attention with shared MLP 256->16->256, then a 4096->4096->2 classification MLP.

## Lightweight substitutions

- `all-mpnet-base-v2` is replaced by 32 handcrafted metadata signals + 736 stateless hashed text features.
- `longcoder-base` is replaced by AST structure + API/security-category features + hashed code tokens (256 + 256 + 256 = 768).
- 4-layer GAT graph encoders are replaced by deterministic graph statistics hashed to `Gcg[512]` and `Gcfg[512]`.
- Selection Feature Fusion is replaced by direct concatenation `Gcg || Gcfg -> Vg[1024]`.
- Paper uses DeepSeek-R1 and GPT-5 for verification; this build supports the user's DeepSeek account through an OpenAI-compatible endpoint/model configured via environment variables.

## Underspecified / not claimed as paper-exact

- The paper does not publish the initial GAT node-feature construction, so the GAT is not reproduced here.
- The paper does not fully specify how LongCoder sequence states become a 768-D snippet vector; the lightweight AST/API replacement avoids guessing.
- The paper constrains Katz alpha but does not publish a concrete alpha. The centrality helper chooses a safe alpha from the graph spectral radius.
- The paper text does not explicitly state convolution activation; this repository uses ReLU after each convolution as an engineering choice.
- This environment does not bundle Scalpel. `StaticAnalysisAdapter` exposes a strict Scalpel integration boundary and provides a runnable AST fallback. A version-specific Scalpel adapter should replace it for closer reproduction.
- The paper reports a 293-API final list but does not publish that complete list in the article. This repository does not fabricate it.
