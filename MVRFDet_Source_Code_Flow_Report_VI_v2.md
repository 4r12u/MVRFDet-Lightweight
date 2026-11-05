# BÁO CÁO SOURCE CODE, STAGE VÀ FLOW LIÊN KẾT — MVRFDet-Lightweight

**Phiên bản báo cáo:** 2.0  
**Ngày:** 2026-09-21  
**Ngôn ngữ:** Tiếng Việt  
**Mục tiêu:** mô tả **code nào gọi code nào**, **dữ liệu đi qua các stage ra sao**, **input/output của từng module**, và **điểm cần kiểm tra khi lỗi**.

---

## 0. Cách đọc báo cáo này

Tài liệu dùng 3 nhãn trạng thái:

- **[CURRENT / CONFIRMED]**: đường gọi hoặc hành vi đã được xác nhận từ traceback, command và cấu trúc project đang sử dụng.
- **[CURRENT / DESIGN]**: module/contract nằm trong bản build lightweight hiện tại theo specification của project.
- **[PLACEHOLDER / BLOCKER]**: file/module đã có nhưng cố ý chưa implement hoàn chỉnh.
- **[TARGET]**: code nên được thêm/sửa ở bước tiếp theo.

Điểm quan trọng nhất hiện tại là `ScalpelAdapter`: pipeline đã đi tới đó nhưng real graph conversion chưa được implement, vì vậy `--backend scalpel` đang dừng có chủ ý bằng `ScalpelIntegrationError`.

---

# 1. SOURCE TREE VÀ VAI TRÒ CỦA TỪNG NHÓM CODE

```text
MVRFDet-Lightweight/
│
├── scripts/
│   ├── extract_features.py          # CLI: package -> Vm/Vc/Vg -> .npz
│   ├── build_sensitive_api_list.py  # CLI offline Stage 0
│   ├── train.py                     # train Stage 3
│   ├── evaluate.py                  # evaluate checkpoint
│   └── predict.py                   # inference package/features
│
├── src/mvrfdet/
│   │
│   ├── pipeline.py                  # ORCHESTRATOR chính của Stage 1
│   │
│   ├── core/
│   │   ├── types.py                 # dataclass / contract nội bộ
│   │   └── constants.py             # dimensions, constants
│   │
│   ├── static_analysis/
│   │   ├── scalpel_adapter.py       # bridge MVRFDet <-> Scalpel
│   │   ├── api_call_graph.py        # model/normalization cho API CG
│   │   └── cfg.py                   # model/normalization cho CFG
│   │
│   ├── stage1/
│   │   ├── metadata/
│   │   │   ├── parser.py            # PKG-INFO -> 7 metadata fields
│   │   │   ├── features.py          # raw metadata -> static features
│   │   │   └── encoder.py           # static features -> Vm[768]
│   │   │
│   │   ├── suspicious_api/
│   │   │   └── locator.py           # load list + match APIs in call graph
│   │   │
│   │   ├── code/
│   │   │   ├── context.py           # source location -> context snippet
│   │   │   ├── ast_features.py      # AST statistics
│   │   │   ├── api_features.py      # API categories / hashed API features
│   │   │   └── encoder.py           # snippets -> Vc[768]
│   │   │
│   │   └── graph/
│   │       ├── api_graph_features.py# API CG -> stats/vector
│   │       ├── cfg_features.py      # CFG -> stats/vector
│   │       ├── encoder.py           # stats -> Gcg[512], Gcfg[512]
│   │       └── fusion.py            # Gcg + Gcfg -> Vg[1024]
│   │
│   ├── stage2/
│   │   └── feature_matrix.py        # Vm/Vc/Vg -> F[3,32,32]
│   │
│   ├── stage3/
│   │   ├── cnn.py                   # F[3,32,32] -> [256,4,4]
│   │   ├── channel_attention.py     # 256 feature maps -> weighted maps
│   │   ├── classifier.py            # flatten -> logits[2]
│   │   └── model.py                 # compose CNN + CAM + classifier
│   │
│   └── offline/sensitive_api/
│       ├── centrality.py
│       ├── aggregate.py
│       ├── ranking.py
│       ├── context.py
│       ├── deepseek.py
│       └── builder.py
│
├── data/
│   ├── sensitive_api/
│   │   ├── candidates_top500.json
│   │   ├── llm_results/
│   │   └── sensitive_apis.json
│   └── cache/
│
└── tests/
```

### Nguyên tắc kiến trúc

`scripts/*` không nên chứa logic feature phức tạp. Script chỉ parse CLI, gọi package trong `src/mvrfdet`, rồi serialize output.

`pipeline.py` là **orchestrator**, không nên chứa thuật toán centrality, AST walker, graph metric hoặc PyTorch layer chi tiết.

Mỗi stage có output contract cố định:

```text
Stage 1 -> Vm[768], Vc[768], Vg[1024]
Stage 2 -> F[3,32,32]
Stage 3 -> logits[2]
```

---

# 2. FLOW THỰC THI HIỆN TẠI ĐÃ ĐƯỢC XÁC NHẬN

Command hiện tại:

```powershell
python scripts\extract_features.py `
  --package ".\dataset\malware\1inch-8.7" `
  --sensitive-list ".\data\sensitive_api\sensitive_apis.json" `
  --output ".\data\cache\1inch-8.7.npz" `
  --backend scalpel
```

Call chain đã được traceback xác nhận:

```text
scripts/extract_features.py
└── main()
    │
    ├── load_sensitive_api_list(args.sensitive_list)
    │     └── stage1/suspicious_api/locator.py
    │           └── Path(path).read_text()
    │           └── json.loads(...)
    │
    ├── pipeline = LightweightFeaturePipeline(...)
    │
    └── pipeline.extract(args.package, sensitive)
          └── src/mvrfdet/pipeline.py
              │
              └── self.analyzer.extract_api_call_graph(package_path)
                    └── static_analysis/scalpel_adapter.py
                        │
                        └── _extract_with_scalpel(package_path)
                              │
                              └── CURRENT BLOCKER:
                                  raise ScalpelIntegrationError(...)
```

### Vị trí lỗi hiện tại

```text
CLI OK
  ↓
import mvrfdet OK
  ↓
load sensitive_apis.json OK
  ↓
pipeline.extract() OK
  ↓
ScalpelAdapter selected OK
  ↓
real Scalpel graph conversion NOT IMPLEMENTED  ← BLOCKER
```

Điều này rất quan trọng khi bảo trì: hiện tại không cần debug CNN, CAM, metadata encoder hay `.npz`; execution chưa đi tới các bước đó.

---

# 3. `scripts/extract_features.py` — ENTRY POINT CỦA FEATURE EXTRACTION

## 3.1 Trách nhiệm

File này nên chỉ làm 4 việc:

```text
1. parse arguments
2. load sensitive API list
3. tạo LightweightFeaturePipeline
4. save feature bundle
```

## 3.2 Call pattern

Khung logic tương ứng với flow hiện tại:

```python
from mvrfdet.pipeline import LightweightFeaturePipeline
from mvrfdet.stage1.suspicious_api.locator import load_sensitive_api_list


def main():
    args = parse_args()

    sensitive = load_sensitive_api_list(args.sensitive_list)

    pipeline = LightweightFeaturePipeline(
        backend=args.backend,
    )

    features = pipeline.extract(
        args.package,
        sensitive,
    )

    save_features(args.output, features)
```

> Đoạn trên mô tả **control flow** của script. Khi bảo trì source thật, chữ ký constructor/serializer phải lấy theo code repository hiện tại.

## 3.3 Input

```text
--package        đường dẫn package đã unpack
--sensitive-list sensitive_apis.json
--output         .npz output
--backend        scalpel / backend khác nếu có
```

## 3.4 Output mong muốn

`.npz` nên ít nhất chứa:

```text
metadata / Vm -> (768,)
code     / Vc -> (768,)
graph    / Vg -> (1024,)
```

Nên thêm metadata debug:

```text
package_path
backend
matched_api_count
context_count
api_graph_node_count
api_graph_edge_count
cfg_count
feature_version
```

### Lỗi thường gặp tại file này

| Triệu chứng | Nguyên nhân thường gặp |
|---|---|
| `No module named mvrfdet` | project chưa editable-install hoặc `src` chưa trên import path |
| `FileNotFoundError sensitive_apis.json` | path list sai/chưa tạo |
| `JSONDecodeError` | list rỗng hoặc JSON sai syntax |
| `ScalpelIntegrationError` | adapter real conversion chưa implement |
| save `.npz` fail | output directory chưa tồn tại |

---

# 4. `pipeline.py` — ORCHESTRATOR GIỮA CÁC STAGE 1

`pipeline.py` là file quan trọng nhất để hiểu **module nào nối với module nào**.

## 4.1 Input/Output

```python
extract(package_path, sensitive_apis)
```

Input:

```text
package_path: unpacked source root
sensitive_apis: set/list API names
```

Output:

```text
MultiViewFeatures
├── metadata: Vm[768]
├── code:     Vc[768]
└── graph:    Vg[1024]
```

## 4.2 Flow dự kiến trong `extract()`

```text
package_path
   │
   ├──────────────────────┐
   │                      │
   ▼                      ▼
Metadata branch      Static-analysis branch
   │                      │
   │                 extract_api_call_graph()
   │                      │
   │                      ▼
   │                 APICallGraph
   │                      │
   │                 match sensitive APIs
   │                      │
   │                      ▼
   │                suspicious occurrences
   │                     /        \
   │                    /          \
   ▼                   ▼            ▼
Vm[768]          context/snippet     CFG(s)
                       │               │
                       ▼               ▼
                    Vc[768]        graph stats
                                       │
                                  Gcg + Gcfg
                                       │
                                       ▼
                                   Vg[1024]
```

## 4.3 Pseudo-code bảo trì

```python
def extract(self, package_path, sensitive_apis):
    # 1. metadata
    metadata = self.metadata_parser.parse_package(package_path)
    vm = self.metadata_encoder.encode(metadata)

    # 2. global API graph
    api_graph = self.analyzer.extract_api_call_graph(package_path)

    # 3. shared suspicious API localization
    occurrences = self.locator.locate(api_graph, sensitive_apis)

    # 4. code view
    contexts = self.context_extractor.extract_many(
        package_path,
        occurrences,
    )
    vc = self.code_encoder.encode_many(contexts)

    # 5. graph view
    cfgs = [
        self.analyzer.extract_cfg(ctx)
        for ctx in contexts
    ]
    vg = self.graph_encoder.encode(
        api_graph=api_graph,
        cfgs=cfgs,
        suspicious_occurrences=occurrences,
    )

    # 6. contracts
    assert vm.shape == (768,)
    assert vc.shape == (768,)
    assert vg.shape == (1024,)

    return MultiViewFeatures(vm, vc, vg)
```

Đây là code flow nên giữ ổn định kể cả khi thay encoder ở bên trong.

---

# 5. `static_analysis/scalpel_adapter.py` — BRIDGE VỚI SCALPEL

## 5.1 Trạng thái hiện tại

**[PLACEHOLDER / BLOCKER]**

Traceback cho thấy:

```text
extract_api_call_graph(package_path)
    ↓
_extract_with_scalpel(package_path)
    ↓
raise ScalpelIntegrationError(...)
```

Nghĩa là adapter hiện cố ý **không đoán schema output của Scalpel**.

## 5.2 Interface cần ổn định

Downstream không nên nhận object thô của Scalpel.

Adapter phải trả model nội bộ:

```python
@dataclass
class APINode:
    id: str
    name: str
    qualified_name: str | None
    kind: str | None
    file: str | None
    line: int | None


@dataclass
class GraphEdge:
    source: str
    target: str


@dataclass
class APICallGraph:
    nodes: dict[str, APINode]
    edges: list[GraphEdge]
```

Lợi ích:

```text
Scalpel version A ─┐
Scalpel version B ─┼─> Adapter -> APICallGraph -> toàn bộ Stage 1
future backend  ───┘
```

Nếu Scalpel đổi schema, chỉ sửa adapter.

## 5.3 TARGET implementation của Call Graph

Scalpel API cần được gọi theo flow:

```python
from scalpel.call_graph.pycg import CallGraphGenerator

cg_generator = CallGraphGenerator(entry_points, package_root)
cg_generator.analyze()

raw_edges = cg_generator.output_edges()
raw_graph = cg_generator.output()
```

Sau đó:

```text
raw Scalpel nodes/edges
        ↓
normalize node identifier
        ↓
resolve API/FQN
        ↓
attach source location nếu lấy được
        ↓
APICallGraph
```

## 5.4 Entry-point discovery

PyPI package không đảm bảo có `main.py`.

Adapter nên tách riêng:

```python
def discover_entry_points(package_root: Path) -> list[Path]:
    ...
```

Thứ tự engineering hợp lý:

```text
1. console_scripts / entry_points metadata
2. setup.py / pyproject declared modules
3. top-level __init__.py
4. top-level *.py
5. fallback: all *.py
```

Không nên nhét logic này trực tiếp vào `_extract_with_scalpel()` vì sẽ khó test.

## 5.5 Normalization API name

Ví dụ source:

```python
import subprocess as sp
sp.Popen(...)
```

Downstream sensitive list cần thấy:

```text
subprocess.Popen
```

Tương tự:

```python
from subprocess import Popen
Popen(...)
```

cũng phải normalize thành:

```text
subprocess.Popen
```

Nên tách:

```python
normalize_api_name(...)
resolve_aliases(...)
```

và test độc lập.

## 5.6 CFG interface

Graph view cần local CFG từ suspicious context/source region.

Interface nên là:

```python
extract_cfg(source_or_file, line_range=None) -> CFGGraph
```

Nếu snippet 4096-token không parse vì bắt đầu giữa block:

```text
try snippet CFG
    ↓ fail
full-file CFG
    ↓
select nodes in relevant line range
```

Phải log rõ fallback; không được âm thầm đổi semantics.

---

# 6. STAGE 1A — METADATA CODE FLOW

Các file:

```text
stage1/metadata/parser.py
        ↓
stage1/metadata/features.py
        ↓
stage1/metadata/encoder.py
        ↓
Vm[768]
```

## 6.1 `parser.py`

Input:

```text
package root
```

Tìm `PKG-INFO` và lấy đúng 7 field:

```text
Name
Summary
Home-page
Author
Author-Email
License
Description
```

Output nên là dataclass/dict chuẩn hóa:

```python
{
    "Name": "...",
    "Summary": "...",
    "Home-page": "...",
    "Author": "...",
    "Author-Email": "...",
    "License": "...",
    "Description": "...",
}
```

Nếu field thiếu:

```text
không crash
→ dùng empty string / missing indicator
```

## 6.2 `features.py`

Chuyển text/raw metadata thành numeric features.

Ví dụ nhóm feature:

```text
presence flags
length statistics
URL/email validity
GitHub homepage indicator
license indicator
character ratios
word statistics
hashed text representation
```

Điểm bảo trì: raw count lớn không nên để scale tùy ý trước L2 của Stage 2. Nên log-transform hoặc standardize feature nhóm count nếu cần.

## 6.3 `encoder.py`

Nhiệm vụ cuối cùng:

```text
metadata numeric representation
        ↓
fixed-size composition
        ↓
Vm[768]
```

Contract bắt buộc:

```python
assert vm.ndim == 1
assert vm.shape[0] == 768
assert np.isfinite(vm).all()
```

---

# 7. `stage1/suspicious_api/locator.py` — SHARED BRIDGE GIỮA CODE VIEW VÀ GRAPH VIEW

File này đã được xác nhận thực thi qua traceback vì `load_sensitive_api_list()` đã chạy.

## 7.1 Load list

Hiện logic đã xác nhận:

```python
data = json.loads(Path(path).read_text(encoding="utf-8"))
```

Do đó input JSON phải hợp lệ.

Nên hỗ trợ rõ 1 schema duy nhất, ví dụ:

```json
[
  "os.system",
  "subprocess.Popen"
]
```

hoặc schema object versioned:

```json
{
  "version": "v1",
  "apis": ["os.system", "subprocess.Popen"]
}
```

Không nên ngầm hỗ trợ nhiều schema không test.

## 7.2 Match

```text
APICallGraph
+
sensitive API set
        ↓
iterate graph/API call nodes
        ↓
normalize FQN
        ↓
exact/defined matching rule
        ↓
SuspiciousOccurrence[]
```

Một occurrence nên chứa tối thiểu:

```python
@dataclass
class SuspiciousOccurrence:
    api_name: str
    file: str | None
    line: int | None
    column: int | None
    node_id: str | None
```

### Lỗi cần tránh

Không match substring mơ hồ:

```text
"system" in "filesystem"
```

Nên match normalized qualified API name theo rule rõ ràng.

---

# 8. STAGE 1B — CODE VIEW: CONTEXT -> AST/API -> `Vc[768]`

Files:

```text
locator.py
   ↓
code/context.py
   ↓
├── code/ast_features.py
├── code/api_features.py
└── code/encoder.py
        ↓
Vc[768]
```

## 8.1 `context.py`

Input:

```text
SuspiciousOccurrence(file, line, ...)
```

Output:

```text
CodeContextSnippet
```

Mỗi snippet tối đa 4096 token-equivalent theo rule của project.

Cơ chế:

```text
API position
    ↓
center window
    ↓
expand left/right
    ↓
file boundary reached?
    ├─ no -> continue balanced
    └─ yes -> chuyển budget còn lại sang phía kia
```

Nếu 2 APIs có window overlap:

```text
extract độc lập
```

Sau đó:

```text
identical text -> deduplicate trước encode
```

## 8.2 `ast_features.py`

Flow:

```python
source_text
   ↓
ast.parse(source_text)
   ↓
ast.walk(tree)
   ↓
count/statistics
```

Các feature điển hình:

```text
Call
Import / ImportFrom
FunctionDef / AsyncFunctionDef
Assign
Attribute
If
For / While
Try
Raise
Lambda
Subscript
Constant
```

Structural:

```text
node_count
max_depth
call_count
branch_count
loop_count
exception_count
function_count
cyclomatic-like complexity
```

Parse error phải trả status rõ, không được tự động coi vector zero mà không log.

## 8.3 `api_features.py`

Tách API behavior thành category:

```text
network
filesystem
process
shell
dynamic_execution
encoding
serialization
environment
credentials
system_information
```

Có thể có:

```text
category counts
binary presence
sensitive API count
hashed API names
```

## 8.4 `code/encoder.py`

Mỗi snippet:

```text
AST vector
+
API vector
+
lexical/hash vector
      ↓
compose/pad/project
      ↓
Vc_snippet[768]
```

N snippets:

```python
vc = mean([v1, v2, ..., vN], axis=0)
```

Edge case không có suspicious API:

- phải định nghĩa rõ `Vc = zeros(768)` hoặc một no-suspicious sentinel representation;
- rule này phải thống nhất giữa train và inference;
- log `context_count=0`.

---

# 9. STAGE 1C — GRAPH VIEW: API CG + CFG -> `Vg[1024]`

Files:

```text
static_analysis/scalpel_adapter.py
   │
   ├── APICallGraph
   │      ↓
   │  graph/api_graph_features.py
   │      ↓
   │  graph/encoder.py -> Gcg[512]
   │
   └── CFG(s)
          ↓
      graph/cfg_features.py
          ↓
      graph/encoder.py -> Gcfg[512]

Gcg[512] + Gcfg[512]
       ↓
graph/fusion.py
       ↓
Vg[1024]
```

## 9.1 `api_graph_features.py`

Input normalized `APICallGraph`, không dùng raw Scalpel object.

Feature groups:

```text
node_count / edge_count / density
in/out degree statistics
components / SCC / WCC
cycle/path statistics
6 centrality distributions
sensitive API count / ratio
sensitive-to-sensitive edge count
sensitive category histogram
```

NetworkX failures phải xử lý từng metric vì Katz/eigenvector có thể không hội tụ ở một số graph.

Nên trả cả value + diagnostic:

```python
{
    "features": vector,
    "diagnostics": {
        "katz_ok": True,
        "eigenvector_ok": False,
        ...
    }
}
```

## 9.2 `cfg_features.py`

Input: một hoặc nhiều CFG local.

Feature groups:

```text
nodes / edges
entry / exit
branch / merge
loops
components
centrality stats
path stats
cyclomatic complexity
```

Edge cases:

```text
single-node CFG -> self-loop nếu cần cho downstream rule
no valid CFG     -> zero/sentinel Gcfg, nhưng phải log
multiple components -> compute stats + pool
```

## 9.3 `graph/encoder.py`

Static graph feature list có thể ngắn hơn 512, nên encoder phải tạo fixed-size vector.

```text
API graph stats -> Gcg[512]
CFG stats       -> Gcfg[512]
```

Nếu dùng hashing/projection, seed/config phải deterministic.

## 9.4 `graph/fusion.py`

### CURRENT lightweight design

Specification build ban đầu dùng:

```python
vg = concatenate([gcg, gcfg])
```

vì:

```text
512 + 512 = 1024
```

### TARGET option đã được thảo luận

Có thể giữ Selection Feature Fusion (SFF) từ paper vì chi phí nhỏ:

```text
Gcg[512] ──selector A────┐
                         × Gcfg
Gcfg[512] ──selector B────┐
                          × Gcg
          concat selected features
                    ↓
                Vg[1024]
```

**Không được đổi concat -> SFF âm thầm** trong cùng experiment version. Đây là thay đổi representation, phải tăng `feature_version` và regenerate cache.

---

# 10. CORE TYPES — CONTRACT GIỮA CÁC FILE

Khuyến nghị các object trung gian phải typed rõ:

```python
@dataclass
class MultiViewFeatures:
    metadata: np.ndarray  # [768]
    code: np.ndarray      # [768]
    graph: np.ndarray     # [1024]


@dataclass
class APICallGraph:
    nodes: dict[str, APINode]
    edges: list[GraphEdge]


@dataclass
class CodeContextSnippet:
    api_name: str
    file: str
    start_line: int
    end_line: int
    text: str
```

Không truyền dict tự do giữa mọi stage vì khi rename key rất khó truy vết lỗi.

---

# 11. STAGE 2 — `feature_matrix.py`: BA VIEW -> CNN INPUT

Đây là phần phải giữ đặc biệt ổn định.

## 11.1 Input

```text
Vm[768]
Vc[768]
Vg[1024]
```

## 11.2 Code flow

```text
Vm -> pad 256 zeros -> 1024 -> L2 normalize -> reshape 32x32
Vc -> pad 256 zeros -> 1024 -> L2 normalize -> reshape 32x32
Vg ----------------> 1024 -> L2 normalize -> reshape 32x32

stack(channel dimension)
          ↓
F[3,32,32]
```

## 11.3 Reference implementation shape

```python
import numpy as np


def pad_to_1024(x):
    assert x.ndim == 1
    if x.shape[0] == 1024:
        return x
    if x.shape[0] != 768:
        raise ValueError(f"Unexpected feature size: {x.shape}")
    return np.pad(x, (0, 256), mode="constant")


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, ord=2)
    return x / (norm + eps)


def build_feature_matrix(vm, vc, vg):
    vm = l2_normalize(pad_to_1024(vm)).reshape(32, 32)
    vc = l2_normalize(pad_to_1024(vc)).reshape(32, 32)
    vg = l2_normalize(pad_to_1024(vg)).reshape(32, 32)

    f = np.stack([vm, vc, vg], axis=0)
    assert f.shape == (3, 32, 32)
    return f
```

Điểm phải test:

```text
channel 0 = metadata
channel 1 = code
channel 2 = graph
```

Nếu đổi order channel sau khi train, model sẽ nhận semantics khác.

## 11.4 Semantic layout risk

`reshape(1024 -> 32x32)` tạo spatial neighborhood nhân tạo. CNN kernel 3x3 sẽ coi feature gần nhau là local pattern.

Do đó lightweight handcrafted features nên có **feature ordering cố định và có nhóm semantic**, không được đổi index tùy tiện giữa phiên bản.

---

# 12. STAGE 3A — `cnn.py`

Input batch:

```text
[B,3,32,32]
```

Architecture:

```text
Conv2d 3 -> 64, kernel 3, stride 1, padding 1
MaxPool2d 2
        ↓
[B,64,16,16]

Conv2d 64 -> 128, kernel 3, stride 1, padding 1
MaxPool2d 2
        ↓
[B,128,8,8]

Conv2d 128 -> 256, kernel 3, stride 1, padding 1
MaxPool2d 2
        ↓
[B,256,4,4]
```

CNN là vị trí **cross-view mixing đầu tiên** vì Conv1 có weights trên cả 3 input channels.

Concept:

```text
Metadata 3x3 ─┐
Code     3x3 ─┼─ weighted sum -> learned feature map
Graph    3x3 ─┘
```

Không nên diễn giải CAM sau này như weight trực tiếp Metadata/Code/Graph; sau Conv1 identity 3 view đã bị trộn vào learned channels.

---

# 13. STAGE 3B — `channel_attention.py`

Input:

```text
X[B,256,4,4]
```

Flow:

```text
X
├── spatial average pool -> avg[B,256]
└── spatial max pool     -> max[B,256]

avg -> shared MLP 256->16->256 ─┐
max -> shared MLP 256->16->256 ─┤
                                 +
                                 ↓
                              sigmoid
                                 ↓
                         attention[B,256]
                                 ↓
                         reshape [B,256,1,1]
                                 ↓
                         X * attention
```

Reference shape logic:

```python
avg = x.mean(dim=(2, 3))
mx = x.amax(dim=(2, 3))

weights = torch.sigmoid(
    self.mlp(avg) + self.mlp(mx)
)

weights = weights[:, :, None, None]
out = x * weights
```

CAM chọn **256 learned CNN channels**, không phải 3 original views.

---

# 14. STAGE 3C — `classifier.py`

Input:

```text
[B,256,4,4]
```

Flatten:

```text
256 * 4 * 4 = 4096
```

Flow:

```text
[B,256,4,4]
    ↓ flatten
[B,4096]
    ↓ Linear
[B,4096]
    ↓ ReLU
[B,4096]
    ↓ Linear
[B,2]
```

Training nên trả **logits**, không softmax trong `forward()` khi dùng `CrossEntropyLoss`.

```python
logits = model(x)
loss = criterion(logits, labels)
```

Inference:

```python
probs = torch.softmax(logits, dim=1)
pred = logits.argmax(dim=1)
```

---

# 15. `stage3/model.py` — COMPOSITION

Đây là wrapper nối 3 module:

```text
Feature Matrix
   ↓
CNN
   ↓
ChannelAttention
   ↓
Classifier
   ↓
logits
```

Pseudo-code:

```python
class MVRFDetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone()
        self.attention = ChannelAttention(256)
        self.classifier = Classifier()

    def forward(self, x):
        x = self.cnn(x)
        x = self.attention(x)
        logits = self.classifier(x)
        return logits
```

Test contract:

```python
x = torch.randn(2, 3, 32, 32)
logits = model(x)
assert logits.shape == (2, 2)
```

---

# 16. END-TO-END FLOW SAU KHI SCALPEL ADAPTER ĐƯỢC HOÀN THIỆN

```text
scripts/extract_features.py
│
├─ load_sensitive_api_list()
│
└─ LightweightFeaturePipeline.extract()
   │
   ├─ metadata/parser.py
   │    └─ read PKG-INFO
   │
   ├─ metadata/features.py
   │
   ├─ metadata/encoder.py
   │    └─ Vm[768]
   │
   ├─ scalpel_adapter.extract_api_call_graph()
   │    ├─ discover_entry_points()
   │    ├─ CallGraphGenerator(...)
   │    ├─ analyze()
   │    ├─ output_edges()/output()
   │    └─ normalize -> APICallGraph
   │
   ├─ suspicious_api/locator.py
   │    └─ SuspiciousOccurrence[]
   │
   ├─ code/context.py
   │    └─ CodeContextSnippet[]
   │
   ├─ code/ast_features.py
   ├─ code/api_features.py
   ├─ code/encoder.py
   │    └─ Vc[768]
   │
   ├─ scalpel_adapter.extract_cfg(...)
   │    └─ CFG[]
   │
   ├─ graph/api_graph_features.py
   ├─ graph/cfg_features.py
   ├─ graph/encoder.py
   │    ├─ Gcg[512]
   │    └─ Gcfg[512]
   │
   └─ graph/fusion.py
        └─ Vg[1024]

save .npz
   │
   ├─ Vm
   ├─ Vc
   └─ Vg

Later training/inference:
.npz
  ↓
feature_matrix.py
  ↓
F[3,32,32]
  ↓
stage3/model.py
  ├─ cnn.py
  ├─ channel_attention.py
  └─ classifier.py
  ↓
logits[2]
```

---

# 17. STAGE 0 OFFLINE — CODE FLOW TẠO `sensitive_apis.json`

Stage này tách khỏi online extraction.

```text
scripts/build_sensitive_api_list.py
    ↓
offline/sensitive_api/builder.py
    │
    ├─ for each malicious package
    │    └─ ScalpelAdapter.extract_api_call_graph()
    │
    ├─ centrality.py
    │    └─ 6 centrality metrics
    │
    ├─ aggregate.py
    │    └─ group by normalized API name
    │
    ├─ ranking.py
    │    └─ top 500
    │
    ├─ context.py
    │    └─ representative contexts
    │
    ├─ deepseek.py
    │    └─ prompt + API call + cache
    │
    └─ write sensitive_apis.json
```

### `centrality.py`

Nên có API kiểu:

```python
compute_six_centralities(graph) -> dict[node_id, CentralityVector]
```

Vector:

```text
[degree, closeness, harmonic, katz, eigenvector, betweenness]
```

Nếu theo reproduction paper, cộng 1 vào values theo rule đã chốt.

### `aggregate.py`

```text
node results từ nhiều package
        ↓
normalized API name
        ↓
mean centrality vector / aggregate score
```

### `ranking.py`

Không phụ thuộc LLM. Chỉ rank và lấy top 500.

### `deepseek.py`

Phải:

```text
read key từ environment
cache raw response
cache parsed label
không gọi lại nếu cache hợp lệ
```

Không bao giờ gọi DeepSeek trong `pipeline.extract()` online.

---

# 18. TRAIN FLOW

Sau khi dataset đã có cached `.npz`:

```text
feature cache dataset
      ↓
DataLoader
      ↓
for each batch:
    Vm/Vc/Vg
      ↓
FeatureMatrixGenerator
      ↓
[B,3,32,32]
      ↓
MVRFDetClassifier
      ↓
[B,2]
      ↓
CrossEntropyLoss
      ↓
backward
      ↓
Adam.step()
```

Chỉ Stage 3 được train trong lightweight static-representation mode, trừ khi sau này đưa trainable projection/SFF vào Stage 1.

Nếu SFF có trainable Linear layers, phải quyết định rõ nó thuộc:

```text
feature extraction checkpoint
```

hay:

```text
end-to-end model checkpoint
```

Không được cache `Vg` trước một trainable SFF rồi kỳ vọng SFF vẫn train được.

---

# 19. CƠ CHẾ VERSION CACHE — RẤT QUAN TRỌNG KHI BẢO TRÌ

Mỗi cache nên có fingerprint:

```json
{
  "feature_version": "0.2.1",
  "backend": "scalpel",
  "sensitive_api_version": "fixture-v1",
  "metadata_encoder_version": "static-v1",
  "code_encoder_version": "ast-api-v1",
  "graph_encoder_version": "stats-concat-v1"
}
```

Nếu đổi một trong các thứ sau, cache phải invalidate:

```text
API normalization rule
sensitive API list
context extraction rule
AST feature order
API category order
Gcg/Gcfg feature order
graph fusion concat <-> SFF
feature matrix channel order
normalization behavior
```

Nếu không version cache, model có thể chạy mà không crash nhưng semantics sai — loại lỗi nguy hiểm nhất.

---

# 20. ERROR MAP THEO MODULE

```text
[1] CLI / environment
    scripts/extract_features.py
    ├─ import error
    ├─ wrong path
    └─ save path

[2] Sensitive list
    suspicious_api/locator.py
    ├─ file missing
    ├─ invalid JSON
    └─ schema mismatch

[3] Static analysis
    scalpel_adapter.py
    ├─ Scalpel import
    ├─ entry point
    ├─ analyze failure
    ├─ output schema change
    └─ graph normalization

[4] Metadata
    metadata/parser.py
    ├─ PKG-INFO not found
    └─ malformed metadata

[5] Code
    context.py / ast_features.py
    ├─ source mapping missing
    ├─ syntax error
    └─ no contexts

[6] Graph
    api_graph_features.py / cfg_features.py
    ├─ empty graph
    ├─ centrality non-convergence
    └─ invalid CFG

[7] Contract
    encoder.py / feature_matrix.py
    ├─ wrong dimensions
    ├─ NaN/Inf
    └─ zero norm

[8] Neural model
    cnn.py / channel_attention.py / classifier.py
    ├─ shape mismatch
    ├─ dtype mismatch
    └─ checkpoint mismatch
```

---

# 21. LOGGING NÊN CÓ CHO MỘT PACKAGE

Lần chạy debug nên sinh log tương tự:

```text
[package]
root=.../1inch-8.7
backend=scalpel

[metadata]
pkg_info=FOUND
fields_found=6/7
Vm.shape=(768,)
Vm.finite=True

[call_graph]
entry_points=3
nodes=87
edges=103
external_modules=12

[sensitive_api]
knowledge_size=10
matched_occurrences=4
unique_api_names=3

[contexts]
raw=4
deduplicated=3
parse_failures=0

[code]
Vc.shape=(768,)

[cfg]
requested=3
built=3
fallback_full_file=1

[graph]
Gcg.shape=(512,)
Gcfg.shape=(512,)
Vg.shape=(1024,)

[matrix]
shape=(3,32,32)

[model]
cnn=(1,256,4,4)
attention=(1,256,4,4)
logits=(1,2)
```

Nhờ log này có thể xác định stage hỏng mà không đọc toàn bộ stack trace.

---

# 22. TEST MAP TƯƠNG ỨNG VỚI SOURCE FILE

## Static analysis tests

```text
test_scalpel_call_graph_basic
    a() calls b()
    -> edge a -> b

test_external_api_fqn
    socket.socket()
    -> normalized socket.socket

test_alias_import
    import subprocess as sp
    sp.Popen()
    -> subprocess.Popen

test_from_import
    from subprocess import Popen
    Popen()
    -> subprocess.Popen

test_cfg_if_else
    -> branch topology exists
```

## Stage 1 tests

```text
test_metadata_shape_768
test_code_shape_768
test_graph_shape_1024
test_no_suspicious_api_behavior
test_context_deduplication
```

## Stage 2 tests

```text
test_pad_768_to_1024
test_graph_not_padded
test_l2_norm
test_matrix_shape_3x32x32
test_channel_order
```

## Stage 3 tests

```text
test_cnn_shape
test_cam_shape
test_model_logits_shape
test_backward_pass
```

## End-to-end test

```text
fixture package
   ↓
extract_features
   ↓
Vm/Vc/Vg
   ↓
feature matrix
   ↓
model forward
```

---

# 23. BLOCKER HIỆN TẠI VÀ THỨ TỰ CODE TIẾP THEO

Hiện tại **không nên sửa Stage 2/3 trước** vì command đang dừng ở Scalpel adapter.

Thứ tự sửa:

```text
1. static_analysis/scalpel_adapter.py
   ├─ discover_entry_points
   ├─ CallGraphGenerator
   ├─ analyze
   ├─ read edges/output
   └─ normalize APICallGraph

2. test API CG trên package nhỏ tự tạo

3. test API CG trên 1inch-8.7

4. locator.py
   └─ verify matched APIs + locations

5. context.py
   └─ extract snippets

6. CFG integration

7. Code/Graph encoders

8. Vm/Vc/Vg .npz successfully generated

9. feature_matrix.py

10. Stage 3 forward
```

Definition of done cho blocker Scalpel:

```text
--backend scalpel
        ↓
không còn ScalpelIntegrationError placeholder
        ↓
APICallGraph nodes > 0
APICallGraph edges >= 0
        ↓
normalized graph downstream đọc được
```

---

# 24. FLOW BẢO TRÌ KHI SỬA MỘT MODULE

Ví dụ sửa `api_features.py`:

```text
change API category mapping
        ↓
run unit tests api_features
        ↓
Vc semantics changed
        ↓
bump code_encoder_version
        ↓
invalidate Vc cache
        ↓
regenerate feature matrices
        ↓
retrain model
```

Ví dụ sửa `scalpel_adapter.py` nhưng chỉ fix path handling, graph semantics không đổi:

```text
change adapter path handling
        ↓
run graph acceptance tests
        ↓
compare normalized graph snapshot
        ↓
if identical -> cache may remain valid
if changed   -> invalidate graph/code caches
```

Ví dụ đổi channel order Stage 2:

```text
[metadata, code, graph]
        ↓ changed to
[graph, code, metadata]
```

Đây là breaking change bắt buộc retrain model.

---

# 25. CHECKLIST CHO MỖI RELEASE

```text
[ ] Python version xác định
[ ] dependency lock/version xác định
[ ] Scalpel commit/revision ghi lại
[ ] sensitive_apis.json versioned
[ ] feature ordering versioned
[ ] Vm shape test pass
[ ] Vc shape test pass
[ ] Vg shape test pass
[ ] F[3,32,32] test pass
[ ] CNN shape test pass
[ ] CAM shape test pass
[ ] logits[2] test pass
[ ] one-package end-to-end test pass
[ ] cache fingerprint lưu đầy đủ
[ ] current known reproduction gaps ghi lại
```

---

# 26. TÓM TẮT CODE FLOW QUAN TRỌNG NHẤT

## Feature extraction

```text
extract_features.py
        ↓
pipeline.py
        ├──────────── metadata/* ───────────────> Vm[768]
        │
        └─ scalpel_adapter.py
              ↓
          APICallGraph
              ↓
          locator.py
           /        \
          /          \
     code/*         graph/*
       ↓               ↓
   Vc[768]         Vg[1024]
        \             /
         \           /
          save .npz
```

## Classification

```text
Vm / Vc / Vg
      ↓
feature_matrix.py
      ↓
[3,32,32]
      ↓
cnn.py
      ↓
[256,4,4]
      ↓
channel_attention.py
      ↓
[256,4,4]
      ↓
classifier.py
      ↓
logits[2]
```

## Offline sensitive API build

```text
malicious packages
      ↓
scalpel_adapter.py
      ↓
centrality.py
      ↓
aggregate.py
      ↓
ranking.py
      ↓
Top 500
      ↓
context.py
      ↓
deepseek.py
      ↓
sensitive_apis.json
```

---

# 27. GHI CHÚ QUAN TRỌNG VỀ “CODE HIỆN TẠI” VÀ “CODE MỤC TIÊU”

Phần execution path từ `scripts/extract_features.py` đến `ScalpelIntegrationError` là trạng thái hiện tại đã được xác nhận trực tiếp từ runtime traceback.

Các module downstream `metadata/*`, `code/*`, `graph/*`, `stage2/*`, `stage3/*` được mô tả theo architecture/specification của MVRFDet-Lightweight đã chốt. Khi bảo trì line-by-line, phải đối chiếu source revision thực tế vì interface có thể khác tên class/function nhỏ.

Điểm không được hiểu nhầm:

```text
"module đã có trong source tree"
không đồng nghĩa
"module đã chạy end-to-end thành công"
```

Hiện tại milestone thực tế là:

```text
CLI + package import + sensitive list loading: PASS
Scalpel selection/import boundary: PASS
Scalpel normalized graph conversion: BLOCKED / NOT IMPLEMENTED
Downstream one-package end-to-end: NOT YET REACHED
```

Đây là trạng thái nên dùng làm baseline trước lần sửa tiếp theo.
