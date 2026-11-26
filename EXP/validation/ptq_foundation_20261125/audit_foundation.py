"""Reconcile the attributed static catalog, measured source-version requests and source manifest."""

from pathlib import Path
from hashlib import sha256
from collections import Counter
import json
import sys
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
KNOW = ROOT / "QAViT/knowledge/ptq"
CAT = ROOT / "QAViT/docs/methods/index.json"
STATUS = KNOW / "implementation_status.json"
PREFIX = HERE.relative_to(ROOT).as_posix()


def read(p):
    return json.loads(p.read_text())


def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def sha(p):
    return sha256(p.read_bytes()).hexdigest()


def intake():
    manifest = read(HERE / "PACKAGE_MANIFEST.json")
    mapping = read(HERE / "source_mapping.json")
    src = ROOT.parent / "QAViT_PTQ_Foundation_20261125.zip"
    if src.exists():
        assert sha(src) == mapping["input_zip_sha256"]
        with ZipFile(src) as z:
            members = {Path(i.filename).name: i for i in z.infolist() if not i.is_dir()}
            assert sha256(z.read(members["PACKAGE_MANIFEST.json"])).hexdigest() == sha(
                HERE / "PACKAGE_MANIFEST.json"
            )
            expected = {r["name"]: r for r in manifest["files"]}
            assert (
                set(expected)
                == set(mapping["materials"])
                == set(members) - {"PACKAGE_MANIFEST.json"}
            )
            for name, entry in expected.items():
                raw = z.read(members[name])
                assert (
                    len(raw) == entry["bytes"]
                    and sha256(raw).hexdigest()
                    == entry["sha256"]
                    == mapping["materials"][name]["sha256"]
                )
    else:
        assert (
            read(HERE / "asset_ingest.json")["input_zip_sha256"]
            == mapping["input_zip_sha256"]
        )
    for name, entry in mapping["materials"].items():
        for path in entry["destination"]:
            assert (ROOT / path.split("#", 1)[0]).exists(), (name, path)
    ledger = read(KNOW / "source_coverage.json")
    records = ledger["records"]
    counts = dict(Counter(r["coverage_category"] for r in records))
    assert (
        len({r["survey_ref"] for r in records}) == len(records) == 83
        and counts["ptq_material"] == 51
    )
    for r in records:
        if r["coverage_category"].startswith("excluded_"):
            assert set(r) <= {
                "id",
                "survey_ref",
                "coverage_category",
                "primary_urls",
                "reason",
                "title",
            }
    return {
        "input_zip_sha256": mapping["input_zip_sha256"],
        "manifest_sha256": sha(HERE / "PACKAGE_MANIFEST.json"),
        "payload_members_verified": len(mapping["materials"]),
        "survey_records": len(records),
        "dispositions": counts,
        "method_versions": 52,
        "structured_assets": "source_mapping.json",
        "source_manifest": "PACKAGE_MANIFEST.json",
    }, records


# Conditions are tied to a published variant and to the numerical evidence named in that same row.
updates = {
    "h-vit": ("grids.nearest_power_grid / nonlinear_calibration.compute","逐层 W/A 候选，LN 通道尺度按最小 PoT 指数对齐。整型一阶和二阶和、token 输出经原位图导出。固定位置与实例 token 两种策略独立声明。Eq.18 码容疑点保留，参考计算未在 Versal 上测量。"),
    "practicaledge": ("integer_functions.IntegerFunction / nonlinear_calibration.compute","Softmax 范围重定标、uint16 LN 输入及 M 移位、二次 erf 的整数 GELU 进入真实图。记录输入/输出尺度、整数上界和舍入约定。软件累加和最终转换使用宽参考中间量；ARM NEON 原生核/时延未测。"),
    "h-vit": (
        "encodings.fit / grids.nearest_power_grid / nonlinear_calibration.compute",
        "W/A 采用相邻 floor/ceil PoT；LN 输入通道尺度按共同最小指数移位，整型一阶、二阶和后归一化，输出按 token 量化。固定位置或实例输出模式须单列；原 Eq.18 码容歧义见正文。",
    ),
    "practicaledge": (
        "integer_functions.IntegerFunction / nonlinear_calibration.compute",
        "Softmax 指数近似与分子位宽约束、LN uint16 输入和 M=15 左移、GELU 二次 erf 与 int32 乘积范围。软件舍入/范围配置与真实 NEON 时延分列；原表依赖设备和输入分布。",
    ),
    "erq-tpami2025-duq": (
        "paired_weights.erq / paired_weights.tail_partition",
        "Aqer 读取变换后配对输入和常数列。作者分正负尾、按 d_out//20 选列；Wqer 不舍入 bias，剩余列和 bias 联合修正。小模型与独立增广 ridge 对照。论文整行尾规则可单列。",
    ),
    "daquant": (
        "domain_fit.fit / domain_fit.activation",
        "源域独立样本 DGA→SmoothL1，目标域显式配对样本 feature KL，WRC、W/A clipping 及静态或实例范围分别导出。首坐标统计和逐通道统计按作者代码/适配区分；不同域模型与原预算待运行。",
    ),
    "outlier-aware-slicing": (
        "assembled_recipes.assemble / reconstruction.fit_stage",
        "按照样本实际经过的起点与终点观测多块 FP 输出。联合 W 软舍入、A scale、输入及内部 QDrop，指数 beta、最终硬化；单个已知 downsample 架构需配置实际子图边界。",
    ),
    "qwt-v2": (
        "output_correction.fit / integer_affine.compile_state",
        "浮点逐通道仿射与显式含零点的整数折叠两种请求。负斜率、零斜率、舍入和饱和有独立整数对照。需给输出 scale/zp、乘数精度、累加器宽度；原生硬件未计时。",
    ),
    "q-hyvit": (
        "block_search.search / configuration_search.configuration_grid",
        "由真实观察边界的后缀梯度对整个候选的误差逐坐标加权；scale 轮次及对称性/粒度候选在请求中。真实 timm MobileViT 复核 patch 展开/折回、局部与全局融合和整桥评分；其余 hybrid 架构须绑定自己的转换块。",
    ),
    "igq_vit": (
        "encodings.fit_groups / source_allocation.group_schedule",
        "实例动态按范围选非连续组；每一周期按累计交替步数拟合中心，比较同一其余量化图上开关单个激活的任务 KL，在声明的 BOP 成本下精确选择。检测框和实例 mask 版本还依赖相应任务 head 与扰动定义。",
    ),
    "p2-vit": (
        "grids.nearest_power_grid / source_allocation.genetic_weights",
        "W/A 相邻 PoT 候选与真实输入误差相连；W4/8 分支另用 Hessian/Hutchinson + trace×W 扰动、加权预算初始 Pareto、真实输出突变/交叉筛选。小模型使用教师类别，FPGA 排程与部署未测。",
    ),
    "mixa-q": (
        "window_routes.bind / configuration_search.configuration_grid",
        "独立 LN、共享 W、按输入窗口排序/分发/恢复；同一套完整分辨率网格在三个比例间复用。原 Swin detection block 实测两个方向及 shifted mask；仅独立 COCO PTQ 分支，无 COCO 成绩。",
    ),
    "trio-vit": (
        "efficient_topology.channel_migration / efficient_topology.bind_lite",
        "跨真实 DW→PW 的尺度/平移迁移，接线性注意力 KᵀV 分母及缩小的重构。原分母指数 −15…8 含 24 档；固定 16 档是适配。验证覆盖相应结构，部署设备另测。",
    ),
    "psaq-vit": (
        "execution.calibration_generator / calibration_sources.synthesize",
        "校准像素目标与 ViT attention KDE 熵独立计费。真实预训练模型上有界合成、准备、独立 worker 复用。对校准图的更新与推理时网格冻结分别审计。",
    ),
    "clamp-vit": (
        "cyclic_calibration.run / calibration_sources.synthesize",
        "同一个像素池刷新、同 patch 正例/异 patch 负例、量化群体的交替选择，保存随机种子/图像/轨迹。局部邻域正例实验是另一个适配。原搜索全集和文中评测预算待运行。",
    ),
    "lampq": (
        "layer_importance.fisher / allocation_feedback",
        "Fisher、类型校准、初始位宽与相邻增减实际评分均已执行。公式的邻域差与论文解释按配置说明；预算约束和真实参数量随更新记录。",
    ),
    "pomq-vit": (
        "configuration_search.configuration_grid",
        "公开全文 E1–E3 写 L1；上传资料的 L2 描述作为单独归约保留。Q/K/V 实际操作数、概率及 encoder 输出均可观测。平台查表须提供设备/原始测量；当前延迟是软件 QDQ。",
    ),
    "ptq-vit-2021": (
        "rank_search / block_search.search / configuration_search.configuration_grid",
        "真实 pre-Softmax 成对排序和 Pearson 交替搜索，每轮 fold bias；混精分支采用完整图测量核范数加权误差与重量位预算。gamma/theta、对照温度及候选范围需显式给定。",
    ),
    "lrp-qvit": (
        "relevance.prepare / precision.select",
        "分类目标、残差与注意力的正贡献传播，真实张量/Jacobian 传播并记录未分配贡献，按预算分配。原始算子的手工传播和此自动分解分别识别。",
    ),
    "mix-qvit": (
        "relevance.prepare / precision.select",
        "仅独立 PTQ；真实单层扰动的归一化相对损失与相关度分别记录，组合参数 bit 与 MAC/W×A BitOps。原文的 validation 数据许可必须显式给出。",
    ),
    "amp-vit": (
        "region_grids / allocation_feedback.greedy_operands",
        "W→A 顺序依 SQNR(b−1)×ln(numel) 实际降位，重新拟合 A、SymAlign 和 post-GELU 三区域。GELU 用真实后缀梯度加权的 FC 输出误差选择三区域，MSE 为独立消融。原任务和标签许可按配置。",
    ),
    "mptq-vit": (
        "region_grids / allocation_feedback.greedy_operands",
        "早期版 SQ-b/OPT-m 和 W→A 贪心预算，按真实边的元素数/位域处理。与后续 AMP 共用组件；每次降 GELU 位宽重新捕获后缀梯度并搜索三区域。原任务预算及候选依版本区分。",
    ),
}


def catalog(records):
    sources = {v["id"]: v for v in records if v["coverage_category"] == "ptq_material"}
    version = sources["erq-icml2024"]["variants"][0]
    sources[version["id"]] = version
    summary = read(HERE / "numerical/summary.json")
    ix = read(CAT)
    index = {r["literature_id"]: r for r in ix["methods"] if r.get("literature_id")}
    data = read(STATUS)
    native = read(HERE / "native_recipes.json")
    alloc = read(HERE / "allocation_recipes.json")
    component = read(HERE / "component_checks.json")
    assert len(sources) == 52 and set(index) == set(sources) == {
        r["literature_id"] for r in data["methods"]
    }
    for row in data["methods"]:
        k = row["literature_id"]
        src = sources[k]
        entry = index[k]
        slug = "recipe_" + k
        proofs = []
        requests = []
        phases = []
        if k in updates:
            row["entry_point"], row["current_conditions"] = updates[k]

        def sourceproof(path, req, description):
            assert (
                req.get("source_id") == k and req.get("version") and req.get("profile")
            ), (k, path)
            proofs.append(PREFIX + "/" + path)
            requests.append(
                {
                    "receipt": PREFIX + "/" + path,
                    "version": req["version"],
                    "profile": req["profile"],
                }
            )
            phases.append(description)

        if summary.get(slug, {}).get("status") == "passed":
            path = "numerical/" + slug + "/source_request.json"
            sourceproof(path, read(HERE / path), "来源绑定的阶段组合")
        if k in native:
            sourceproof(
                "native_recipes.json#" + k,
                native[k]["source_request"],
                "宿主已有作者分类图",
            )
        if k in alloc:
            sourceproof(
                "allocation_recipes.json#" + k, alloc[k]["request"], "实际任务/候选分配"
            )
        for case, r in component.items():
            req = r.get("source_request", r.get("request", {}))
            if req.get("source_id") == k:
                sourceproof("component_checks.json#" + case, req, "结构或合成与量化")
        for key, filename in (
            ("clamp-vit", "cyclic_images.json"),
            ("mixa-q", "window_graph.json"),
            ("q-hyvit", "hybrid_bridge.json"),
        ):
            if k == key:
                sourceproof(
                    filename,
                    read(HERE / filename)["request"],
                    "多阶段图像更新"
                    if k == "clamp-vit"
                    else "真实 MobileViT 桥接图"
                    if k == "q-hyvit"
                    else "真实 Swin 窗口图",
                )
        for key, alias in (
            ("daquant", "daquant_frozen"),
            ("qwt-v2", "qwt_v2_integer_fold"),
        ):
            if k == key:
                path = "numerical/" + alias + "/source_request.json"
                sourceproof(path, read(HERE / path), "额外冻结状态")
        assert requests, (k, phases)
        if k == "biascomp":
            proofs.extend(
                PREFIX + "/" + v + ".json"
                for v in ("focus_measurement", "focus_cold", "memory_restore")
            )
        tests = sorted(
            set(
                [
                    t
                    for t in row.get("tests", [])
                    if summary.get(t, {}).get("status") == "passed"
                ]
                + ([slug] if slug in summary else [])
            )
        )
        proofs.extend(PREFIX + "/numerical/" + t + "/request.json" for t in tests)
        extent = (
            " / ".join(dict.fromkeys(phases)) + "；缩小预算，已执行、保存、加载并前向"
        )
        evidence_asset = entry["document"]
        assert (ROOT / "QAViT/docs/methods" / evidence_asset).is_file()
        row.update(
            implementation_status=extent,
            public_entry="execution.recipe.source_id = " + k,
            read_asset={"asset": "methods/" + evidence_asset},
            source_version={
                "origin": src.get("title"),
                "author_code_status": src.get("code_status"),
                "read_scope": src.get("read_scope"),
                "primary_urls": src.get("primary_urls"),
            },
            source_requests=requests,
            verification=list(dict.fromkeys(proofs)),
            tests=tests,
            target_protocol="由实际 W/A 边、码容、固定参数、动态路由、额外计算、结构和例外验证，来源 ID 不授予协议达标状态。",
        )
        entry.update(
            implementation_status=extent,
            current_conditions=row["current_conditions"],
            implementation=row["entry_point"],
            validation_cases=tests,
        )
    data["evidence_scope"] = (
        "51 条主记录和 1 个版本均有绑定来源的有界软件调用及重载。每行按实际阶段标记。独立算子、原作者图、QAViT 适配、协议覆盖、原预算精度和原生设备实测分别登记。"
    )
    return ix, data


if __name__ == "__main__":
    evidence, records = intake()
    ix, status = catalog(records)
    if "--update" in sys.argv:
        save(CAT, ix)
        save(STATUS, status)
        save(HERE / "asset_ingest.json", evidence)
    else:
        assert read(STATUS) == status and read(CAT) == ix
    print(
        "verified",
        evidence["payload_members_verified"],
        "supplied assets,",
        len(records),
        "records,",
        len(status["methods"]),
        "source versions",
    )
