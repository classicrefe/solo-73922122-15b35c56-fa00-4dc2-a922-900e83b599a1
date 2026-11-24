"""全局绘图方案的单元测试。

绘图（叠加图与单工序换算图）必须使用同一套“每道工序各一个变换”的完整可行
方案，而不是让每道工序各自挑第一个可行变换——后者在同号定位点只出现在非基准
工序、且各工序源格不同的时候，可能把本应同格的缺陷画到两格。

方案总数、缺陷候选格与待定判定始终以全部可行方案为准，本文件一并守住。
"""
from __future__ import annotations

from wafer_review.geometry import (
    ALL_TRANSFORMS,
    IDENTITY,
    apply_transform,
    transform_label,
)
from wafer_review.model import (
    Analysis,
    Defect,
    Locator,
    Step,
    Wafer,
    analyze,
    compute_feasibility,
)
from wafer_review.overlay import compare_grid

N = 8


def _coupled_wafer() -> Wafer:
    """基准无定位点；工序2/3 只有同号 C1，且两张图上 C1 源格不同。

    恒等 + 恒等并不构成可行方案（C1 会落在两格），因此逐工序各取第一个可行
    变换的旧画法必然违背约束；只有真正成套的全局方案才能让 C1 重合。
    """
    base = Step(1, 1, "基准", [], [])
    s2 = Step(2, 2, "工序2", [Locator(1, "C1", 4, 5)], [Defect(10, "D1", 2, 7)])
    s3 = Step(3, 3, "工序3", [Locator(2, "C1", 5, 4)], [Defect(11, "D1", 6, 3)])
    return Wafer(1, "耦合", N, 0, "", "", [base, s2, s3])


def test_chosen_scheme_is_one_complete_feasible_combination():
    analysis = analyze(_coupled_wafer())
    scheme = analysis.feasibility.chosen_scheme
    assert len(scheme) == 3
    assert scheme[0] == IDENTITY  # 基准固定恒等

    # 逐工序变换都属于该工序在全部方案下的可行投影
    for idx, step in enumerate(analysis.wafer.steps):
        assert scheme[idx] in analysis.transforms(step) or step.is_baseline

    # 成套方案必须让同号定位点 C1 换算后落在同一格
    s2, s3 = analysis.wafer.steps[1], analysis.wafer.steps[2]
    c1_s2 = next(loc for loc in s2.locators if loc.pid == "C1")
    c1_s3 = next(loc for loc in s3.locators if loc.pid == "C1")
    mapped_s2 = apply_transform(scheme[1], c1_s2.row, c1_s2.col, N)
    mapped_s3 = apply_transform(scheme[2], c1_s3.row, c1_s3.col, N)
    assert mapped_s2 == mapped_s3


def test_per_step_first_picks_would_violate_shared_locator():
    """回归保护：旧的“各取第一个可行变换”在本构造下确实不构成方案。"""
    analysis = analyze(_coupled_wafer())
    s2, s3 = analysis.wafer.steps[1], analysis.wafer.steps[2]
    old2, old3 = analysis.transforms(s2)[0], analysis.transforms(s3)[0]
    assert (old2, old3) != (analysis.chosen_transform(s2), analysis.chosen_transform(s3))
    assert apply_transform(old2, 4, 5, N) != apply_transform(old3, 5, 4, N)


def test_overlay_and_mapped_drawing_share_one_global_scheme():
    analysis = analyze(_coupled_wafer())
    s2, s3 = analysis.wafer.steps[1], analysis.wafer.steps[2]

    # 叠加网格里每个非基准工序的 D1 都按全局方案换算，与 chosen_transform 手算一致
    grid = compare_grid(analysis)
    grid_cells = {
        (r, c)
        for r, row in enumerate(grid.rows, start=1)
        for c, cell in enumerate(row, start=1)
        for mark in cell
        if mark.kind == "overlay" and mark.label == "D1"
    }
    expected_cells = {
        apply_transform(analysis.chosen_transform(step), step.defects[0].row, step.defects[0].col, N)
        for step in (s2, s3)
    }
    assert grid_cells == expected_cells

    # 关键回归：全局方案让同号定位点 C1 在叠加坐标中同格（旧的各取第一变换会分在两格）
    c1_s2 = next(loc for loc in s2.locators if loc.pid == "C1")
    c1_s3 = next(loc for loc in s3.locators if loc.pid == "C1")
    c1_cells = {
        apply_transform(analysis.chosen_transform(s2), c1_s2.row, c1_s2.col, N),
        apply_transform(analysis.chosen_transform(s3), c1_s3.row, c1_s3.col, N),
    }
    assert c1_cells == {(4, 5)}

    # 单工序换算取的变换与叠加图层来自同一套方案
    for step in (s2, s3):
        chosen = analysis.chosen_transform(step)
        assert chosen is not None
        layer_cell = apply_transform(chosen, step.defects[0].row, step.defects[0].col, N)
        assert layer_cell in expected_cells


def test_chosen_scheme_is_deterministic_across_analyses():
    w1 = _coupled_wafer()
    w2 = _coupled_wafer()
    assert analyze(w1).feasibility.chosen_scheme == analyze(w2).feasibility.chosen_scheme


def test_demo_wafer_chosen_scheme_matches_planted_transforms():
    from wafer_review.csvparse import parse_csv
    from wafer_review.demo import demo_samples

    steps = []
    for seq, (name, text) in enumerate(demo_samples(), start=1):
        parsed = parse_csv(text, N)
        steps.append(
            Step(
                seq,
                seq,
                name,
                [Locator(i, rec.ident, rec.row, rec.col) for i, rec in enumerate(parsed.locators, 1)],
                [Defect(i, rec.ident, rec.row, rec.col) for i, rec in enumerate(parsed.defects, 1)],
            )
        )
    wafer = Wafer(1, "演示", N, 0, "", "", steps)
    analysis = analyze(wafer)
    # 演示数据各工序“植入”的真实变换（见 demo.DEMO_STEPS）必须是第一套可行方案
    assert analysis.feasibility.chosen_scheme == (
        (False, 0),
        (False, 1),
        (True, 0),
        (False, 2),
        (True, 1),
        (False, 3),
    )
    # 同一修订下反复取，绘图方案不变（切换工序沿用同一套）
    labels = {
        step.id: transform_label(analysis.chosen_transform(step))
        for step in wafer.steps
    }
    assert labels[wafer.steps[5].id] == "顺时针270°"
    again = analyze(wafer)
    assert {
        step.id: again.chosen_transform(step) for step in wafer.steps
    } == {step.id: analysis.chosen_transform(step) for step in wafer.steps}


def test_scheme_count_and_pending_still_consider_all_solutions():
    analysis = analyze(_coupled_wafer())
    # 全部可行方案共 16 套：基准恒等；C1 公共格任选 8 个 D4 落点，每个落点对应
    # 一对满足约束的变换组合（8 × 2 = 16）。
    assert analysis.feasibility.scheme_count == 16
    result = analysis.result(analysis.wafer.steps[2], analysis.wafer.steps[2].defects[0])
    assert result.status == "pending"
    assert len(result.candidates) > 1
    assert result.scheme_count == 16
    # 候选格集合等于该工序所有可行变换的落点集合，而不是只看绘图方案
    projection_cells = {
        apply_transform(t, 6, 3, N) for t in analysis.transforms(analysis.wafer.steps[2])
    }
    assert set(result.candidates) == projection_cells


def test_no_solution_has_empty_chosen_scheme():
    base = Step(1, 1, "基准", [Locator(1, "L1", 1, 1)], [])
    s2 = Step(2, 2, "工序2", [Locator(2, "L1", 2, 2)], [Defect(3, "D1", 3, 3)])
    wafer = Wafer(1, "无解", N, 0, "", "", [base, s2])
    feasibility = compute_feasibility(wafer)
    assert not feasibility.solvable
    assert feasibility.chosen_scheme == ()
    analysis = Analysis(wafer)
    assert analysis.chosen_transform(s2) is None
    # 全部变换仍被枚举，投影为空
    assert set(ALL_TRANSFORMS)  # 枚举顺序本身不受影响
