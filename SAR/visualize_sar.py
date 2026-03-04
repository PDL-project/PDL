"""
visualize_sar.py — SAR PDL 실행 시각화 + 동영상 변환

Usage:
    cd /home/nuc/Desktop/PDL/SAR
    python visualize_sar.py --scene 1 --agents 3 --seed 42 --fps 4 --task-idx 0
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_SAR_ROOT    = Path(__file__).resolve().parent
_PDLSAR_ROOT = _SAR_ROOT / "PDL_SAR"

for _p in [str(_SAR_ROOT), str(_PDLSAR_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# -----------------------------------------------------------------------
# 색상
# -----------------------------------------------------------------------

# 셀 배경색
_CELL = {
    "fire_L":  (255, 220,   0),   # LOW
    "fire_M":  (255, 130,   0),   # MED
    "fire_H":  (220,  30,   0),   # HIGH
    "fire_X":  (140,   0,   0),   # MAX
    "res_A":   (160, 110,  50),   # Sand reservoir
    "res_B":   (  0, 160, 190),   # Water reservoir
    "deposit": ( 80,  80,  80),
    "bg":      (215, 215, 218),   # Empty cell
}

# 원형 마커색 (agent, person)
_CIRC = {
    "agent":   ( 30, 144, 255),
    "person":  (123,  31, 162),
    "carried": (194,  24,  91),
}

# Flammable intensity.value → (cell_key, label)
# NONE=1 (extinguished) → skip
_FIRE_IV = {2: ("fire_L", "L"), 3: ("fire_M", "M"), 4: ("fire_H", "H")}


def _cell_rgb(key):
    r, g, b = _CELL[key]
    return r / 255., g / 255., b / 255.


def _circ_rgb(key):
    r, g, b = _CIRC[key]
    return r / 255., g / 255., b / 255.


# -----------------------------------------------------------------------
# 프레임 저장 함수
# -----------------------------------------------------------------------

def _make_save_frame(sar_env, frame_dir: Path, executor_ref: list = None):
    """
    sar_env.save_frame 을 교체할 클로저를 반환합니다.
    executor_ref: [executor] 형태의 리스트 — executor.run() 후에 값이 채워짐.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Circle, FancyBboxPatch
    import numpy as np
    from core import Flammable, AbsAgent, Reservoir, Deposit, Person, Coordinate

    # 테마
    _FIG_BG   = "#f4f6f8"   # 밝은 청회색 배경
    _PANEL_BG = "#ffffff"   # 흰 정보 패널
    _TEXT     = "#212121"   # 기본 텍스트
    _MUTED    = "#757575"   # 보조 텍스트
    _BLUE     = "#1565c0"   # 에이전트 / 섹션 헤더
    _AMBER    = "#e65100"   # 서브태스크 헤더
    _GREEN    = "#2e7d32"   # 성공
    _RED      = "#c62828"   # 실패

    def _fire_short(name: str, lbl: str) -> str:
        """'CaldorFire_Region_1' → 'Caldor\\n[L]'"""
        parts = name.split("_Region_")
        if len(parts) == 2:
            base = parts[0].replace("Fire", "").strip("_")[:7]
            return f"{base}\n[{lbl}]"
        return f"{name[:8]}\n[{lbl}]"

    def _save_frame_impl():
        frame_dir.mkdir(parents=True, exist_ok=True)

        step_num = sar_env.step_num[0]
        objects  = sar_env.controller.field.all_objects(expand=True, with_memory=False)

        W = Coordinate.WIDTH
        H = Coordinate.HEIGHT

        # ── Phase 1: 격자 배경 구성 ──────────────────────────────
        grid_rgb  = np.full((H, W, 3), list(_cell_rgb("bg")))
        cell_text = {}   # (row, col) -> list of (text, fg_color)
        agent_cells = set()  # (x, y) cells that have agents

        for ob in objects:
            x, y, z = ob.position.get()
            if isinstance(ob, AbsAgent) and 0 <= x < W and 0 <= y < H:
                agent_cells.add((x, y))

        for ob in objects:
            x, y, z = ob.position.get()
            if not (0 <= x < W and 0 <= y < H):
                continue

            if isinstance(ob, Flammable):
                if ob.name is None:   # 이름 없는 잠재적 fire region 셀은 건너뜀
                    continue
                iv = ob.intensity.value
                if iv not in _FIRE_IV:
                    if iv > 4:   # MAX
                        grid_rgb[y, x] = list(_cell_rgb("fire_X"))
                        cell_text.setdefault((y, x), []).append(
                            (_fire_short(ob.name, "X"), "white"))
                    # iv == 1 → NONE (extinguished) → 배경 유지
                    continue
                ckey, lbl = _FIRE_IV[iv]
                grid_rgb[y, x] = list(_cell_rgb(ckey))
                text_col = "#1a1a1a" if ckey in ("fire_L", "fire_M") else "white"
                cell_text.setdefault((y, x), []).append(
                    (_fire_short(ob.name, lbl), text_col))

            elif isinstance(ob, Reservoir):
                key   = "res_A" if ob.type == "A" else "res_B"
                rtype = "Sand"  if ob.type == "A" else "Water"
                grid_rgb[y, x] = list(_cell_rgb(key))
                rname = ob.name.replace("Reservoir", "").strip("_")[:8] or ob.name[:8]
                cell_text.setdefault((y, x), []).append(
                    (f"{rname}\n{rtype}", "white"))

            elif isinstance(ob, Deposit):
                grid_rgb[y, x] = list(_cell_rgb("deposit"))
                cell_text.setdefault((y, x), []).append(("DEP", "white"))

        # ── Phase 2: Figure 설정 ─────────────────────────────────
        fig = plt.figure(figsize=(16, 9), facecolor=_FIG_BG)
        gs  = fig.add_gridspec(
            1, 2, width_ratios=[3, 1],
            left=0.03, right=0.97, bottom=0.04, top=0.93, wspace=0.04
        )
        ax_g = fig.add_subplot(gs[0])
        ax_i = fig.add_subplot(gs[1])
        ax_g.set_facecolor("white")
        ax_i.set_facecolor(_PANEL_BG)

        # ── Phase 3: imshow ──────────────────────────────────────
        ax_g.imshow(grid_rgb, aspect="equal", interpolation="nearest", origin="upper")

        # 격자 선
        ax_g.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax_g.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax_g.grid(which="minor", color="#999999", linewidth=0.6, alpha=0.45)

        # 좌표 축 레이블
        ax_g.set_xticks(np.arange(0, W))
        ax_g.set_yticks(np.arange(0, H))
        ax_g.set_xticklabels(range(W), fontsize=7, color="#555555")
        ax_g.set_yticklabels(range(H), fontsize=7, color="#555555")
        ax_g.tick_params(which="minor", length=0)
        ax_g.tick_params(which="major", length=2, color="#bbbbbb", pad=2)
        ax_g.set_xlim(-0.5, W - 0.5)
        ax_g.set_ylim(H - 0.5, -0.5)

        for sp in ax_g.spines.values():
            sp.set_edgecolor("#cccccc")

        # ── Phase 4: 셀 텍스트 (화재·저장소·보관소) ─────────────
        font_sz = max(6, min(12, 180 // max(W, H)))
        for (row, col), items in cell_text.items():
            txt = "\n".join(t for t, _ in items)
            fg  = items[0][1]
            ax_g.text(col, row, txt, ha="center", va="center",
                      fontsize=font_sz, color=fg,
                      fontweight="bold", clip_on=True,
                      linespacing=1.1)

        # ── Phase 5: 에이전트 → 파란 원 ─────────────────────────
        r_agent = 0.38
        agent_positions = {}

        for ob in objects:
            x, y, z = ob.position.get()
            if not (0 <= x < W and 0 <= y < H):
                continue
            if not isinstance(ob, AbsAgent):
                continue

            agent_positions[ob.name] = (x, y)
            idx = (sar_env.agent_names.index(ob.name)
                   if ob.name in sar_env.agent_names else "?")

            # 파란 원
            ax_g.add_patch(Circle(
                (x, y), r_agent,
                facecolor=_circ_rgb("agent"),
                linewidth=1.8, zorder=4
            ))
            # 에이전트 번호
            ax_g.text(x, y, f"A{idx}",
                      ha="center", va="center",
                      fontsize=font_sz, color="white",
                      fontweight="bold", zorder=5)

        # ── Phase 6: 사람 → 보라/분홍 원 ────────────────────────
        r_person  = 0.30
        r_carried = 0.18

        for ob in objects:
            x, y, z = ob.position.get()
            if not (0 <= x < W and 0 <= y < H):
                continue
            if not isinstance(ob, Person):
                continue
            if ob.deposited:
                continue

            if ob.grabbed:
                # 에이전트에게 들린 상태 → 작은 분홍 원
                ax_g.add_patch(Circle(
                    (x + 0.26, y + 0.26), r_carried,
                    facecolor=_circ_rgb("carried"),
                    linewidth=1.2, zorder=6
                ))
            else:
                # 구조 대상자 → 보라 원
                # 같은 셀에 에이전트 있으면 우측으로 오프셋
                cx = x + 0.22 if (x, y) in agent_cells else x
                cy = y
                ax_g.add_patch(Circle(
                    (cx, cy), r_person,
                    facecolor=_circ_rgb("person"),
                    linewidth=1.5, zorder=4
                ))
                ax_g.text(cx, cy, "P",
                          ha="center", va="center",
                          fontsize=max(4, font_sz - 1), color="white",
                          fontweight="bold", zorder=5)

        # ── Phase 7: 제목 ────────────────────────────────────────
        ax_g.set_title(
            f"  Step {step_num:>3d}   ·   Scene {sar_env.scene}"
            f"   ·   {sar_env.num_agents} Agents  ",
            fontsize=12, fontweight="bold", pad=8, color=_BLUE,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", alpha=1.0, edgecolor="#90caf9")
        )

        # ── Phase 8: 정보 패널 ───────────────────────────────────
        ax_i.axis("off")
        ax_i.add_patch(FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.01",
            facecolor=_PANEL_BG, edgecolor="#c5cae9",
            linewidth=1.5, transform=ax_i.transAxes, clip_on=False, zorder=0
        ))

        def _txt(y, s, color=_TEXT, size=8, weight="normal", indent=0.05):
            ax_i.text(indent, y, s, transform=ax_i.transAxes,
                      va="top", ha="left", fontsize=size,
                      color=color, fontfamily="monospace", fontweight=weight)

        def _section(y, title, color=_BLUE):
            # 섹션 헤더 배경
            ax_i.add_patch(FancyBboxPatch(
                (0.01, y - 0.030), 0.98, 0.034,
                boxstyle="round,pad=0.002",
                facecolor="#e8eaf6", edgecolor="none",
                transform=ax_i.transAxes, clip_on=True, zorder=1
            ))
            _txt(y, title, color=color, size=8.0, weight="bold", indent=0.04)
            return y - 0.050

        y = 0.97

        # ── AGENTS ──
        y = _section(y, "AGENTS")
        for i, name in enumerate(sar_env.agent_names):
            hist = sar_env.action_history.get(name, [])
            succ = sar_env.action_success_history.get(name, [])
            pos  = agent_positions.get(name)
            pstr = f"({pos[0]:2d},{pos[1]:2d})" if pos else "(?,?)"

            try:
                inv = sar_env.controller.get_inventory(i)
                if isinstance(inv, dict):
                    parts = [f"{k}:{v}" for k, v in inv.items() if v]
                    inv_s = ",".join(parts) if parts else "empty"
                else:
                    inv_s = str(inv) or "empty"
            except Exception:
                inv_s = "?"

            _txt(y, f"A{i}  {name[:12]}", color=_BLUE, size=7.5, weight="bold")
            y -= 0.021
            _txt(y, f"   pos:{pstr}   inv:{inv_s[:16]}", color=_MUTED, size=6.8)
            y -= 0.019

            if hist:
                last = hist[-1][:30]
                ok   = succ[-1] if succ else True
                mark = "v" if ok else "x"
                col  = _GREEN if ok else _RED
                _txt(y, f"   [{mark}] {last}", color=col, size=6.8)
                y -= 0.019

            y -= 0.007

        # ── SUBTASKS ──
        executor = executor_ref[0] if executor_ref is not None else None
        if executor is not None and getattr(executor, "assignment", None):
            y -= 0.006
            y = _section(y, "SUBTASKS", color=_AMBER)
            import re as _re
            n = sar_env.num_agents
            titles = getattr(executor, "subtask_titles", {})
            for sid, rid in sorted(executor.assignment.items()):
                if isinstance(rid, list):
                    rid_s = "+".join(f"A{(r - 1) % n}" for r in rid)
                else:
                    rid_s = f"A{(rid - 1) % n}"
                raw = titles.get(sid, f"ST{sid}")
                short = _re.sub(r"\s*\[.*?\]", "", raw).rstrip(". ")[:22]
                _txt(y, f"   {short:<22}  {rid_s}", color=_AMBER, size=7)
                y -= 0.019

        # ── 범례 ──
        legend_handles = [
            mpatches.Patch(color=_cell_rgb("fire_L"),  label="Fire  LOW"),
            mpatches.Patch(color=_cell_rgb("fire_M"),  label="Fire  MED"),
            mpatches.Patch(color=_cell_rgb("fire_H"),  label="Fire  HIGH"),
            mpatches.Patch(color=_circ_rgb("agent"),   label="Agent  (circle)"),
            mpatches.Patch(color=_cell_rgb("res_A"),   label="Reservoir Sand"),
            mpatches.Patch(color=_cell_rgb("res_B"),   label="Reservoir Water"),
            mpatches.Patch(color=_cell_rgb("deposit"), label="Deposit"),
            mpatches.Patch(color=_circ_rgb("person"),  label="Person  (lost)"),
            mpatches.Patch(color=_circ_rgb("carried"), label="Person  (carried)"),
        ]
        ax_i.legend(
            handles=legend_handles, loc="lower left",
            fontsize=6.2, framealpha=0.9, handlelength=1.2,
            facecolor="white", labelcolor=_TEXT, edgecolor="#c5cae9"
        )

        # ── 저장 ──
        save_path = frame_dir / f"frame_{step_num:04d}.png"
        fig.savefig(str(save_path), dpi=110, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    return _save_frame_impl


# -----------------------------------------------------------------------
# 프레임 → MP4 변환
# -----------------------------------------------------------------------

def frames_to_video(frame_dir: Path, output_path: Path, fps: int = 4):
    import cv2

    pattern = str(frame_dir / "frame_*.png")
    frame_files = sorted(
        glob.glob(pattern),
        key=lambda p: int(Path(p).stem.split("_")[1])
    )
    if not frame_files:
        print(f"[Viz] 프레임 없음: {frame_dir}")
        return

    sample = cv2.imread(frame_files[0])
    if sample is None:
        print(f"[Viz] 첫 프레임 읽기 실패: {frame_files[0]}")
        return
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for fpath in frame_files:
        img = cv2.imread(fpath)
        if img is not None:
            writer.write(img)

    writer.release()
    print(f"[Viz] 동영상 저장 완료: {output_path}")
    print(f"      ({len(frame_files)} frames, {fps} fps, {len(frame_files)/fps:.1f}s)")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAR PDL 실행 시각화")
    parser.add_argument("--scene",    type=int, default=1)
    parser.add_argument("--agents",   type=int, default=3)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--fps",      type=int, default=4,
                        help="출력 동영상 FPS (기본: 4)")
    parser.add_argument("--output",   type=str, default="",
                        help="출력 MP4 경로 (기본: SAR 루트 디렉토리)")
    args = parser.parse_args()

    # ── 환경 초기화 ─────────────────────────────────────────────
    from env import SAREnv
    print(f"\n[Viz] SAREnv 초기화  scene={args.scene}, agents={args.agents}, seed={args.seed}")
    sar_env = SAREnv(
        num_agents=args.agents,
        scene=args.scene,
        seed=args.seed,
        save_frames=False,
    )
    sar_env.reset()
    print(f"  Task: {sar_env.task}")

    frame_dir = sar_env.render_image_path
    print(f"  Frame dir: {frame_dir}")

    # executor_ref: executor.run() 후에 채워지는 컨테이너
    executor_ref = [None]

    # ── save_frame 패치 ──────────────────────────────────────────
    enhanced_sf = _make_save_frame(sar_env, frame_dir, executor_ref)
    sar_env.save_frame  = enhanced_sf
    sar_env.save_frames = True

    # ── step 0 (초기 상태) ───────────────────────────────────────
    print("[Viz] 초기 상태 캡처...")
    enhanced_sf()

    # ── SARExecutor 로드 + 실행 ──────────────────────────────────
    from sar_executor import SARExecutor
    executor = SARExecutor(str(_PDLSAR_ROOT))
    executor.run(
        task_idx=args.task_idx,
        task_name="task",
        task_description=sar_env.task,
        num_agents=sar_env.num_agents,
    )

    all_object_names = list(sar_env.controller.all_names)
    executor.set_object_names(all_object_names)

    # 이후 프레임에 서브태스크 정보 반영
    executor_ref[0] = executor

    print("\n[Viz] Task Assignment:")
    for sid, rid in sorted(executor.assignment.items()):
        print(f"  Subtask {sid} -> Robot {rid}")
    print(f"  Parallel Groups: {executor.parallel_groups}")

    print("\n[Viz] 실행 시작...")
    results = executor.execute_in_sar(sar_env)

    # ── 결과 출력 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[PDL-SAR] Execution Summary")
    print("=" * 60)
    n_ok = sum(1 for r in results.values() if r.success)
    print(f"  Succeeded: {n_ok} / {len(results)}")
    for sid, r in sorted(results.items()):
        status = "SUCCESS" if r.success else f"FAIL ({r.error_message})"
        print(f"    Subtask {sid}: {status}")

    checker = sar_env.checker
    if checker is not None:
        try:
            coverage       = checker.get_coverage()
            transport_rate = checker.get_transport_rate()
            finished       = checker.check_success()
            balance        = executor._compute_balance_metric(finished=finished)
            exec_rate      = executor._compute_exec_rate()
            print(
                f"\n  Coverage:{coverage:.3f}, Transport:{transport_rate:.3f}, "
                f"Finished:{finished}, Balance:{balance:.3f}, Exec:{exec_rate:.3f}"
            )
        except Exception as e:
            print(f"  [Metrics] {e}")
    print("=" * 60)

    # ── 동영상 조합 ──────────────────────────────────────────────
    out_path = args.output
    if not out_path:
        out_path = str(
            _SAR_ROOT / f"sar_scene{args.scene}_a{args.agents}_s{args.seed}.mp4"
        )

    frames_to_video(frame_dir, Path(out_path), fps=args.fps)


if __name__ == "__main__":
    main()
