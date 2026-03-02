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
import types
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_SAR_ROOT    = Path(__file__).resolve().parent          # /PDL/SAR
_PDLSAR_ROOT = _SAR_ROOT / "PDL_SAR"

for _p in [str(_SAR_ROOT), str(_PDLSAR_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# -----------------------------------------------------------------------
# 색상 정의
# -----------------------------------------------------------------------
_COLORS = {
    "fire_1" : (255, 215,   0),   # Low  — gold
    "fire_2" : (255, 140,   0),   # Med  — orange
    "fire_3" : (255,  60,   0),   # High — red-orange
    "fire_4" : (200,   0,   0),   # Max  — dark red
    "agent"  : ( 30, 144, 255),   # Dodger blue
    "res_A"  : (139,  90,  43),   # Saddle brown  (Sand)
    "res_B"  : (  0, 180, 200),   # Dark turquoise (Water)
    "deposit": ( 60,  60,  60),   # Dark gray
    "person" : (160,  32, 240),   # Purple
    "carried": (255, 105, 180),   # Hot pink
    "bg"     : (220, 220, 220),   # Light gray
}


def _rgb01(key: str):
    r, g, b = _COLORS[key]
    return (r / 255.0, g / 255.0, b / 255.0)


# -----------------------------------------------------------------------
# 프레임 저장 함수
# -----------------------------------------------------------------------

def _make_save_frame(sar_env, frame_dir: Path):
    """
    sar_env.save_frame 을 교체할 클로저를 반환합니다.
    매 step 후 matplotlib 으로 격자를 그리고 PNG 로 저장합니다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from core import Flammable, AbsAgent, Reservoir, Deposit, Person, Fire, Coordinate

    def _save_frame_impl():
        frame_dir.mkdir(parents=True, exist_ok=True)

        step_num = sar_env.step_num[0]
        objects  = sar_env.controller.field.all_objects(expand=True, with_memory=False)

        W = Coordinate.WIDTH
        H = Coordinate.HEIGHT

        # 격자 데이터 구성
        grid_rgb    = np.full((H, W, 3), [c / 255.0 for c in _COLORS["bg"]])
        cell_labels = {}

        for ob in objects:
            x, y, z = ob.position.get()
            if not (0 <= x < W and 0 <= y < H):
                continue

            if isinstance(ob, Flammable):
                iv  = ob.intensity.value
                key = f"fire_{min(iv, 4)}"
                grid_rgb[y, x] = _rgb01(key)

            elif isinstance(ob, AbsAgent):
                grid_rgb[y, x] = _rgb01("agent")
                cell_labels.setdefault((y, x), []).append(ob.name[:4])

            elif isinstance(ob, Reservoir):
                key = "res_A" if ob.type == "A" else "res_B"
                grid_rgb[y, x] = _rgb01(key)
                short = ob.name[:6]
                cell_labels.setdefault((y, x), []).append(short)

            elif isinstance(ob, Deposit):
                grid_rgb[y, x] = _rgb01("deposit")
                cell_labels.setdefault((y, x), []).append("DEP")

            elif isinstance(ob, Person):
                if ob.grabbed:
                    grid_rgb[y, x] = _rgb01("carried")
                elif not ob.deposited:
                    grid_rgb[y, x] = _rgb01("person")
                cell_labels.setdefault((y, x), []).append(ob.name[:5])

        # Figure 레이아웃: 격자(왼쪽) + 정보 패널(오른쪽)
        fig = plt.figure(figsize=(14, 9))
        gs  = fig.add_gridspec(
            1, 2, width_ratios=[3, 1],
            left=0.02, right=0.98, bottom=0.02, top=0.94, wspace=0.04
        )
        ax_g = fig.add_subplot(gs[0])
        ax_i = fig.add_subplot(gs[1])

        # 격자 그리기
        ax_g.imshow(grid_rgb, aspect="equal", interpolation="nearest",
                    origin="upper")

        ax_g.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax_g.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax_g.grid(which="minor", color="gray", linewidth=0.4, alpha=0.5)
        ax_g.tick_params(which="both", bottom=False, left=False,
                         labelbottom=False, labelleft=False)
        ax_g.set_xlim(-0.5, W - 0.5)
        ax_g.set_ylim(H - 0.5, -0.5)

        # 셀 Label
        font_sz = max(3, 100 // max(W, H))
        for (row, col), lbls in cell_labels.items():
            txt = "\n".join(lbls)
            ax_g.text(col, row, txt,
                      ha="center", va="center",
                      fontsize=font_sz, color="white",
                      fontweight="bold", clip_on=True)

        ax_g.set_title(
            f"Step {step_num:3d}  |  Scene {sar_env.scene}  |  "
            f"{sar_env.num_agents} Agents",
            fontsize=11, fontweight="bold", pad=6
        )

        ax_i.axis("off")
        lines = []

        lines.append("── Last Actions ──")
        for name in sar_env.agent_names:
            hist = sar_env.action_history.get(name, [])
            succ = sar_env.action_success_history.get(name, [])
            if hist:
                last_act = hist[-1]
                mark     = "✓" if (succ and succ[-1]) else "✗"
                act_str  = last_act[:32]
                lines.append(f"{name[:6]}: {mark} {act_str}")
            else:
                lines.append(f"{name[:6]}: –")

        lines.append("")
        lines.append("── Inventory ──")
        for i, name in enumerate(sar_env.agent_names):
            try:
                inv = sar_env.controller.get_inventory(i)
                if isinstance(inv, dict):
                    inv_parts = [f"{k}:{v}" for k, v in inv.items() if v]
                    inv_str   = ", ".join(inv_parts) if inv_parts else "empty"
                else:
                    inv_str = str(inv)
            except Exception:
                inv_str = "?"
            lines.append(f"{name[:6]}: {inv_str}")

        ax_i.text(0.03, 0.98, "\n".join(lines),
                  transform=ax_i.transAxes,
                  va="top", ha="left",
                  fontsize=7.5, fontfamily="monospace",
                  linespacing=1.5)

        # 범례
        leg = [
            mpatches.Patch(color=_rgb01("fire_1"),  label="Fire (Low)"),
            mpatches.Patch(color=_rgb01("fire_2"),  label="Fire (Med)"),
            mpatches.Patch(color=_rgb01("fire_3"),  label="Fire (High)"),
            mpatches.Patch(color=_rgb01("agent"),   label="Agent"),
            mpatches.Patch(color=_rgb01("res_A"),   label="Reservoir (Sand)"),
            mpatches.Patch(color=_rgb01("res_B"),   label="Reservoir (Water)"),
            mpatches.Patch(color=_rgb01("deposit"), label="Deposit"),
            mpatches.Patch(color=_rgb01("person"),  label="Person"),
            mpatches.Patch(color=_rgb01("carried"), label="Person (Carried)"),
        ]
        ax_i.legend(handles=leg, loc="lower left",
                    fontsize=6.5, framealpha=0.85, handlelength=1.5)

        # 저장
        save_path = frame_dir / f"frame_{step_num:04d}.png"
        fig.savefig(str(save_path), dpi=100, bbox_inches="tight")
        plt.close(fig)

    return _save_frame_impl


# -----------------------------------------------------------------------
# 프레임 → MP4 변환
# -----------------------------------------------------------------------

def frames_to_video(frame_dir: Path, output_path: Path, fps: int = 4):
    """frame_dir 내 frame_NNNN.png 파일을 모아 MP4 동영상으로 조합합니다."""
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

    for i, fpath in enumerate(frame_files):
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
    parser.add_argument("--scene",    type=int,   default=1)
    parser.add_argument("--agents",   type=int,   default=3)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--task-idx", type=int,   default=0)
    parser.add_argument("--fps",      type=int,   default=4,
                        help="출력 동영상 FPS (기본: 4)")
    parser.add_argument("--output",   type=str,   default="",
                        help="출력 MP4 경로 (기본: SAR 루트 디렉토리)")
    parser.add_argument("--keep-frames", action="store_true",
                        help="동영상 생성 후 PNG 프레임 파일 유지 (기본: 유지)")
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

    # ── 프레임 디렉토리 결정 ────────────────────────────────────
    frame_dir = sar_env.render_image_path  # render/{n}_agents/seed_{s}/scene_{sc}/
    print(f"  Frame dir: {frame_dir}")

    # ── save_frame 패치 ──────────────────────────────────────────
    enhanced_sf = _make_save_frame(sar_env, frame_dir)
    sar_env.save_frame  = enhanced_sf   # 인스턴스 메서드 교체
    sar_env.save_frames = True          # step() 내 if 분기 활성화

    # ── 초기 상태 프레임 (step 0) ────────────────────────────────
    print("[Viz] 초기 상태 캡처...")
    enhanced_sf()

    # ── SARExecutor 로드 + 실행 ──────────────────────────────────
    from sar_executor import SARExecutor
    sar_pdl_root = str(_PDLSAR_ROOT)
    executor = SARExecutor(sar_pdl_root)
    executor.run(
        task_idx=args.task_idx,
        task_name="task",
        task_description=sar_env.task,
        num_agents=sar_env.num_agents,
    )

    all_object_names = list(sar_env.controller.all_names)
    executor.set_object_names(all_object_names)

    print("\n[Viz] Task Assignment:")
    for sid, rid in sorted(executor.assignment.items()):
        print(f"  Subtask {sid} → Robot {rid}")
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
