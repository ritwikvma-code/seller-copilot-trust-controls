"""Architecture diagrams for the Trustworthy AI Copilots paper (Figures 1-4)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
NAVY, LIGHT, ACC, GRAY = "#1F3864", "#E8EDF5", "#C05621", "#666666"
GREEN, RED = "#2F855A", "#C53030"


def box(ax, x, y, w, h, text, fc=LIGHT, ec=NAVY, fs=8.5, bold=False, tc=None, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal",
            color=tc or (("white" if fc == NAVY else "#1a1a1a")))


def arrow(ax, x1, y1, x2, y2, color=NAVY, lw=1.3, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls))


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    return fig, ax


# ---------------- Figure 1: system overview ----------------
fig, ax = canvas(8.6, 5.0)
box(ax, 0.2, 7.6, 1.8, 1.2, "Seller\n(chat / events)", fc="white")
box(ax, 3.4, 7.5, 3.2, 1.5, "Orchestrator LLM\ntask decomposition & routing", fc=NAVY, bold=True, fs=9)
box(ax, 7.2, 7.6, 2.6, 1.2, "Shared memory context\n(cross-agent state)", fc="white")
arrow(ax, 2.0, 8.2, 3.4, 8.2); arrow(ax, 6.6, 8.25, 7.2, 8.25)
arrow(ax, 7.2, 8.0, 6.6, 8.0)

agents = [("Listing\nAgent", 0.4), ("Orders\nAgent", 2.85), ("Payments\nAgent", 5.3), ("Post-Transaction\nAgent", 7.75)]
for i, (name, x) in enumerate(agents):
    hl = (i == 2)
    box(ax, x, 4.6, 1.9, 1.4, name, fc=(NAVY if hl else LIGHT), bold=hl, lw=2.2 if hl else 1.2)
    arrow(ax, 5.0, 7.5, x + 0.95, 6.0)

mcps = [("Catalog\nMCP", 0.4), ("Fulfillment /\nCarrier MCP", 2.85),
        ("Settlement · Fee\nDispute · Fraud\nMCP servers", 5.3), ("Performance\nMetrics MCP", 7.75)]
for i, (name, x) in enumerate(mcps):
    hl = (i == 2)
    box(ax, x, 2.0, 1.9, 1.7 if hl else 1.4, name, fc="white", ec=(ACC if hl else NAVY),
        lw=2.0 if hl else 1.2)
    arrow(ax, x + 0.95, 4.6, x + 0.95, 3.7 if hl else 3.4)
box(ax, 1.6, 0.2, 6.8, 1.0, "Platform APIs  (settlement, fees, disputes, fraud signals, catalog, fulfillment, metrics)", fc=LIGHT, fs=8)
for x in [1.35, 3.8, 6.25, 8.7]:
    arrow(ax, x, 2.0, x, 1.2)
ax.text(5, 9.6, "Multi-agent Seller AI Copilot: system overview", ha="center", fontsize=10.5, weight="bold", color=NAVY)
fig.tight_layout(); fig.savefig(f"{OUT}/arch1_system_overview.png", dpi=200); plt.close(fig)

# ---------------- Figure 2: Payments Agent pipeline ----------------
fig, ax = canvas(8.6, 4.2)
steps = [("Settlement\nevent", 0.2, LIGHT), ("Retrieve\ndisbursement\nbreakdown\n(Settlement MCP)", 1.9, "white"),
         ("Recompute\nexpected fees\n(Fee MCP,\nversion-aware)", 3.85, "white"),
         ("Delta analysis &\ntyped anomaly\nclassification", 5.8, "white"),
         ("Confidence\ngate", 7.65, NAVY)]
for name, x, fc in steps:
    box(ax, x, 5.4, 1.6, 2.2, name, fc=fc, bold=(fc == NAVY), fs=8)
for x in [1.8, 3.75, 5.7, 7.55]:
    arrow(ax, x, 6.5, x + 0.12, 6.5)
box(ax, 0.6, 2.2, 2.4, 1.3, "Auto-execute\ndispute filing\n(Dispute MCP)", fc="white", ec=GREEN, lw=1.8)
box(ax, 3.8, 2.2, 2.4, 1.3, "Escalate to seller\n(evidence bundle,\none-click approval)", fc="white", ec=ACC, lw=1.8)
box(ax, 7.0, 2.2, 2.4, 1.3, "Log & monitor", fc="white", ec=GRAY, lw=1.4)
arrow(ax, 8.0, 5.4, 1.8, 3.5, color=GREEN); ax.text(4.1, 4.65, "conf ≥ θ_auto", fontsize=8, color=GREEN)
arrow(ax, 8.35, 5.4, 5.0, 3.5, color=ACC); ax.text(6.6, 4.65, "θ_esc ≤ conf < θ_auto", fontsize=8, color=ACC)
arrow(ax, 8.7, 5.4, 8.2, 3.5, color=GRAY); ax.text(8.75, 4.5, "conf < θ_esc", fontsize=8, color=GRAY)
box(ax, 0.6, 0.3, 5.6, 1.1, "Compliance MCP check (execution-layer gate: blocks non-compliant actions)", fc=LIGHT, fs=8)
arrow(ax, 1.8, 2.2, 2.4, 1.4, color=RED); arrow(ax, 5.0, 2.2, 4.4, 1.4, color=RED)
box(ax, 7.0, 0.3, 2.4, 1.1, "Audit log\n(all decisions)", fc="white")
arrow(ax, 6.2, 0.85, 7.0, 0.85)
ax.text(5, 9.0, "Payments Agent reconciliation and action pipeline", ha="center", fontsize=10.5, weight="bold", color=NAVY)
fig.tight_layout(); fig.savefig(f"{OUT}/arch2_payments_pipeline.png", dpi=200); plt.close(fig)

# ---------------- Figure 3: confidence-gated state machine ----------------
fig, ax = canvas(8.2, 4.6)
box(ax, 3.6, 8.4, 2.8, 1.1, "Anomaly / signal\ndetected", fc=LIGHT)
box(ax, 3.6, 6.3, 2.8, 1.1, "Confidence scoring\n(calibrated posterior)", fc="white")
arrow(ax, 5.0, 8.4, 5.0, 7.4)
box(ax, 0.4, 3.9, 2.6, 1.2, "AUTO-EXECUTE", fc=NAVY, bold=True)
box(ax, 3.8, 3.9, 2.6, 1.2, "SELLER APPROVAL\n(escalated)", fc="white", ec=ACC, lw=1.8)
box(ax, 7.2, 3.9, 2.4, 1.2, "MONITOR\n(logged only)", fc="white", ec=GRAY)
arrow(ax, 4.3, 6.3, 1.9, 5.1, color=GREEN); ax.text(2.0, 5.75, "conf ≥ θ_auto\n∧ compliant", fontsize=7.5, color=GREEN)
arrow(ax, 5.0, 6.3, 5.0, 5.1, color=ACC); ax.text(5.15, 5.6, "θ_esc ≤ conf < θ_auto", fontsize=7.5, color=ACC)
arrow(ax, 5.9, 6.3, 8.3, 5.1, color=GRAY); ax.text(7.3, 5.75, "conf < θ_esc", fontsize=7.5, color=GRAY)
box(ax, 0.4, 1.4, 2.6, 1.1, "Execute action\n+ audit log", fc="white", ec=GREEN, lw=1.6)
arrow(ax, 1.7, 3.9, 1.7, 2.5, color=GREEN)
arrow(ax, 3.8, 4.3, 3.0, 2.3, color=GREEN); ax.text(2.9, 3.3, "approve", fontsize=7.5, color=GREEN)
box(ax, 4.6, 1.4, 2.2, 1.1, "Reject → log,\nrecalibrate", fc="white", ec=GRAY)
arrow(ax, 5.3, 3.9, 5.5, 2.5, color=GRAY); ax.text(5.6, 3.2, "reject", fontsize=7.5, color=GRAY)
box(ax, 7.4, 1.4, 2.2, 1.1, "HALT + notify\n(non-compliant)", fc="white", ec=RED, lw=1.8)
arrow(ax, 2.2, 3.9, 7.9, 2.5, color=RED, ls="--")
ax.text(2.6, 2.75, "compliance check fails (any state)", fontsize=7.5, color=RED)
ax.text(5, 9.8, "Confidence-gated action state machine", ha="center", fontsize=10.5, weight="bold", color=NAVY)
fig.tight_layout(); fig.savefig(f"{OUT}/arch3_state_machine.png", dpi=200); plt.close(fig)

# ---------------- Figure 4: cascade detection flow ----------------
fig, ax = canvas(8.6, 3.9)
box(ax, 0.3, 6.4, 2.0, 1.6, "Payments\nAgent\nstep j", fc=NAVY, bold=True)
box(ax, 4.2, 6.4, 2.0, 1.6, "Orchestrator\nstep j+1", fc=LIGHT)
box(ax, 7.9, 6.4, 1.9, 1.6, "Listing\nAgent\nstep j+2", fc=LIGHT)
arrow(ax, 2.3, 7.2, 4.2, 7.2); arrow(ax, 6.2, 7.2, 7.9, 7.2)
ax.text(3.2, 7.5, "value V_j", fontsize=8, ha="center")
ax.text(7.05, 7.5, "V_{j+1}", fontsize=8, ha="center")
box(ax, 2.2, 3.4, 2.4, 1.5, "Inline check\n|V_j − GT_j| / GT_j\n> 5% ?", fc="white", ec=ACC, lw=1.8, fs=8)
arrow(ax, 3.3, 6.4, 3.4, 4.9, color=ACC)
box(ax, 0.3, 0.6, 2.6, 1.2, "Ground truth\nSettlement API", fc="white")
arrow(ax, 1.6, 1.8, 2.8, 3.4)
box(ax, 5.6, 3.5, 2.0, 1.3, "Pass value\ndownstream", fc="white", ec=GREEN, lw=1.6)
arrow(ax, 4.6, 4.15, 5.6, 4.15, color=GREEN); ax.text(5.05, 4.4, "no", fontsize=8, color=GREEN)
box(ax, 3.6, 0.6, 3.0, 1.3, "HALT downstream actions\nnotify seller + corrections", fc="white", ec=RED, lw=1.8, fs=8)
arrow(ax, 3.4, 3.4, 4.4, 1.9, color=RED); ax.text(3.3, 2.6, "yes", fontsize=8, color=RED)
box(ax, 7.3, 0.6, 2.4, 1.3, "Cascade risk score\nf(Δmax, depth, agents)", fc=LIGHT, fs=8)
arrow(ax, 6.6, 1.2, 7.3, 1.2, color=RED)
ax.text(5, 9.2, "Inline value verification at agent boundaries", ha="center", fontsize=10.5, weight="bold", color=NAVY)
fig.tight_layout(); fig.savefig(f"{OUT}/arch4_cascade_flow.png", dpi=200); plt.close(fig)

print("4 architecture figures saved")
