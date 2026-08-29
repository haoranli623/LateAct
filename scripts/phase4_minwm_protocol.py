#!/usr/bin/env python3
"""Shared protocol loading and locks for minWM Phase 4 experiments."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


PROJECT = Path(__file__).resolve().parents[1]
CONFIRMATORY_PHASE = "phase4_minwm_yaw_confirmatory"
ORIGINAL_ARTIFACT_DIRECTORY = Path(
    "/mnt/NAS/data/hl5757/generated_artifacts/lateact/phase4_minwm"
)

CONFIRMATORY_LOCK = {
    "upstream.repository": "https://github.com/shengshu-ai/minWM",
    "upstream.commit": "df522a26cd4409d3e3e8f269cc98eac069b5df47",
    "upstream.backbone": "Wan-AI/Wan2.1-T2V-1.3B",
    "upstream.backbone_revision": "37ec512624d61f7aa208f7ea8140a131f93afc9a",
    "upstream.checkpoint_repo": "MIN-Lab/minWM",
    "upstream.checkpoint_path": "Wan21/Action2V/dmd/model.pt",
    "upstream.checkpoint_revision": "21bd74da43b5a061c0b8ff277515088ccd2c798b",
    "upstream.checkpoint_etag": "bdb947d45fb04513305492c2ee393d51d0621ec0e99fd312224f5d61a330aa77",
    "sampler.nfe": 4,
    "sampler.configured_steps": [1000, 750, 500, 250],
    "sampler.timestep_shift": 5.0,
    "sampler.num_frames": 20,
    "sampler.num_frames_per_block": 4,
    "sampler.resolution": [480, 832],
    "sampler.dtype": "bfloat16",
    "sampler.stochastic_coupling": "initial_noise_and_three_renoise_tensors",
    "control.axis": "native_yaw_camera_rotation",
    "control.negative": "j",
    "control.positive": "l",
    "control.trajectory_steps_per_branch": 3,
    "control.rotation_degrees_per_step": 3.0,
    "control.terminal_rotation_degrees": 9.0,
    "control.oracle_endpoint_separation_degrees": 18.0,
    "control.old_new_directions": ["j_to_l", "l_to_j"],
    "control.intrinsics": [0.5, 0.5, 0.5, 0.5],
    "control.only_tensor_allowed_to_differ_by_nfe": "viewmats",
    "replication.scenes": 8,
    "replication.prompt_source": "Wan21/prompts/demos.txt",
    "replication.prompt_indexing": "zero_based",
    "replication.prompt_indices": list(range(8, 16)),
    "replication.seeds": list(range(41008, 41016)),
    "replication.switch_positions": [0, 1, 2, 3, 4],
    "replication.same_action_control_prompt_indices": [8, 9],
    "replication.same_action_control": "j_to_j",
    "replication.prompt_file_sha256": "847f82434bf4173dd9a46c2cb05db282876190dfec5b2653a9290ee9c060f5b7",
    "replication.selected_prompts_sha256": "2e72cbfdbedbec3e7ebfc9f098ec9a4c6e49bc907df867ad602f78f60a75c82b",
    "replication.scene_screening": "prohibited",
    "replication.prompt_replacement": "prohibited",
    "evaluator.primary": "accumulated_robust_affine_horizontal_translation",
    "evaluator.independent_check": "forward_backward_lk_median_horizontal_translation",
    "evaluator.min_tracks_per_transition": 20,
    "evaluator.response": "(switch_minus_old)/(new_minus_old)",
    "evaluator.clipping": False,
    "validity.oracle_videos_must_differ": True,
    "validity.min_oracle_gap_pixels": 8.0,
    "validity.evaluator_sign_agreement": True,
    "validity.same_action_floor_multiplier": 10.0,
    "validity.visual_check": "finite_nonconstant_decoded_video_and_manual_contact_sheet_review",
    "phase4b_pass.min_valid_direction_scene_pairs": 12,
    "phase4b_pass.min_monotone_pairs": 12,
    "phase4b_pass.monotonic_tolerance": 0.10,
    "phase4b_pass.min_bounded_pairs": 12,
    "phase4b_pass.bounded_range": [-0.10, 1.10],
    "phase4b_pass.min_median_adjacent_drop": 0.15,
    "phase4b_pass.min_directions_each_orientation": 6,
    "phase4b_pass.required_visual_valid_pairs": 16,
    "phase4b_pass.require_same_action_latent_equality": True,
    "phase4b_pass.require_stochastic_noise_cache_audits": True,
    "rollback.execution": "prohibited",
    "rollback.boundary_min_current_response": 0.80,
    "rollback.boundary_max_following_response": 0.65,
    "rollback.boundary_min_adjacent_drop": 0.15,
    "termination.on_gate_failure": "terminate_cross_model_work",
    "termination.allow_prompt_replacement": False,
    "termination.allow_threshold_change": False,
    "termination.allow_alternative_evaluator": False,
    "termination.allow_additional_action_family": False,
    "termination.allow_rollback_experiment_in_this_followup": False,
}


def _read_nested(value: dict, dotted: str):
    current = value
    for part in dotted.split("."):
        if part not in current:
            raise ValueError(f"frozen protocol is missing {dotted}")
        current = current[part]
    return current


def load_protocol(path: Path) -> dict:
    path = path.resolve()
    protocol = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(protocol, dict):
        raise ValueError(f"protocol must be a mapping: {path}")
    validate_protocol(protocol)
    protocol["_config_path"] = str(path)
    protocol["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return protocol


def validate_protocol(protocol: dict) -> None:
    negative, positive = actions(protocol)
    if negative == positive or not all(re.fullmatch(r"[a-z]+", item) for item in (negative, positive)):
        raise ValueError(f"invalid opposite action pair: {negative!r}, {positive!r}")
    indices = prompt_indices(protocol)
    seeds = prompt_seeds(protocol)
    if len(indices) != int(protocol["replication"]["scenes"]):
        raise ValueError("prompt count does not match replication.scenes")
    if len(indices) != len(set(indices)) or len(indices) != len(seeds):
        raise ValueError("prompt indices must be unique and aligned one-to-one with seeds")
    if switch_positions(protocol) != [0, 1, 2, 3, 4]:
        raise ValueError("Phase 4 requires the unchanged switch positions [0,1,2,3,4]")
    controls = control_prompt_indices(protocol)
    if any(index not in indices for index in controls):
        raise ValueError("same-action control prompts must be inside the frozen prompt set")

    if protocol.get("phase") == CONFIRMATORY_PHASE:
        for dotted, expected in CONFIRMATORY_LOCK.items():
            actual = _read_nested(protocol, dotted)
            if actual != expected:
                raise ValueError(
                    f"confirmatory protocol drift at {dotted}: expected {expected!r}, got {actual!r}"
                )


def actions(protocol: dict) -> tuple[str, str]:
    control = protocol["control"]
    return str(control["negative"]), str(control["positive"])


def prompt_indices(protocol: dict) -> list[int]:
    replication = protocol["replication"]
    if "prompt_indices" in replication:
        return [int(value) for value in replication["prompt_indices"]]
    return list(range(int(replication["scenes"])))


def prompt_seeds(protocol: dict) -> list[int]:
    return [int(value) for value in protocol["replication"]["seeds"]]


def seed_by_prompt(protocol: dict) -> dict[int, int]:
    return dict(zip(prompt_indices(protocol), prompt_seeds(protocol)))


def switch_positions(protocol: dict) -> list[int]:
    return [int(value) for value in protocol["replication"]["switch_positions"]]


def control_prompt_indices(protocol: dict) -> list[int]:
    replication = protocol["replication"]
    values = replication.get(
        "same_action_control_prompt_indices",
        replication.get("same_action_control_scenes", []),
    )
    return [int(value) for value in values]


def trajectory_steps_per_branch(protocol: dict) -> int:
    return int(protocol["control"].get("trajectory_steps_per_branch", 3))


def direction_mappings(protocol: dict) -> dict[str, dict[int, str]]:
    negative, positive = actions(protocol)
    return {
        f"{negative}_to_{positive}": {
            0: f"oracle_{positive}",
            1: f"{negative}_to_{positive}_s1",
            2: f"{negative}_to_{positive}_s2",
            3: f"{negative}_to_{positive}_s3",
            4: f"oracle_{negative}",
        },
        f"{positive}_to_{negative}": {
            0: f"oracle_{negative}",
            1: f"{positive}_to_{negative}_s1",
            2: f"{positive}_to_{negative}_s2",
            3: f"{positive}_to_{negative}_s3",
            4: f"oracle_{positive}",
        },
    }


def unique_run_specs(protocol: dict) -> list[tuple[str, str, str, int]]:
    negative, positive = actions(protocol)
    specs = [
        (f"oracle_{negative}", negative, negative, 0),
        (f"oracle_{positive}", positive, positive, 0),
    ]
    for old, new in ((negative, positive), (positive, negative)):
        for switch in switch_positions(protocol)[1:-1]:
            specs.append((f"{old}_to_{new}_s{switch}", old, new, switch))
    return specs


def load_and_validate_prompts(upstream: Path, protocol: dict) -> list[str]:
    if protocol.get("phase") == CONFIRMATORY_PHASE:
        result = subprocess.run(
            [
                "git", "-c", f"safe.directory={upstream}",
                "-C", str(upstream), "rev-parse", "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        actual_commit = result.stdout.strip()
        expected_commit = protocol["upstream"]["commit"]
        if actual_commit != expected_commit:
            raise ValueError(
                f"minWM commit changed: expected {expected_commit}, got {actual_commit}"
            )
    path = upstream / "Wan21/prompts/demos.txt"
    raw = path.read_bytes()
    prompts = [line.strip() for line in raw.decode("utf-8").splitlines() if line.strip()]
    indices = prompt_indices(protocol)
    if max(indices) >= len(prompts):
        raise ValueError(f"official prompt file contains only {len(prompts)} prompts")
    if protocol.get("phase") == CONFIRMATORY_PHASE:
        file_hash = hashlib.sha256(raw).hexdigest()
        expected_file_hash = protocol["replication"]["prompt_file_sha256"]
        if file_hash != expected_file_hash:
            raise ValueError(
                f"official prompt file changed: expected {expected_file_hash}, got {file_hash}"
            )
        selected = "\n".join(prompts[index] for index in indices) + "\n"
        selected_hash = hashlib.sha256(selected.encode("utf-8")).hexdigest()
        expected_selected_hash = protocol["replication"]["selected_prompts_sha256"]
        if selected_hash != expected_selected_hash:
            raise ValueError(
                f"selected prompt content changed: expected {expected_selected_hash}, got {selected_hash}"
            )
    return prompts


def artifact_root() -> Path:
    return Path(
        os.environ.get(
            "LATEACT_ARTIFACT_ROOT",
            PROJECT / "artifacts" / "lateact",
        )
    )


def default_output(protocol: dict) -> Path:
    subdirectory = protocol.get("storage", {}).get("artifact_subdirectory")
    if subdirectory is None:
        subdirectory = "phase4_minwm"
    return artifact_root() / str(subdirectory)


def validate_confirmatory_output(path: Path, protocol: dict, require_empty: bool) -> None:
    if protocol.get("phase") != CONFIRMATORY_PHASE:
        return
    resolved = path.resolve()
    expected = default_output(protocol).resolve()
    original_candidates = {
        ORIGINAL_ARTIFACT_DIRECTORY.resolve(),
        (artifact_root() / "phase4_minwm").resolve(),
    }
    if resolved in original_candidates:
        raise ValueError("confirmatory output must never target the original Phase 4 artifacts")
    if resolved != expected or resolved.name != "phase4_minwm_yaw_confirmatory":
        raise ValueError(f"confirmatory output must be exactly {expected}, got {resolved}")
    if require_empty and resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(
            f"confirmatory output is non-empty; refusing to overwrite or resume: {resolved}"
        )


def selected_prompt_indices(
    protocol: dict,
    scene_start: int | None,
    scene_end: int | None,
) -> list[int]:
    frozen = prompt_indices(protocol)
    start = frozen[0] if scene_start is None else scene_start
    end = frozen[-1] + 1 if scene_end is None else scene_end
    selected = [index for index in frozen if start <= index < end]
    if not selected or selected != list(range(start, end)):
        raise ValueError(
            f"requested prompt range [{start},{end}) is not a contiguous subset of {frozen}"
        )
    return selected
