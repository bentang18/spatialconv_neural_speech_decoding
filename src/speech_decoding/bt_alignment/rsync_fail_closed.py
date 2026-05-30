"""Fail-closed rsync to DCC persistent tier.

DCC's `/work` scratch purges every 75 days, so durable alignment artifacts must
land on `/hpc/group/coganlab/ht203/` (1 TB lab-shared). This module wraps rsync with a
checksum verification stage so the copy is *only* declared successful when every
destination file's SHA-256 matches the source.

Why fail-closed:
- A network glitch during rsync can produce a truncated file with no exit-code
  signal.
- The teacher cache is on the critical path for v14 P3 distillation; a single
  corrupted clip silently corrupts L8 features for that anchor.
- A 1 TB cache rebuild on DCC takes hours of GPU; preventing a re-run is worth
  the extra hash pass.

Usage:
    rsync_with_verify(src_dir, dst_dir)  # returns RsyncVerifyResult
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import hashlib
import shutil
import subprocess


@dataclass
class RsyncVerifyResult:
    src_root: str
    dst_root: str
    n_src_files: int
    n_dst_files: int
    n_verified: int
    n_mismatch: int
    rsync_exit_code: int
    mismatches: list = field(default_factory=list)

    @property
    def success(self) -> bool:
        return (
            self.rsync_exit_code == 0
            and self.n_src_files == self.n_dst_files == self.n_verified
            and self.n_mismatch == 0
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success"] = self.success
        return d


def sha256_of_file(path: Path, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def walk_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file())


def run_rsync(src: Path, dst: Path, extra_args: list[str] | None = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    src_arg = str(src).rstrip("/") + "/"
    cmd = ["rsync", "-a", "--checksum"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend([src_arg, str(dst)])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return res.returncode


def verify_checksums(src_root: Path, dst_root: Path) -> tuple[int, list[dict]]:
    """Walk src_root, compute SHA-256 for each file, compare to dst_root's same-relpath
    file. Returns (n_verified, list_of_mismatches)."""
    src_files = walk_files(src_root)
    mismatches = []
    n_verified = 0
    for src_path in src_files:
        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        if not dst_path.exists():
            mismatches.append({
                "rel": str(rel), "reason": "missing",
                "src_sha": None, "dst_sha": None,
            })
            continue
        src_sha = sha256_of_file(src_path)
        dst_sha = sha256_of_file(dst_path)
        if src_sha != dst_sha:
            mismatches.append({
                "rel": str(rel), "reason": "checksum mismatch",
                "src_sha": src_sha, "dst_sha": dst_sha,
            })
        else:
            n_verified += 1
    return n_verified, mismatches


def rsync_with_verify(
    src: Path,
    dst: Path,
    extra_rsync_args: list[str] | None = None,
    on_mismatch_remove: bool = True,
) -> RsyncVerifyResult:
    """Rsync src -> dst, then verify SHA-256 of every src file against its dst twin.

    If any mismatch (or missing file) is found and `on_mismatch_remove=True`, the
    offending dst files are removed so the next call retries the broken set.
    """
    code = run_rsync(src, dst, extra_args=extra_rsync_args)
    src_files = walk_files(src)
    n_verified, mismatches = verify_checksums(src, dst)
    dst_files = walk_files(dst)

    if on_mismatch_remove:
        for m in mismatches:
            p = dst / m["rel"]
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass

    return RsyncVerifyResult(
        src_root=str(src),
        dst_root=str(dst),
        n_src_files=len(src_files),
        n_dst_files=len(dst_files),
        n_verified=n_verified,
        n_mismatch=len(mismatches),
        rsync_exit_code=code,
        mismatches=mismatches,
    )


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
