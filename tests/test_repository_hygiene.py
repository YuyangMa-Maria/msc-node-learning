"""Keep machine-specific and generated material out of the submission tree."""

import unittest
from pathlib import Path
from subprocess import run


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bib", ".c", ".cpp", ".csv", ".h", ".html", ".js", ".json",
    ".md", ".ps1", ".py", ".toml", ".txt", ".yml",
}
IGNORED_PARTS = {
    ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__",
    "build", "managed_components",
}


def is_local_or_generated(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return any(part in IGNORED_PARTS or part.startswith(".venv") for part in parts)


class RepositoryHygieneTests(unittest.TestCase):
    def test_no_personal_absolute_paths(self) -> None:
        offenders: list[str] = []
        for path in ROOT.rglob("*"):
            if (
                path.is_file()
                and not is_local_or_generated(path)
                and path.suffix.lower() in TEXT_SUFFIXES
            ):
                text = path.read_text(encoding="utf-8-sig")
                windows_home = "C:" + "\\Users\\"
                posix_home = "C:/" + "Users/"
                if windows_home in text or posix_home in text:
                    offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

    def test_generated_directories_are_not_committed(self) -> None:
        generated = ("dataset", "experiments", "outputs", "artifacts", "runs")
        ignored = {
            line.strip()
            for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual([name for name in generated if f"{name}/" not in ignored], [])

        try:
            repository = run(
                ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return
        # A university source archive may deliberately omit the .git directory.
        if repository.returncode != 0 or Path(repository.stdout.strip()).resolve() != ROOT:
            return

        tracked_result = run(
            ["git", "-C", str(ROOT), "ls-files", "--", *generated],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        tracked = [line for line in tracked_result.stdout.splitlines() if line]
        self.assertEqual(tracked, [])


if __name__ == "__main__":
    unittest.main()
