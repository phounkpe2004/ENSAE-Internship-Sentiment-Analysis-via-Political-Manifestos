from dataclasses import dataclass
from pathlib import Path

@dataclass
class MCMCConfig:

    project_root: Path

    raw_data_dir: Path = None
    processed_data_dir: Path = None
    random_seed: int = 42

    data_code: str = "Important"

    def __post_init__(self):
        self.project_root = Path(self.project_root)

        if self.raw_data_dir is None:
            self.raw_data_dir = self.project_root / "data" / "raw"
        else:
            self.raw_data_dir = Path(self.raw_data_dir)  # tolère une string en entrée

        if self.processed_data_dir is None:
            self.processed_data_dir = self.project_root / "data" / "processed"
        else:
            self.processed_data_dir = Path(self.processed_data_dir)

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        assert self.raw_data_dir.exists(), (
            f"Dossier de données brutes introuvable : {self.raw_data_dir}"
        )