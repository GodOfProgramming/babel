from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Lang:
    id: str

    def __eq__(self, other: "Lang") -> bool:
        return self.id == other.id


EN = Lang("en")

with open("language_mapping.json", "r", encoding="utf-8") as f:
    LANGUAGES: dict[Lang, dict[str, str]] = {
        Lang(k): v for k, v in json.load(f).items()
    }
