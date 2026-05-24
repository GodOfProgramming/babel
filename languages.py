import json
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Lang:
    id: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Lang) and self.id == other.id


EN = Lang("en")

type LanguageMap = dict[Lang, dict[str, Optional[str]]]

with open("language_mapping.json", "r", encoding="utf-8") as f:
    LANGUAGES: LanguageMap = {Lang(k): v for k, v in json.load(f).items()}
