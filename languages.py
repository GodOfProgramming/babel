MODELS = {
    "facebook/nllb-200-distilled-600M": [
        "fra_Latn",
        "deu_Latn",
        "rus_Cyrl",
        "arb_Arab",
        "jpn_Jpan",
        "kor_Hang",
        "zul_Latn",
        "amh_Ethi",
        "tam_Taml",
        "tha_Thai",
        "eng_Latn",
    ]
}

from dataclasses import dataclass


@dataclass(frozen=True)
class Lang:
    id: str

    def __eq__(self, other: "Lang") -> bool:
        return self.id == other.id


EN = Lang("en")

LANGUAGES = [
    Lang("fr"),
    Lang("ar"),
    Lang("de"),
    EN,
    Lang("it"),
    Lang("ja"),
    Lang("ko"),
    Lang("pl"),
    Lang("ru"),
    Lang("tr"),
    Lang("zh"),
]
