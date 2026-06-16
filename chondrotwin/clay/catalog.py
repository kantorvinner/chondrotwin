from __future__ import annotations

from .schema import CorpusRecord, SplitCounts

OAI = CorpusRecord(
    name="Osteoarthritis Initiative (OAI)",
    role="WORMS grading cohort (3D DESS)",
    counts=SplitCounts(train=3212, val=802, test=782),
    access="https://nda.nih.gov/oai",
    license="NIH Data Use Agreement; registration required",
    resolves=True,
)

OAI_ZIB = CorpusRecord(
    name="OAI-ZIB segmentations",
    role="dense six-compartment segmentation",
    counts=SplitCounts(train=404, val=51, test=52),
    access="https://pubdata.zib.de",
    license="OAI DUA plus ZIB redistribution terms",
    resolves=True,
)

SKI10 = CorpusRecord(
    name="SKI10 (MICCAI 2010 challenge)",
    role="zero-shot femur/tibia cartilage transfer",
    counts=SplitCounts(train=100, val=0, test=50),
    access="MICCAI 2010 challenge archive (original host retired)",
    license="challenge-distribution terms",
    resolves=False,
)

IWOAI = CorpusRecord(
    name="IWOAI 2019 partition (derived from OAI-ZIB)",
    role="held-out segmentation evaluation",
    counts=SplitCounts(train=60, val=14, test=14),
    access="derived from OAI-ZIB; see Desai et al., Radiol. Artif. Intell. 3(3):e200078, 2021",
    license="inherits OAI/OAI-ZIB terms",
    resolves=False,
)

REGISTRY: dict[str, CorpusRecord] = {
    "oai": OAI,
    "oai_zib": OAI_ZIB,
    "ski10": SKI10,
    "iwoai": IWOAI,
}


def resolvable_links() -> dict[str, str]:
    return {key: rec.access for key, rec in REGISTRY.items() if rec.resolves}
