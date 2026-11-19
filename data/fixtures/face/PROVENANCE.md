# Canonical Face Fixture Provenance

This directory stores the canonical, tracked input fixtures used for deterministic face dataset generation and automated testing.

## Fixture Inventory & Cryptographic Hashes

| Fixture File | Role | Resolution | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `canonical_reference.png` | Primary identity probe template (`person_01`) | 128 × 128 | `19aef5180b9daed0e5f2dec591f420876bb38ef2dd8cfc82cb83ca47c88b9b55` |
| `canonical_negative.png` | Non-matching identity probe template | 512 × 512 | `1452b1a31d025d6d1b105f322532135dfff4e3c642c8e174a85d7b4153ecc908` |

## Usage & Generation Role

- `scripts/generate_datasets.py` reads directly from these fixture sources to populate:
  - `data/training/face/person_01_reference.png`
  - `data/evaluation/face/positive_query.png` (reference with calibrated +5% natural illumination shift)
  - `data/evaluation/face/negative_other_face.png`
  - `data/evaluation/face/no_face_distractor.png` (synthesized non-face geometric distractor)
- This guarantees dataset generation is self-contained within `data/` and has zero dependency on generated screenshot artifacts or evaluation output directories.
