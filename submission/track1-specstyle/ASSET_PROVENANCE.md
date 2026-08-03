# Asset Provenance

## Demo inputs

`media/source.png` and `media/style.png` were generated for this submission
with OpenAI image generation without a reference image. They contain no people,
private source media, readable text, logos, or watermark.

The retained prompt descriptions are:

- Source: unbranded frosted-glass perfume bottle centered on a neutral gray
  product-photography set.
- Style: abstract editorial studio palette and material reference using cobalt
  blue, coral red, sunflower yellow, translucent acrylic, crisp hard shadows,
  and fine printed grain.

The exact original tool payload is not embedded in this repository. The files,
descriptive prompts, byte sizes, and SHA-256 hashes are recorded in
[`evidence/asset_provenance.json`](evidence/asset_provenance.json).

## Radeon outputs

`media/preview-seed-0.png` through `preview-seed-3.png` are byte-identical to
the four files published by the recorded Radeon wall
`preview-wall-3a18518754104e71ab06f32050c2e3bf`. The wall record binds each file
to its deterministic seed, content hash, execution fingerprint, compiled
request, and runtime precision policy.

These are demonstration outputs generated from the recorded demo inputs using
the local, approved runtime model supply. Model weights are not redistributed.
Use of the outputs remains subject to the applicable model and platform terms.

## Screenshots and recordings

The UI screenshots and recordings are first-party captures of SpecStyle running
on the AMD competition instance. They show filenames and engineering evidence,
but no credentials, tokens, private filesystem locations, or private media.

`SpecStyle_Radeon_UI_Run.webm` is the uncut 73.28-second browser capture.
`SpecStyle_Radeon_Demo.mp4` presents the run between evidence slides rendered
from the editable deck. Its encoded UI segment is 73.24 seconds because the
concat boundary consumes one 25 fps frame; the standalone WebM is the archival
record. The machine-readable timeline and media hashes are recorded in
[`evidence/video_manifest.json`](evidence/video_manifest.json).

## License identifiers

- `LicenseRef-OpenAI-Generated-Output`: generated demo input under the
  applicable OpenAI terms.
- `LicenseRef-SpecStyle-Demo-Output`: locally generated demonstration output;
  no model weight or third-party source image is included.
- `LicenseRef-First-Party-Screenshot`: first-party software screen capture.
- `LicenseRef-First-Party-Demo`: first-party evidence deck and software capture
  compiled into the submission video.

The repository code license is intentionally not selected by this provenance
record. Model license approvals remain separate deployment evidence.
