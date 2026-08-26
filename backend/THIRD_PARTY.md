# Third-party model

This backend vendors the source configuration and landmark metadata from
[`hawaii-ai/dxa-pointplacement`](https://github.com/hawaii-ai/dxa-pointplacement)
at commit `7ac19eb9c99d9a5edc631446b45528e889d627ed`.

The vendored source is licensed under Apache License 2.0. The original files are
kept in `third_party/dxa_pointplacement/`, together with the upstream license.
Production wrappers in `app/` are Osseo AI code and explicitly describe all
behavioral changes from the original batch CLI.

The pretrained checkpoint is hosted separately by the upstream authors on
Google Drive and is not committed here. Its redistribution terms are not stated
separately in the upstream repository. Confirm the checkpoint license before
shipping it as part of a product.

Downloaded upstream artifact metadata:

- size: `277761941` bytes;
- SHA-256: `92a90627ecde530a2c259fa864c578806cdda5b0718a26180e891779ad453ced`.

The model is a research baseline for 105 fiducial landmarks on extracted
air-ratio total-body DXA images. It is not a spine/hip ROI model, an artifact
classifier, a diagnostic model, or a clinically validated quality-control model.
