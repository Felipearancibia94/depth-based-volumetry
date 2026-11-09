# Gallery

One image per pet, chosen by the owner, for a personal site. Only the latest image of each
pet is kept here: updating replaces `<name>.jpg` and `<name>.json` (git history keeps older
versions once they are committed).

| Pet | Image | Data |
|---|---|---|
| Negrita | [negrita.jpg](negrita.jpg) | [negrita.json](negrita.json) |

Rules for this folder:

- Crops of the animal only: no people, no room details. Check an image by eye before
  committing it.
- The name is assigned by the owner. The lab does not identify individual animals (track ids
  are not identities), so a name here is a manual label.
- The text in the JSON is the local VLM's interpretation of the image, kept apart from the
  detector's data. It is not a measurement and its `certainty` is not a confidence.
- Nothing is published automatically: the lab is local only (see
  [ADR 0004](../adr/0004-local-only-loopback-endpoints.md)). Copying an image here and
  committing it is a deliberate step by the owner.
