"""How the lab names the pets. Drawings only: no photos are embedded in this diagram."""

from excalidraw_lib import GREEN, MUTED, ORANGE, RED, cat, dog

OUTPUT = "docs/diagrams/pet-identification"
SEED = 5000


def build(d, repo):
    d.header(
        "How the lab names Negrita, Nube and Leia",
        "Green = works in my tests (saved images, synthetic data).  Orange = limited or "
        "depends on conditions.  Grey = not tested yet.",
    )
    cat(d, "c1", 1640, 28, 60, "#343a40", "#ffd43b")
    cat(d, "c2", 1730, 28, 60, "#f59f00", "#2f9e44", patch="#1e1e1e")
    dog(d, "d1", 1820, 28, 60)

    d.zone("z1", 0, 85, 1900, 175, ("#dbe4ff", "#4a9eed"))
    d.label("z1l", 12, 91, "From a camera frame to a named pet", 17, "#1971c2")
    d.row("b", 20, 125, 240, 110, 265, [
        ("Camera frame\n640x480, 15 fps\nneeds light", ORANGE),
        ("Detector\ncat or dog + box + score\nsensitive to orientation", ORANGE),
        ("Evidence\n4 crops + 1 full frame\n448 px each", GREEN),
        ("VLM prompt\nfacts + list of pets +\n'unknown' allowed", GREEN),
        ("Answer as JSON\npet, activity, description\nuncertainty, certainty", GREEN),
        ("Name gate\nkept if detector score\n>= 0.4 and certainty\nnot low, else 'unknown'",
         ORANGE),
        ("Stored event + dashboard\n'Nube - resting - text'\nhint; owner can correct",
         ORANGE),
    ], fs=15)  # fmt: skip

    d.zone("z2", 0, 280, 1900, 110, ("#e5dbff", "#9775fa"))
    d.label("z2l", 12, 286, "In parallel: alerts never wait for the VLM", 17, "#6741d9")
    d.box("r1", 330, 310, 275, 70, "Rules (7, deterministic)\nmeasured facts only", GREEN, 15)
    d.box("r2", 640, 310, 585, 70,
          "Alert or notification\nsent as soon as a rule fires; the VLM text arrives later "
          "(about 40-55 s)", GREEN, 15)  # fmt: skip
    d.arrow("a_b1_r1", "b1", "r1")
    d.arrow("a_r1_r2", "r1", "r2")

    d.label("pt", 0, 420, "Your three animals ([[vlm.pets]] in the config)", 24, GREEN[1])
    for name, x in (("Negrita", 0), ("Nube", 640), ("Leia", 1280)):
        d.zone(f"card_{name}", x, 465, 620, 330, ("#f1f3f5", "#868e96"), opacity=60)
        d.label(f"n_{name}", x + 15, 475, name, 24)
    cat(d, "neg", 130, 650, 190, "#343a40", "#ffd43b")
    d.label("t_neg", 255, 530, "Prompt description:\na cat with a fully\nblack coat\n\n"
            "Named correctly:\n4 of 4 crops\n4 of 4 orientations", 15)  # fmt: skip
    cat(d, "nube", 770, 650, 190, "#f59f00", "#2f9e44", patch="#1e1e1e")
    d.ellipse("nube_white", 770 - 190 * 0.43, 650 - 190 * 0.30, 190 * 0.36, 190 * 0.26,
              "#ffffff", stroke="#ffffff", stroke_width=1)  # fmt: skip
    d.label("t_nube", 895, 520, "Prompt description:\na cat with a tricolor\ncoat (black, "
            "orange,\nwhite patches)\n\nNamed correctly:\n4 of 4 orientations", 15)  # fmt: skip
    d.label("t_nube2", 895, 690, "As captured, the detector read her\nlying down as 'dog' "
            "0.73 (model l);\nturned, as 'cat' 0.93.", 15, MUTED)  # fmt: skip
    dog(d, "dg", 1410, 650, 190)
    d.label("t_leia", 1535, 530, "Prompt description:\na dog\n(add coat, size,\nmarks)\n\n"
            "Not tested:\nno image of her yet", 15)  # fmt: skip

    d.zone("z3", 0, 825, 1900, 160, RED, opacity=30)
    d.label(
        "z3l",
        12,
        831,
        "Known limits (measured here; the name gate is not re-tested live)",
        20,
        RED[1],
    )
    d.label("lims", 12, 868,
            "- A room with a person and no animal was named \"Negrita\" with certainty high "
            "(1 case, before the name gate existed): treat the name as a hint, not a fact.\n"
            "- In a dark room the colour camera is almost black (brightness 5.8 of 255) and "
            "the detector finds nothing.\n"
            "- A new event takes about 40-55 s in the VLM; repeating the same request is "
            "cached and takes about 3 s.\n"
            "- The detector depends on orientation: a black cat upside down fell from 0.92 to "
            "0.33.\n"
            "- Leia and a live run with light on are untested; the numbers are one session on "
            "one PC, not accuracy.", 16)  # fmt: skip
    d.label("foot", 0, 1010,
            "Details: ROADMAP.md and docs/pet-monitoring.md.  The name is a label from the "
            "VLM's reading of the image; the lab has no recognition model.", 14, MUTED)  # fmt: skip
