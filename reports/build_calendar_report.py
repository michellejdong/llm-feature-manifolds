"""Render the visual calendar study walkthrough from the checked-in results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPOSITORY = "https://github.com/michellejdong/llm-feature-manifolds"
ENGELS = "https://arxiv.org/html/2405.14860"
MAISI = "https://maisi.org/research/directions/interpretability-feature-manifolds"
STEERING = "https://www.goodfire.com/research/manifold-steering"
GEOMETRY = "https://www.goodfire.com/research/the-world-inside-neural-networks"
SAES = "https://www.goodfire.com/research/can-saes-capture-neural-geometry"
STORIES = "https://www.goodfire.com/research/stories-in-space"
WIDTH, HEIGHT = letter
MARGIN = 43
BODY = WIDTH - 2 * MARGIN
INK = HexColor("#112B3A")
MUTED = HexColor("#415B69")
TEAL = HexColor("#086E64")
TEAL_BG = HexColor("#E4F1EC")
PALE = HexColor("#F2F6F7")
LINE = HexColor("#D6E2E6")
OCHRE = HexColor("#98501E")
OCHRE_BG = HexColor("#FAEEE3")
BLUE = HexColor("#385E86")


def read_csv(path: str) -> list[dict]:
    with (RESULTS / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path: str) -> dict:
    return json.loads((RESULTS / path).read_text())


def size(label: str, font: str, points: float) -> float:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    return stringWidth(label, font, points)


def line(pdf, x1, top1, x2, top2, stroke=LINE, weight=0.65):
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(weight)
    pdf.line(x1, HEIGHT - top1, x2, HEIGHT - top2)


def label(pdf, text, x, top, points=10, color=INK, face="Helvetica"):
    pdf.setFillColor(color)
    pdf.setFont(face, points)
    pdf.drawString(x, HEIGHT - top - points, text)


def paragraph(pdf, text, x, top, width=BODY, points=11.2, leading=15.9, color=INK,
              face="Helvetica", limit=1000):
    style = ParagraphStyle("report", fontName=face, fontSize=points, leading=leading, textColor=color)
    block = Paragraph(text, style)
    _, height = block.wrap(width, HEIGHT)
    if height > limit:
        raise ValueError(f"Text block at y={top} exceeds the layout: {text[:80]}")
    block.drawOn(pdf, x, HEIGHT - top - height)
    return top + height


def panel(pdf, x, top, width, height, fill=PALE, stroke=LINE):
    pdf.setFillColor(fill)
    pdf.setStrokeColor(stroke)
    pdf.setLineWidth(0.65)
    pdf.roundRect(x, HEIGHT - top - height, width, height, 8, fill=1, stroke=1)


def arrow(pdf, x1, x2, top, color=TEAL):
    line(pdf, x1, top, x2, top, color, 1.05)
    line(pdf, x2 - 4, top - 3, x2, top, color, 1.05)
    line(pdf, x2 - 4, top + 3, x2, top, color, 1.05)


def figure(pdf, path, x, top, width=BODY):
    png = ImageReader(str(RESULTS / path))
    pixel_width, pixel_height = png.getSize()
    height = width * pixel_height / pixel_width
    pdf.drawImage(png, x, HEIGHT - top - height, width, height, mask="auto")
    return top + height


def references(pdf, items, top=736):
    line(pdf, MARGIN, top - 9, WIDTH - MARGIN, top - 9)
    x = MARGIN
    for text, target in items:
        n = size(text, "Helvetica", 8.5)
        label(pdf, text, x, top, 8.5, TEAL)
        pdf.linkURL(target, (x, HEIGHT - top - 10, x + n, HEIGHT - top + 1), relative=0)
        x += n + 17
    if x > WIDTH - MARGIN + 17:
        raise ValueError("Reference labels exceed the page")


def page(pdf, number, key, topic, headline, intro=None):
    pdf.bookmarkPage(key)
    pdf.addOutlineEntry(headline.replace("<br/>", " "), key)
    label(pdf, topic, MARGIN, 31, 9.4, TEAL, "Helvetica-Bold")
    line(pdf, MARGIN, 50, WIDTH - MARGIN, 50)
    end = paragraph(pdf, headline, MARGIN, 68, points=31, leading=33, face="Helvetica-Bold", limit=70)
    if intro is not None:
        paragraph(pdf, intro, MARGIN, end + 9, points=11.1, leading=15.7, color=MUTED, limit=49)
    line(pdf, MARGIN, 761, WIDTH - MARGIN, 761)
    label(pdf, "LLM feature manifolds", MARGIN, 766, 8.1, MUTED)
    folio = f"{number} / 8"
    label(pdf, folio, WIDTH - MARGIN - size(folio, "Helvetica", 8.1), 766, 8.1, MUTED)


def next_steps(pdf, left, right, top, height):
    gap = 13
    width = (BODY - gap) / 2
    for i, (heading, copy) in enumerate((left, right)):
        x = MARGIN + i * (width + gap)
        panel(pdf, x, top, width, height, TEAL_BG if i == 0 else PALE)
        label(pdf, heading, x + 13, top + 13, 11.0, TEAL if i == 0 else INK, "Helvetica-Bold")
        paragraph(pdf, copy, x + 13, top + 33, width - 26, points=10.25, leading=14.3, limit=height - 38)


def overview(pdf):
    page(pdf, 1, "overview", "Research walkthrough", "From hidden states<br/>to causal tests",
         "A visual guide to what we built, how the calendar experiments worked, and what the results support.")
    for i, (value, title, body) in enumerate((
        ("1,000", "Published GPT-2 groups", "Each reconstructed cloud was ranked while token identities were hidden."),
        ("99.7%", "Held-out weekday readout", "1,363 of 1,367 named weekdays landed nearest the expected angle in news."),
        ("~80%", "February after an edit", "An 11-dimensional edit made February the top answer on the framed Mistral prompt."),
    )):
        width = (BODY - 2 * 11) / 3
        x = MARGIN + i * (width + 11)
        panel(pdf, x, 209, width, 139, TEAL_BG if i == 1 else PALE)
        label(pdf, value, x + 13, 221, 31, TEAL, "Helvetica-Bold")
        label(pdf, title, x + 13, 264, 10.2, INK, "Helvetica-Bold")
        paragraph(pdf, body, x + 13, 284, width - 26, 10.2, 13.8, limit=55)
    label(pdf, "Three questions, three tests", MARGIN, 379, 19, INK, "Helvetica-Bold")
    links = (
        ("Find a shape", "Rank geometry and inspect which known concepts surface.", "search", "p. 2"),
        ("Recover it again", "Train a separate sparse autoencoder on unlabeled activations.", "training", "p. 5"),
        ("Test a causal edit", "Change one hidden state and measure the downstream answer.", "intervention", "p. 6"),
    )
    for i, (title, body, target, number) in enumerate(links):
        y = 420 + i * 61
        line(pdf, MARGIN, y, WIDTH - MARGIN, y)
        label(pdf, title, MARGIN, y + 10, 12.1, TEAL, "Helvetica-Bold")
        label(pdf, body, MARGIN, y + 30, 10.75, MUTED)
        label(pdf, number, WIDTH - MARGIN - 29, y + 14, 10, TEAL, "Helvetica-Bold")
        pdf.linkRect(title, target, (MARGIN, HEIGHT - y - 55, WIDTH - MARGIN, HEIGHT - y), relative=0, thickness=0)
    line(pdf, MARGIN, 603, WIDTH - MARGIN, 603)
    paragraph(pdf, "GPT-2-small supplies the blind discovery and autoencoder experiments. "
              "The causal benchmark uses a separate 4-bit Mistral-7B-v0.1 model with a prompt it can answer correctly.",
              MARGIN, 629, points=11.1, leading=15.4, limit=48)
    references(pdf, [("Experiment files", REPOSITORY), ("Research question", MAISI), ("Engels et al.", ENGELS)])
    pdf.showPage()


def search(pdf):
    page(pdf, 2, "search", "Discovery design", "How we looked for a manifold",
         "A candidate is a repeatable shape in a group of activations. A feature label is added only after the geometry has been scored.")
    steps = (
        ("Freeze GPT-2", "Save pre-block layer-7 residuals from natural text."),
        ("Encode sparsely", "Use the published 24,576-latent sparse autoencoder."),
        ("Reconstruct a group", "Add the active latent values times their decoder directions."),
        ("Rank the clouds", "Score shape from distances, using anonymous samples."),
    )
    gap = 14
    width = (BODY - 3 * gap) / 4
    for i, (title, body) in enumerate(steps):
        x = MARGIN + i * (width + gap)
        panel(pdf, x, 165, width, 102, TEAL_BG if i == 3 else PALE)
        label(pdf, title, x + 9, 178, 10.55, INK, "Helvetica-Bold")
        paragraph(pdf, body, x + 9, 199, width - 18, 9.6, 13.2, limit=59)
        if i < len(steps) - 1:
            arrow(pdf, x + width + 2, x + width + gap - 2, 216)
    panel(pdf, MARGIN, 287, BODY, 83)
    label(pdf, "One point in the cloud", MARGIN + 13, 299, 11.0, TEAL, "Helvetica-Bold")
    label(pdf, "group reconstruction = sum of (latent value x decoder direction) in the group",
          MARGIN + 13, 321, 11.55, INK)
    label(pdf, "Each point is the group's contribution at one sampled token position.", MARGIN + 13, 343, 10.5, MUTED)
    label(pdf, "What the geometry scorer sees", MARGIN, 395, 17, INK, "Helvetica-Bold")
    paragraph(pdf, "A loop score measures how long a hole persists as we connect points at increasing distance "
              "thresholds. The prototype variant first replaces nearby points with cluster centers. Using several "
              "scorers lets us see which ranks depend on that choice.", MARGIN, 422, limit=52)
    label(pdf, "What the evaluation adds", MARGIN, 496, 17, INK, "Helvetica-Bold")
    next_steps(pdf,
               ("Independent text", "Fit in the Pile discovery split. Apply fixed coordinates to separate Pile, WikiText, and news samples. Cap ranked geometry samples per document."),
               ("Labels after scores", "Open token identities after saving the rank table. Orient the calendar axes on discovery examples, then compare held-out angles with the known order."),
               527, 124)
    paragraph(pdf, "A score can surface a circle, a document-specific cluster, or a shape with an unclear interpretation. "
              "The comparison across corpora tells us how consistently it surfaces.", MARGIN, 676, points=10.8, leading=14.5, limit=36)
    references(pdf, [("Discovery procedure", f"{REPOSITORY}/blob/main/README.md"), ("Published feature groups", f"{ENGELS}#S4")])
    pdf.showPage()


def discovery(pdf):
    page(pdf, 3, "discovery", "Discovery results", "A readable circle can rank low",
         "The left panel asks whether a scorer finds the group blind. The right panel asks whether a known group preserves calendar order.")
    end = figure(pdf, "overview/discovery-and-transfer.png", MARGIN, 162)
    paragraph(pdf, "Rank 1 is best. The same fixed weekday projection decoded 1,363 of 1,367 "
              "named news tokens from 466 documents. Under prototype persistence the group placed first in news, "
              "with ranks of 251 to 672 in the other three corpora.", MARGIN, end + 10, limit=51)
    figure(pdf, "overview/ranking-methods.png", MARGIN, 431)
    paragraph(pdf, "The month group ranked first by the geometric mean of raw ranks across four corpora. "
              "A score based on mixture and separability gave a different ordering. Each cell above is a rank out of 1,000.",
              MARGIN, 613, points=11.0, leading=15.5, limit=48)
    paragraph(pdf, "<b>Sampling check.</b> A single document supplied 94.7% of named weekday tokens in the Pile holdout. "
              "The news error rate was measured across hundreds of documents.", MARGIN, 678, points=10.5, leading=14.4, limit=33)
    references(pdf, [("Rank and label data", f"{REPOSITORY}/blob/main/RESULTS.md#discovery-without-labels"),
                     ("GPT-2 notebook", f"{REPOSITORY}/blob/main/notebooks/01_blind_discovery.ipynb")])
    pdf.showPage()


def point_clouds(pdf):
    page(pdf, 4, "clouds", "Label reveal", "What the fixed projection shows",
         "The Pile discovery examples set the axes. Each news point is then projected into those same coordinates. Token names color the points after scoring.")
    end = figure(pdf, "document-controls/calendar-geometry.png", MARGIN, 155)
    paragraph(pdf, "Each dot is a token-position reconstruction using one SAE group. The large outlined marks are "
              "centers of named day or month classes. The weekday ordering is clear in new text. The month groups "
              "overlap more, with 52.5% of named news tokens closest to their expected angle.",
              MARGIN, end + 11, points=10.7, leading=15.0, limit=59)
    paragraph(pdf, "<b>Readout scope.</b> The score applies to the named tokens present in the held-out sample. "
              "Shortened or differently cased names were harder to read with the same coordinates.",
              MARGIN, 660, points=10.65, leading=15.0, limit=45)
    references(pdf, [("News label data", f"{REPOSITORY}/blob/main/results/document-controls/calendar-results.csv"),
                     ("Calendar spelling tests", f"{REPOSITORY}/tree/main/results/format-transfer")])
    pdf.showPage()


def training(pdf):
    page(pdf, 5, "training", "Dictionary learning", "What a new SAE kept",
         "A new 3,072-latent TopK dictionary learned from the same type of unlabeled GPT-2 activations. The GPT-2 weights stayed frozen.")
    next_steps(pdf,
               ("Matched training design", "Architecture, training data, initialization within a seed, batch draws, number of steps, and the held-out calendar projection."),
               ("What changes", "The reconstruction loss. Compare ordinary mean squared error with square-root and full reweighting by the residual size of each token."),
               160, 107)
    end = figure(pdf, "token-loss/matched-loss-tradeoff.png", MARGIN, 293)
    measured = read_csv("token-loss/matched-loss-geometry.csv")
    scores = {}
    for objective in ("mse", "relative-token"):
        values = [float(r["document_macro_accuracy"]) for r in measured
                  if r["loss"] == objective and r["split"] == "news-test" and r["concept"] == "Weekdays"]
        scores[objective] = 100 * sum(values) / len(values)
    paragraph(pdf, f"<b>{scores['mse']:.1f}% to {scores['relative-token']:.1f}%</b> mean weekday angle accuracy in news, "
              "using full per-token weighting. Each dot is one of three training seeds. The angle metric gives documents equal weight.",
              MARGIN, end + 11, limit=48)
    panel(pdf, MARGIN, 570, BODY, 117, OCHRE_BG)
    label(pdf, "Why report two metrics?", MARGIN + 13, 584, 11.5, OCHRE, "Helvetica-Bold")
    paragraph(pdf, "Whole-validation FVU asks how much squared variation the SAE misses. A small global error can "
              "coexist with large errors on the tokens that carry the geometry. In the original TopK run, FVU was "
              "1.35% across validation and 43% on named weekdays in the Pile holdout. The two denominators use their own fit means.",
              MARGIN + 13, 609, BODY - 26, 10.6, 15.0, limit=64)
    references(pdf, [("Matched training results", f"{REPOSITORY}/blob/main/RESULTS.md#what-the-new-autoencoders-recovered"),
                     ("Goodfire on SAEs", SAES)])
    pdf.showPage()


def intervention(pdf):
    page(pdf, 6, "intervention", "Causal design", "What exactly did we edit?",
         "Mistral uses the paper's framing to make the calendar task answerable. The edited run keeps the original prompt and changes one internal state.")
    prompt = read_json("steering/harmonics/harmonic-metadata.json")["exact_example"][0]["prompt"]
    for i, (role, token, answer, color) in enumerate((
        ("Original prompt", "November", "January", BLUE), ("Donor prompt", "December", "February", TEAL),
    )):
        y = 166 + i * 77
        panel(pdf, MARGIN, y, BODY, 66, PALE if i == 0 else TEAL_BG)
        label(pdf, role, MARGIN + 13, y + 10, 9.1, color, "Helvetica-Bold")
        paragraph(pdf, prompt.replace("November", f"<b>{token}</b>"),
                  MARGIN + 13, y + 34, 389, 11.0, 15.0, limit=24)
        label(pdf, answer, WIDTH - MARGIN - 77, y + 33, 12.0, color, "Helvetica-Bold")
    label(pdf, "A patch at one layer and token", MARGIN, 345, 17, INK, "Helvetica-Bold")
    for i, (title, body) in enumerate((
        ("Measure the donor difference", "Subtract the November hidden state from the December hidden state."),
        ("Choose the edit direction", "Project the difference into the calendar plane or the larger subspace."),
        ("Replay the original prompt", "Add the projected difference at the November token. Read the next-token scores."),
    )):
        y = 378 + i * 61
        line(pdf, MARGIN, y, WIDTH - MARGIN, y)
        label(pdf, title, MARGIN, y + 8, 11.3, TEAL, "Helvetica-Bold")
        paragraph(pdf, body, MARGIN, y + 26, points=10.6, leading=14.7, limit=31)
    next_steps(pdf,
               ("Norm-matched controls", "Compare with random and PCA directions at the same edit length. A full donor patch measures what the entire state difference can do."),
               ("Two outcome measures", "On the exact prompt, measure February's full-vocabulary probability. Across many pairs, measure the target logit minus the original logit."),
               576, 120)
    references(pdf, [("Intervention code", f"{REPOSITORY}/blob/main/steer_months.py"),
                     ("Calendar precedent", f"{ENGELS}#S5"), ("Goodfire steering", STEERING)])
    pdf.showPage()


def causal_results(pdf):
    page(pdf, 7, "causal", "Causal results", "When February became the answer",
         "The exact prompt and the average effect answer different questions. The panels also use different intervention layers.")
    end = figure(pdf, "overview/calendar-edits.png", MARGIN, 164)
    paragraph(pdf, "<b>Layer 14, exact November prompt.</b> The first two calendar coordinates left February near "
              "2%. Expanding to 11 dimensions made February the top token at about 80%. The 11-dimensional PCA control "
              "did about as well. Three random edits at the same norm averaged close to 2%.",
              MARGIN, end + 14, limit=66)
    natural = read_json("steering/published-sae/sae-plane-steering.json")
    entries = {r["condition"]: r["mean_target_gap_change"] for r in natural["aggregate"] if r["donor_shift"] == 1}
    paragraph(pdf, f"<b>Layer 8, average across 72 one-month edits.</b> A plane fit on unlabeled natural-text "
              f"reconstructions shifted the target logit gap by {entries['natural_text_plane']:.4f} on average. The matched "
              f"random planes shifted it by {entries['random_plane_equal_norm']:.3f}. Error bars resample source months.",
              MARGIN, 463, limit=64)
    next_steps(pdf,
               ("Causal influence", "The natural-text geometry carries some information that the downstream computation uses. Its average effect is measurable and smaller than the full donor patch."),
               ("Dimensionality", "The chosen two-dimensional plane explains only part of the effect. The larger PCA control leaves the role of the precise calendar shape open."),
               575, 122)
    references(pdf, [("Paired steering results", f"{REPOSITORY}/blob/main/RESULTS.md#what-changed-mistrals-answer"),
                     ("Steering notebook", f"{REPOSITORY}/blob/main/notebooks/03_calendar_steering.ipynb")])
    pdf.showPage()


def read_next(pdf):
    page(pdf, 8, "reading", "Interpretation and reproduction", "How to read the evidence",
         "The calendar experiments separate discovery, semantic interpretation, and causal testing. Each stage answers a different question.")
    columns = (
        ("Rank", "Does a candidate score highly before token identities are opened?", "Months ranked well in several raw-score runs. Weekday ranks depended on the corpus and method."),
        ("Readout", "Does the geometry order the named concepts in independent text?", "The selected weekday group transferred to held-out news. Calendar spelling changes weakened some readouts."),
        ("Edit", "Does moving the state change an answer relative to controls?", "The natural-text plane changed the average target gap. The larger subspace changed the exact answer."),
    )
    for i, (name, question, result) in enumerate(columns):
        top = 161 + i * 96
        panel(pdf, MARGIN, top, BODY, 83, TEAL_BG if i == 2 else PALE)
        label(pdf, name, MARGIN + 12, top + 14, 13.0, TEAL, "Helvetica-Bold")
        paragraph(pdf, f"<b>{question}</b><br/>{result}", MARGIN + 112, top + 12, BODY - 128,
                  10.65, 15.0, limit=63)
    label(pdf, "Run the experiment in pieces", MARGIN, 470, 17, INK, "Helvetica-Bold")
    paragraph(pdf, "The repository contains the raw score tables, trained SAE weights, plotting code, and three executed "
              "notebooks with small data extracts. The first notebook walks through ranking and label reveal. The next two "
              "cover training and steering.", MARGIN, 498, limit=50)
    label(pdf, "Where Goodfire fits", MARGIN, 570, 17, INK, "Helvetica-Bold")
    paragraph(pdf, "The overview motivates looking for structure across multiple coordinates. The steering article motivates "
              "comparing paths and norm-matched edits. The story study motivates tracking a representation over prompt prefixes.",
              MARGIN, 597, limit=50)
    paragraph(pdf, "A follow-up search can use the same procedure. Fix a candidate from unlabeled geometry, then inspect "
              "its contexts and test targeted edits on fresh prompts.", MARGIN, 668, points=10.8, leading=14.7, limit=36)
    references(pdf, [("Jupyter tutorials", f"{REPOSITORY}/tree/main/notebooks"), ("Geometry overview", GEOMETRY),
                     ("Steering study", STEERING), ("Story trajectories", STORIES)])
    pdf.showPage()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/calendar-study-walkthrough.pdf")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(args.output), pagesize=letter, invariant=True, pageCompression=1)
    pdf.setTitle("From Hidden States to Causal Tests")
    pdf.setSubject("A visual walkthrough of the GPT-2 and Mistral calendar feature manifold experiments")
    for draw in (overview, search, discovery, point_clouds, training, intervention, causal_results, read_next):
        draw(pdf)
    pdf.save()


if __name__ == "__main__":
    main()
