/*
 * Build paper/submission/hausa-med-qa.docx from the document model that
 * src/paper_docx.py emits. The model is generated from paper/main.tex, so the
 * Word file is never edited by hand -- regenerate it instead, or it drifts
 * from the LaTeX the moment a result changes.
 *
 *   python -c "from src import paper_docx as P; P.write_model('paper/submission/paper_model.json')"
 *   node tools/build_docx.js paper/submission/paper_model.json paper/submission/hausa-med-qa.docx
 *
 * `docx` is resolved from NODE_PATH or a local node_modules.
 */
const fs = require("fs");
const path = require("path");
const D = require("docx");

const [, , modelPath, outPath] = process.argv;
if (!modelPath || !outPath) {
  console.error("usage: node build_docx.js <model.json> <out.docx>");
  process.exit(2);
}
const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const root = process.cwd();

const FONT = "Calibri";
const runs = (rs, opts = {}) =>
  (rs || []).map(
    (r) =>
      new D.TextRun({
        text: r.text,
        bold: r.bold || opts.bold || false,
        italics: r.italic || opts.italics || false,
        font: r.mono ? "Consolas" : FONT,
        size: opts.size || 22,
      }),
  );

// Figures are sized to the text column: 6.5in at 96dpi, scaled by aspect ratio.
const PNG_COLUMN_PX = 624;
function imageParagraph(rel) {
  const abs = path.resolve(root, rel);
  const buf = fs.readFileSync(abs);
  // PNG header: width and height are big-endian uint32 at bytes 16 and 20.
  const w = buf.readUInt32BE(16);
  const h = buf.readUInt32BE(20);
  const width = PNG_COLUMN_PX;
  const height = Math.round((h / w) * width);
  return new D.Paragraph({
    alignment: D.AlignmentType.CENTER,
    spacing: { before: 180, after: 60 },
    children: [
      new D.ImageRun({ type: "png", data: buf, transformation: { width, height } }),
    ],
  });
}

function caption(rs, label, n) {
  return new D.Paragraph({
    alignment: D.AlignmentType.CENTER,
    spacing: { after: 220 },
    children: [
      new D.TextRun({ text: `${label} ${n}. `, bold: true, font: FONT, size: 18 }),
      ...runs(rs, { size: 18 }),
    ],
  });
}

function table(block) {
  const cols = Math.max(...block.rows.map((r) => r.length));
  const total = 9360; // 6.5in in DXA
  const widths = Array(cols).fill(Math.floor(total / cols));
  widths[0] += total - widths.reduce((a, b) => a + b, 0);
  return new D.Table({
    columnWidths: widths,
    width: { size: total, type: D.WidthType.DXA },
    rows: block.rows.map(
      (row, i) =>
        new D.TableRow({
          tableHeader: i === 0,
          children: Array.from({ length: cols }, (_, j) => {
            const cell = row[j] || [];
            return new D.TableCell({
              width: { size: widths[j], type: D.WidthType.DXA },
              shading:
                i === 0
                  ? { type: D.ShadingType.CLEAR, fill: "F2F2F2" }
                  : undefined,
              margins: { top: 60, bottom: 60, left: 100, right: 100 },
              children: [
                new D.Paragraph({
                  spacing: { before: 0, after: 0 },
                  children: runs(cell, { size: 20, bold: i === 0 }),
                }),
              ],
            });
          }),
        }),
    ),
  });
}

const children = [];
children.push(
  new D.Paragraph({
    heading: D.HeadingLevel.TITLE,
    alignment: D.AlignmentType.CENTER,
    children: [new D.TextRun({ text: model.title, font: FONT, size: 36, bold: true })],
  }),
);
model.authors.forEach((a) =>
  children.push(
    new D.Paragraph({
      alignment: D.AlignmentType.CENTER,
      spacing: { after: 0 },
      children: [new D.TextRun({ text: a, font: FONT, size: 22 })],
    }),
  ),
);

children.push(
  new D.Paragraph({
    spacing: { before: 320, after: 80 },
    children: [new D.TextRun({ text: "Abstract", bold: true, font: FONT, size: 24 })],
  }),
  new D.Paragraph({
    alignment: D.AlignmentType.JUSTIFIED,
    spacing: { after: 240 },
    indent: { left: 400, right: 400 },
    children: runs(model.abstract, { size: 21 }),
  }),
);

let figN = 0;
let tabN = 0;
for (const b of model.blocks) {
  if (b.type === "heading") {
    children.push(
      new D.Paragraph({
        heading: D.HeadingLevel.HEADING_1,
        children: runs(b.runs, { size: 28, bold: true }),
      }),
    );
  } else if (b.type === "paragraph") {
    children.push(
      new D.Paragraph({
        alignment: D.AlignmentType.JUSTIFIED,
        spacing: { after: 140 },
        children: runs(b.runs),
      }),
    );
  } else if (b.type === "bullet") {
    children.push(
      new D.Paragraph({
        numbering: { reference: "bullets", level: 0 },
        spacing: { after: 80 },
        children: runs(b.runs),
      }),
    );
  } else if (b.type === "figure") {
    figN += 1;
    children.push(imageParagraph(b.image), caption(b.caption, "Figure", figN));
  } else if (b.type === "table") {
    tabN += 1;
    children.push(table(b), caption(b.caption, "Table", tabN));
  }
}

children.push(
  new D.Paragraph({
    heading: D.HeadingLevel.HEADING_1,
    children: [new D.TextRun({ text: "References", font: FONT, size: 28, bold: true })],
  }),
);
model.references.forEach((r, i) =>
  children.push(
    new D.Paragraph({
      spacing: { after: 80 },
      indent: { left: 360, hanging: 360 },
      children: [new D.TextRun({ text: `[${i + 1}] ${r}`, font: FONT, size: 20 })],
    }),
  ),
);

const doc = new D.Document({
  styles: {
    default: {
      title: {
        run: { font: FONT, size: 36, bold: true },
        paragraph: { spacing: { before: 0, after: 160 }, keepNext: true },
      },
      heading1: {
        run: { font: FONT, size: 28, bold: true },
        paragraph: { spacing: { before: 280, after: 120 }, keepNext: true },
      },
      heading2: {
        run: { font: FONT, size: 24, bold: true },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true },
      },
    },
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0,
            format: D.LevelFormat.BULLET,
            text: "•",
            alignment: D.AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 11906, height: 16838 }, // A4, matching the LaTeX
          margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 },
        },
      },
      children,
    },
  ],
});

D.Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(path.resolve(root, outPath)), { recursive: true });
  fs.writeFileSync(path.resolve(root, outPath), buf);
  console.log(
    `wrote ${outPath}  (${(buf.length / 1024).toFixed(0)} KB, ` +
      `${children.length} blocks, ${figN} figures, ${tabN} table, ` +
      `${model.references.length} references)`,
  );
});
