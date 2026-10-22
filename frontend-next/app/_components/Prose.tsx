import { Fragment, ReactNode } from "react";

// Minimal, safe renderer for the model's light markdown: paragraphs, bullet
// lists and **bold**. No HTML injection.
const LIST_ITEM = /^\s*([-*•]|\d+[.)])\s+/;

function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, index) =>
    part.startsWith("**") && part.endsWith("**") ? <strong key={index}>{part.slice(2, -2)}</strong> : <Fragment key={index}>{part}</Fragment>,
  );
}

type Block = { kind: "p" | "ul"; lines: string[] };

// Blank lines separate chunks; inside a chunk, consecutive lines of the same
// kind (prose or list item) form one block.
function toBlocks(text: string): Block[] {
  return text.trim().split(/\n\s*\n/).flatMap((chunk) => {
    const blocks: Block[] = [];
    for (const line of chunk.split("\n").filter((item) => item.trim())) {
      const kind = LIST_ITEM.test(line) ? "ul" : "p";
      const last = blocks[blocks.length - 1];
      if (last?.kind === kind) last.lines.push(line);
      else blocks.push({ kind, lines: [line] });
    }
    return blocks;
  });
}

export function Prose({ text }: { text: string }) {
  return (
    <>
      {toBlocks(text).map((block, index) =>
        block.kind === "ul" ? (
          <ul key={index}>
            {block.lines.map((line, item) => <li key={item}>{inline(line.replace(LIST_ITEM, ""))}</li>)}
          </ul>
        ) : (
          <p key={index}>{inline(block.lines.join(" "))}</p>
        ),
      )}
    </>
  );
}
