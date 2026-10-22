// Word-level diff of two provision texts (longest common subsequence).

type Part = { kind: "same" | "added" | "removed"; text: string };

const MAX_CELLS = 1_500_000;

export function diffWords(before: string, after: string): Part[] | null {
  const a = before.split(/\s+/).filter(Boolean);
  const b = after.split(/\s+/).filter(Boolean);
  if (a.length * b.length > MAX_CELLS) return null;

  // lengths[i][j] = LCS length of a[i:] and b[j:]
  const lengths = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      lengths[i][j] = a[i] === b[j] ? lengths[i + 1][j + 1] + 1 : Math.max(lengths[i + 1][j], lengths[i][j + 1]);
    }
  }

  const parts: Part[] = [];
  const push = (kind: Part["kind"], word: string) => {
    const last = parts[parts.length - 1];
    if (last?.kind === kind) last.text += ` ${word}`;
    else parts.push({ kind, text: word });
  };
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      push("same", a[i]);
      i++;
      j++;
    } else if (lengths[i + 1][j] >= lengths[i][j + 1]) {
      push("removed", a[i++]);
    } else {
      push("added", b[j++]);
    }
  }
  while (i < a.length) push("removed", a[i++]);
  while (j < b.length) push("added", b[j++]);
  return parts;
}

export function WordDiff({ before, after }: { before: string; after: string }) {
  const parts = diffWords(before, after);
  if (!parts) return <p className="hint">These sections are too long to compare word by word. Read them side by side instead.</p>;
  return (
    <div className="law-text diff">
      <p>
        {parts.map((part, index) => {
          const text = `${index ? " " : ""}${part.text}`;
          if (part.kind === "added") return <ins key={index}>{text}</ins>;
          if (part.kind === "removed") return <del key={index}>{text}</del>;
          return <span key={index}>{text}</span>;
        })}
      </p>
    </div>
  );
}
