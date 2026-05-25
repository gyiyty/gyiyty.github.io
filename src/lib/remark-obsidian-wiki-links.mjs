import { visit } from "unist-util-visit";

function slugifyWikiTarget(value) {
  return value
    .trim()
    .replace(/\.md$/i, "")
    .replace(/[\\/#?]+/g, " ")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "-");
}

export function remarkObsidianWikiLinks() {
  return (tree) => {
    visit(tree, "text", (node, index, parent) => {
      if (!parent || typeof node.value !== "string" || !node.value.includes("[[")) return;

      const parts = [];
      const pattern = /\[\[([^\]]+)\]\]/g;
      let lastIndex = 0;
      let match;

      while ((match = pattern.exec(node.value))) {
        if (match.index > lastIndex) {
          parts.push({ type: "text", value: node.value.slice(lastIndex, match.index) });
        }

        const [rawTarget, rawLabel] = match[1].split("|");
        const target = rawTarget.trim();
        const label = (rawLabel || rawTarget).trim();
        const anchorIndex = target.indexOf("#");
        const pageTarget = anchorIndex >= 0 ? target.slice(0, anchorIndex) : target;
        const section = anchorIndex >= 0 ? target.slice(anchorIndex + 1) : "";
        const href = pageTarget
          ? `/notes/${slugifyWikiTarget(pageTarget)}/${section ? `#${encodeURIComponent(section)}` : ""}`
          : `#${encodeURIComponent(section)}`;

        parts.push({
          type: "link",
          url: href,
          title: null,
          children: [{ type: "text", value: label }]
        });
        lastIndex = pattern.lastIndex;
      }

      if (lastIndex < node.value.length) {
        parts.push({ type: "text", value: node.value.slice(lastIndex) });
      }

      if (parts.length > 0) parent.children.splice(index, 1, ...parts);
    });
  };
}
