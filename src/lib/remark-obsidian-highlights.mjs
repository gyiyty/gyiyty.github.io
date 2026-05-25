import { visit } from "unist-util-visit";

function markerNode(children) {
  return {
    type: "emphasis",
    data: {
      hName: "mark",
      hProperties: {
        className: ["obsidian-highlight"]
      }
    },
    children
  };
}

function splitTextHighlight(value) {
  if (!value.includes("==")) return null;

  const nodes = [];
  let cursor = 0;
  let openIndex = value.indexOf("==");

  while (openIndex !== -1) {
    const closeIndex = value.indexOf("==", openIndex + 2);
    if (closeIndex === -1) break;

    if (openIndex > cursor) {
      nodes.push({ type: "text", value: value.slice(cursor, openIndex) });
    }

    const highlighted = value.slice(openIndex + 2, closeIndex);
    if (highlighted.length > 0) {
      nodes.push(markerNode([{ type: "text", value: highlighted }]));
    } else {
      nodes.push({ type: "text", value: "====" });
    }

    cursor = closeIndex + 2;
    openIndex = value.indexOf("==", cursor);
  }

  if (cursor === 0) return null;
  if (cursor < value.length) {
    nodes.push({ type: "text", value: value.slice(cursor) });
  }
  return nodes;
}

export function remarkObsidianHighlights() {
  return (tree) => {
    visit(tree, "text", (node, index, parent) => {
      if (!parent || typeof index !== "number" || typeof node.value !== "string") return;
      const replacement = splitTextHighlight(node.value);
      if (!replacement) return;
      parent.children.splice(index, 1, ...replacement);
    });
  };
}
