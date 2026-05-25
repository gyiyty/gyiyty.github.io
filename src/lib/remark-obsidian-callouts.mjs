import { visit } from "unist-util-visit";

const defaultTitles = {
  abstract: "Abstract",
  bug: "Bug",
  caution: "Caution",
  check: "Success",
  danger: "Danger",
  done: "Done",
  error: "Error",
  example: "Example",
  fail: "Failure",
  failure: "Failure",
  faq: "Question",
  help: "Question",
  highlight: "Highlight",
  important: "Important",
  info: "Info",
  missing: "Missing",
  note: "Note",
  question: "Question",
  quote: "Quote",
  success: "Success",
  summary: "Summary",
  tip: "Tip",
  todo: "Todo",
  warning: "Warning"
};

function cleanType(type) {
  return type.toLowerCase().replace(/[^\w-]/g, "");
}

function paragraphIsEmpty(paragraph) {
  return paragraph.children.every((child) => child.type === "text" && child.value.trim() === "");
}

function titleNode(title) {
  return {
    type: "paragraph",
    data: {
      hProperties: {
        className: ["callout-title"]
      }
    },
    children: [{ type: "text", value: title }]
  };
}

export function remarkObsidianCallouts() {
  return (tree) => {
    visit(tree, "blockquote", (node) => {
      const firstChild = node.children?.[0];
      if (!firstChild || firstChild.type !== "paragraph") return;

      const firstTextIndex = firstChild.children.findIndex((child) => child.type === "text");
      if (firstTextIndex < 0) return;

      const firstText = firstChild.children[firstTextIndex];
      const match = firstText.value.match(/^\s*\[!([^\]\s]+)\]([+-]?)(?:[ \t]+([^\r\n]+))?(?:\r?\n)?/);
      if (!match) return;

      const type = cleanType(match[1]);
      if (!type) return;

      const customTitle = match[3]?.trim();
      const title = customTitle || defaultTitles[type] || type[0].toUpperCase() + type.slice(1);
      firstText.value = firstText.value.slice(match[0].length);

      if (firstText.value === "") {
        firstChild.children.splice(firstTextIndex, 1);
      }
      if (firstChild.children.length === 0 || paragraphIsEmpty(firstChild)) {
        node.children.shift();
      }

      node.children.unshift(titleNode(title));
      node.data = {
        ...node.data,
        hName: "aside",
        hProperties: {
          ...(node.data?.hProperties || {}),
          className: ["callout", `callout-${type}`],
          "data-callout": type,
          ...(match[2] ? { "data-callout-fold": match[2] } : {})
        }
      };
    });
  };
}
