import { spawn } from "node:child_process";
import path from "node:path";

const args = process.argv.slice(2);
const astroBin = path.resolve("node_modules", "astro", "astro.js");

const child = spawn(process.execPath, [astroBin, ...args], {
  stdio: "inherit",
  env: {
    ...process.env,
    ASTRO_TELEMETRY_DISABLED: "1"
  }
});

child.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
