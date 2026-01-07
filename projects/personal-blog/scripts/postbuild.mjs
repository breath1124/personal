import { writeFile } from "node:fs/promises";
import { join } from "node:path";

const distDir = join(process.cwd(), "dist");
await writeFile(join(distDir, ".nojekyll"), "");

