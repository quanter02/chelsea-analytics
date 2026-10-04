// 사용: node factcheck/run.js <텍스트 파일>
const fs = require("fs"), path = require("path");
const FC = require("./engine.js");
const ref = JSON.parse(fs.readFileSync(path.join(__dirname, "reference.json"), "utf8"));
const text = fs.readFileSync(process.argv[2], "utf8");
const out = FC.run(text, ref);
if (process.argv.includes("--json")) { console.log(JSON.stringify(out.claims)); process.exit(0); }
for (const c of out.claims) console.log(`[${c.verdict}] ${c.country || "?"} ${c.year || "?"} ${c.name} "${c.raw}" → ${c.detail}`);
console.log(out.summary);
