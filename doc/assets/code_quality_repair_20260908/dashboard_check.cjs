const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "../../..");
const base = "http://dashboard.test";
const results = [];

async function main() {
  const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
  const browser = await chromium.launch({ headless: true,
    ...(fs.existsSync(edge) ? { executablePath: edge } : {}) });
  try {
    for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
      const page = await browser.newPage({ viewport });
      const errors = [];
      page.on("pageerror", error => errors.push(error.message));
      const resource = { ts: new Date().toISOString(), root_pid: null, processes: [],
        gpu_util_pct: null, gpu_memory_used_mib: null, gpu_memory_total_mib: null };
      const progress = [
        { type: "manifest", total_folds: 1, run_name: "software-check" },
        { type: "fold", fold: 0, n_folds_done: 1, hit: null, fold_auc: null },
        { type: "done" },
      ].map(row => JSON.stringify(row)).join("\n");
      await page.route(`${base}/**`, async route => {
        const name = new URL(route.request().url()).pathname;
        if (name === "/dashboard.html" || name === "/chart.umd.min.js") {
          await route.fulfill({ path: path.join(root, "experiments", name.slice(1)) });
          return;
        }
        let body = {};
        if (name === "/api/status") body = { ok: true, port: 18812, latest_run: "software-check" };
        else if (name === "/api/runs") body = { ok: true, runs: [{ name: "software-check", done: true }] };
        else if (name.startsWith("/api/monitor/")) body = { ok: true, client_id: "test", state: "offline" };
        else if (name.endsWith("/record.json")) body = { auc_mean: null, hit_rate_mean: null };
        else if (name.endsWith("/resources.latest.json")) body = resource;
        else if (name.endsWith(".jsonl")) {
          await route.fulfill({ contentType: "text/plain", body: name.endsWith("/progress.jsonl") ? progress :
            name.endsWith("/resources.jsonl") ? JSON.stringify(resource) : "" });
          return;
        }
        await route.fulfill({ json: body });
      });
      await page.goto(`${base}/dashboard.html?run=software-check`);
      await page.waitForFunction(() => document.querySelectorAll("#kpis .kpi").length === 9);
      const values = await page.locator("#kpis .v").allTextContents();
      const labels = await page.locator("#kpis .l").allTextContents();
      const trends = await page.evaluate(() => Chart.getChart("c1").data.datasets.map(item => item.data));
      const sources = await page.locator("#sources").innerText();
      const fit = await page.locator("#kpis .kpi").evaluateAll(items =>
        items.every(item => item.scrollWidth <= item.clientWidth));
      await page.screenshot({ path: path.join(__dirname, `dashboard-${viewport.width}.png`), fullPage: true });
      results.push({ viewport, values, labels, trends, sources, textFits: fit, errors });
      assert.equal(values[2], "\u2014");
      assert.equal(values[3], "\u2014");
      assert.equal(values[5], "\u2014%");
      assert.equal(values[6], "\u2014 MiB");
      assert.equal(values[8], "\u2014");
      assert.deepEqual(trends, [[null], [null]]);
      assert.match(sources, /worker private \u2014 GiB/);
      assert.equal(fit, true);
      assert.deepEqual(errors, []);
      await page.close();
    }
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(__dirname, "dashboard-results.json"), JSON.stringify(results, null, 2) + "\n");
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
