import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const packageDir = process.argv[2];
if (!packageDir) {
  throw new Error("Usage: node build_manuscript_workbook.mjs <manuscript-package-dir>");
}

const tablesDir = path.join(packageDir, "tables");
const manifestPath = path.join(packageDir, "manuscript_readiness_manifest.json");
const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
if (manifest.status !== "ready_for_manuscript_writing") {
  throw new Error(`Package is not ready: ${manifest.status}`);
}
if (manifest.participant_level_outputs_written !== false || manifest.mice_performed !== false) {
  throw new Error("Privacy or analysis-policy guard failed.");
}

const tableSpecs = [
  ["table1_core_cohort_characteristics.csv", "Table 1 Core"],
  ["table2_nested_cv_model_performance.csv", "Table 2 Nested CV"],
  ["table3_final_model_configurations.csv", "Table 3 Configs"],
  ["table4_locked_internal_external_performance.csv", "Table 4 Performance"],
  ["table5_development_operating_thresholds.csv", "Table 5 Thresholds"],
  ["table6_three_layer_external_analysis.csv", "Table 6 External"],
  ["table_s1_all_predictor_characteristics.csv", "S1 Predictors"],
  ["table_s2_nested_outer_fold_metrics.csv", "S2 Outer Folds"],
  ["table_s3_external_reliability_bins.csv", "S3 Reliability"],
  ["table_s4_posthoc_sensitivity_metrics.csv", "S4 Sensitivity"],
  ["table_s5_posthoc_sensitivity_scenarios.csv", "S5 Scenarios"],
  ["table_s6_local_calibration_brier_decomposition.csv", "S6 Calibration"],
  ["table_s7_subgroup_performance.csv", "S7 Subgroups"],
  ["table_s8_subgroup_interaction_tests.csv", "S8 Interactions"],
  ["table_s9_mrmr_rank_stability.csv", "S9 mRMR Ranks"],
  ["table_s10_mrmr_set_stability.csv", "S10 mRMR Sets"],
  ["table_s11_elastic_net_stability.csv", "S11 Elastic Net"],
  ["table_s12_transportability_drift.csv", "S12 Transport"],
  ["table_s13_prevalence_scenario_ppv_npv.csv", "S13 Prevalence"],
];

for (const [file] of tableSpecs) {
  await fs.access(path.join(tablesDir, file));
}

function columnLabel(index) {
  let value = index + 1;
  let label = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    label = String.fromCharCode(65 + remainder) + label;
    value = Math.floor((value - 1) / 26);
  }
  return label;
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.getRange("A1:F1").merge();
readme.getRange("A1").values = [["MCI screening model - manuscript tables package"]];
readme.getRange("A1:F1").format = {
  fill: "#2F5D7C",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:F1").format.rowHeight = 30;
readme.getRange("A3:B10").values = [
  ["Readiness status", manifest.status],
  ["Education harmonization", manifest.education_harmonization_mode],
  ["Development selection", "Repeated nested CV on locked Development train-80 only"],
  ["External role", "Locked validation; local updating reported separately"],
  ["MICE", "Not performed at investigator request"],
  ["Participant-level exports", "None"],
  ["Confidence intervals", "Bootstrap intervals as specified in each table"],
  ["Interpretation boundary", manifest.interpretation_boundary],
];
readme.getRange("A3:A10").format = {
  fill: "#EAF0F4",
  font: { bold: true, color: "#263238" },
};
readme.getRange("A3:B10").format.borders = {
  preset: "inside",
  style: "thin",
  color: "#D8DEE2",
};
readme.getRange("A12:C12").values = [["Worksheet", "Source table", "Purpose"]];
readme.getRange("A12:C12").format = {
  fill: "#D98B45",
  font: { bold: true, color: "#FFFFFF" },
};
const purposes = {
  "Table 1 Core": "Primary cohort characteristics",
  "Table 2 Nested CV": "Development-only model-family selection",
  "Table 3 Configs": "Frozen model specifications",
  "Table 4 Performance": "Locked internal and External performance",
  "Table 5 Thresholds": "Development-derived operating thresholds",
  "Table 6 External": "Locked and locally updated External layers",
};
readme.getRange(`A13:C${12 + tableSpecs.length}`).values = tableSpecs.map(
  ([file, sheet]) => [sheet, file, purposes[sheet] ?? "Supplementary analysis table"],
);
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 58;
readme.getRange("C:C").format.columnWidth = 38;
readme.getRange("B3:B10").format.wrapText = true;
readme.freezePanes.freezeRows(1);

const catalog = [];
for (let index = 0; index < tableSpecs.length; index += 1) {
  const [file, sheetName] = tableSpecs[index];
  const csvText = await fs.readFile(path.join(tablesDir, file), "utf8");
  await workbook.fromCSV(csvText, { sheetName });
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  const used = sheet.getUsedRange(true);
  const values = used.values;
  const rows = values.length;
  const columns = values[0]?.length ?? 0;
  if (rows < 2 || columns < 1) {
    throw new Error(`Unexpected empty table: ${file}`);
  }
  const lastColumn = columnLabel(columns - 1);
  const fullRange = sheet.getRange(`A1:${lastColumn}${rows}`);
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#2F5D7C",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  header.format.rowHeight = 36;
  fullRange.format.font = { name: "Aptos", size: 10 };
  fullRange.format.borders = {
    insideHorizontal: { style: "thin", color: "#E5E9EC" },
    bottom: { style: "thin", color: "#AEB8BE" },
  };
  fullRange.format.autofitColumns();
  for (let column = 0; column < columns; column += 1) {
    const label = String(values[0][column] ?? "").toLowerCase();
    const columnRange = sheet.getRange(`${columnLabel(column)}:${columnLabel(column)}`);
    if (label.includes("json") || label.includes("features") || label.includes("warning") || label.includes("method") || label.includes("source")) {
      columnRange.format.columnWidth = 38;
      columnRange.format.wrapText = true;
    } else if (label.includes("model") || label.includes("variable") || label.includes("partition") || label.includes("scenario") || label.includes("operating")) {
      columnRange.format.columnWidth = 24;
    } else {
      columnRange.format.columnWidth = 15;
    }
  }
  sheet.tables.add(`A1:${lastColumn}${rows}`, true, `MCI_Table_${String(index + 1).padStart(2, "0")}`);
  catalog.push({ sheetName, rows, columns });
}

const summaryInspect = await workbook.inspect({
  kind: "table",
  range: "README!A1:C31",
  include: "values,formulas",
  tableMaxRows: 31,
  tableMaxCols: 3,
  maxChars: 6000,
});
console.log(summaryInspect.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

const previewDir = path.join(packageDir, "qc_workbook_previews");
await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["README", ...catalog.map((item) => item.sheetName)]) {
  const preview = await workbook.render({
    sheetName,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = sheetName.replace(/[^A-Za-z0-9]+/g, "_");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const outputPath = path.join(packageDir, "MCI_Manuscript_Tables.xlsx");
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: catalog.length + 1, renderedSheets: catalog.length + 1 }));
