# NDHS Flower confirmatory runs: continue from Run 16 through Run 40
# Run this from ...\NDHS_FL_Research\09_flower_implementation_v1_2_1
# Runs sequentially, audits each manifest, checks matched initial hashes,
# and stops safely at the first failure. Existing output folders are never overwritten.

$ErrorActionPreference = "Stop"
$packageRoot = (Get-Location).Path
$configRoot = Join-Path $packageRoot "flower_confirmatory_package_v2_seeds_42_51\configs"
$outputRoot = Join-Path $packageRoot "outputs\confirmatory_v2_seeds_42_51"

if (-not (Test-Path (Join-Path $packageRoot "pyproject.toml"))) {
    throw "Wrong working directory. Open PowerShell in 09_flower_implementation_v1_2_1."
}
if (-not (Test-Path $configRoot)) { throw "Missing config folder: $configRoot" }
if (-not (Test-Path $outputRoot)) { throw "Missing output folder: $outputRoot" }

$regionCounts = @(2304, 2698, 3767, 1402, 1231, 1449)
$pairedInitialHashes = @{
    "baseline|45" = "d34739a26b807b987ee3c097be0762540c90678c04edc7a6adbd442f7fc8ba13"
    "expanded|45" = "72fdb07bbd7bb01a1ddd19614a72184cdfbe26bc9def4b285b1a69e2a2ebe76d"
}

# Run 16 first, then seeds 46-51 in the frozen order.
$runSpecs = @([pscustomobject]@{ Feature="expanded"; Algorithm="fedprox"; Seed=45 })
foreach ($seed in 46..51) {
    $runSpecs += [pscustomobject]@{ Feature="baseline"; Algorithm="fedavg";  Seed=$seed }
    $runSpecs += [pscustomobject]@{ Feature="baseline"; Algorithm="fedprox"; Seed=$seed }
    $runSpecs += [pscustomobject]@{ Feature="expanded"; Algorithm="fedavg";  Seed=$seed }
    $runSpecs += [pscustomobject]@{ Feature="expanded"; Algorithm="fedprox"; Seed=$seed }
}

$logPath = Join-Path $outputRoot ("overnight_runs_16_to_40_{0}.txt" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
Start-Transcript -Path $logPath
try {
    Write-Host "Starting $($runSpecs.Count) sequential runs. Keep this PowerShell window open."
    $completed = 0

    foreach ($spec in $runSpecs) {
        $feature = $spec.Feature
        $algorithm = $spec.Algorithm
        $seed = [int]$spec.Seed
        $stem = "{0}_{1}_e5_r10_s{2}" -f $feature, $algorithm, $seed
        $configPath = Join-Path $configRoot ($stem + ".toml")
        $runOutput = Join-Path $outputRoot $stem
        $manifestPath = Join-Path $runOutput "run_manifest.json"

        Write-Host "`n========== RUN: $stem ==========" -ForegroundColor Yellow
        if (-not (Test-Path $configPath)) { throw "Missing config: $configPath" }
        if (Test-Path $runOutput) { throw "Output folder already exists; refusing to overwrite: $runOutput" }

        & flwr run . "local-simulation" --run-config $configPath --stream
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) { throw "Flower exited with code $exitCode for $stem. Inspect before retrying." }
        if (-not (Test-Path $manifestPath)) { throw "Manifest missing after run: $manifestPath" }

        $m = Get-Content $manifestPath -Raw | ConvertFrom-Json -AsHashtable
        $c = $m["configuration"]
        if ($m["status"] -ne "completed" -or $null -ne $m["failure_reason"]) { throw "$stem did not complete cleanly." }
        if ($m["implementation_version"] -ne "1.2.1") { throw "$stem implementation version mismatch." }
        if ($c["feature_config"] -ne $feature -or $c["algorithm"] -ne $algorithm -or [int]$c["seed"] -ne $seed) { throw "$stem feature/algorithm/seed mismatch." }
        if ([int]$c["local_epochs"] -ne 5 -or [int]$c["num_server_rounds"] -ne 10) { throw "$stem epoch/round mismatch." }
        if ([int]$c["batch_size"] -ne 32 -or [double]$c["learning_rate"] -ne 0.001) { throw "$stem batch-size/learning-rate mismatch." }
        $mu = if ($algorithm -eq "fedavg") { 0.0 } else { 0.01 }
        if ([double]$c["proximal_mu"] -ne $mu) { throw "$stem proximal-mu mismatch." }
        $dim = if ($feature -eq "baseline") { 15 } else { 51 }
        if ([int]$c["input_dim"] -ne $dim) { throw "$stem input dimension mismatch." }
        if ($c.ContainsKey("test_data")) { throw "$stem unexpectedly references test_data." }
        if ([int]$m["validation_rows"] -ne 77) { throw "$stem validation row count is not 77." }

        $rounds = @($m["round_audit"])
        if ($rounds.Count -ne 10) { throw "$stem does not have 10 audited rounds." }
        foreach ($r in $rounds) {
            if ([int]$r["clients_expected"] -ne 6 -or [int]$r["clients_replied"] -ne 6) { throw "$stem round $($r['server_round']) did not receive six clients." }
            $clients = @($r["client_participation"] | Sort-Object { [int]$_.region_code })
            if ($clients.Count -ne 6) { throw "$stem round $($r['server_round']) has incorrect client count." }
            for ($i=0; $i -lt 6; $i++) {
                if ([int]$clients[$i]["region_code"] -ne ($i+1)) { throw "$stem region mapping mismatch." }
                if ([int]$clients[$i]["partition_id"] -ne $i) { throw "$stem partition mapping mismatch." }
                if ([int]$clients[$i]["num_examples"] -ne $regionCounts[$i]) { throw "$stem regional sample count mismatch." }
                if ([int]$clients[$i]["unique_source_ids"] -ne $regionCounts[$i]) { throw "$stem unique-ID count mismatch." }
            }
        }

        $pairKey = "$feature|$seed"
        $initialHash = [string]$m["initial_parameters_sha256"]
        if ([string]::IsNullOrWhiteSpace($initialHash)) { throw "$stem initial hash missing." }
        if ($algorithm -eq "fedavg") {
            $pairedInitialHashes[$pairKey] = $initialHash
        } else {
            if (-not $pairedInitialHashes.ContainsKey($pairKey)) { throw "No FedAvg initial hash stored for $pairKey." }
            if ($pairedInitialHashes[$pairKey] -ne $initialHash) { throw "$stem initial hash does not match its FedAvg pair." }
        }

        $completed++
        Write-Host "AUDIT PASSED: $stem ($completed/$($runSpecs.Count))" -ForegroundColor Green
        Write-Host "Initial hash: $initialHash"
        Write-Host "Final hash:   $($m['final_parameters_sha256'])"
    }

    Write-Host "`nALL 25 RUNS COMPLETED AND MANIFEST AUDITS PASSED." -ForegroundColor Green
}
catch {
    Write-Host "`nSTOPPED SAFELY: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Inspect the current run folder and manifest before attempting any retry."
    throw
}
finally {
    Stop-Transcript
}
Write-Host "Run log: $logPath"
