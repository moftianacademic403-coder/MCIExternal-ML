param(
    [int]$SelectionPid = 0,
    [string]$PythonPath = '',
    [string]$DevelopmentPath = '',
    [string]$ExternalPath = '',
    [string]$ExternalEducationPath = ''
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = if ($PythonPath) { $PythonPath } else { Join-Path $ProjectRoot '.conda-env\python.exe' }
$Development = if ($DevelopmentPath) { $DevelopmentPath } else { Join-Path $ProjectRoot 'Developement.csv' }
$External = if ($ExternalPath) { $ExternalPath } else { Join-Path $ProjectRoot 'External.xlsx' }
$ExternalEducation = if ($ExternalEducationPath) { $ExternalEducationPath } else { Join-Path $ProjectRoot 'phas3_DR.Moftian.xlsx' }
$HeavyRoot = Join-Path $ProjectRoot 'outputs\workstation_heavy_four_level_final'
$PosthocRoot = Join-Path $ProjectRoot 'outputs\workstation_posthoc_four_level_final\analysis'
$ManuscriptRoot = Join-Path $ProjectRoot 'outputs\manuscript_four_level_final'
$PublicationRoot = Join-Path $ProjectRoot 'outputs\publication_figure_pack'
$LogPath = Join-Path $HeavyRoot 'local_pipeline_runner.log'
$StatusPath = Join-Path $HeavyRoot 'local_pipeline_status.json'

foreach ($requiredPath in @($Python, $Development, $External, $ExternalEducation)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required local file was not found: $requiredPath"
    }
}

New-Item -ItemType Directory -Force -Path $HeavyRoot | Out-Null
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUNBUFFERED = '1'
$env:HF_HUB_DISABLE_XET = '1'

function Write-RunnerLog([string]$Message) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -LiteralPath $LogPath -Value $line -Encoding UTF8
}

function Write-RunnerStatus([string]$Status, [string]$Stage, [string]$Message) {
    [ordered]@{
        status = $Status
        stage = $Stage
        message = $Message
        updated_at = (Get-Date).ToString('o')
        selection_pid = $SelectionPid
    } | ConvertTo-Json | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Assert-ManifestStatus([string]$Path, [string]$Expected) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required manifest was not created: $Path"
    }
    $manifest = Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
    if ($manifest.status -ne $Expected) {
        throw "Unexpected manifest status '$($manifest.status)'; expected '$Expected': $Path"
    }
}

function Invoke-PythonStage([string]$Name, [string[]]$Arguments) {
    Write-RunnerLog "START $Name"
    Write-RunnerStatus 'running' $Name "Running $Name"
    $stdoutPath = Join-Path $HeavyRoot ("{0}.stdout.log" -f $Name)
    $stderrPath = Join-Path $HeavyRoot ("{0}.stderr.log" -f $Name)
    $stageProcess = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot `
        -NoNewWindow `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath
    if ($stageProcess.ExitCode -ne 0) {
        $errorTail = ''
        if (Test-Path -LiteralPath $stderrPath) {
            $errorTail = (Get-Content -LiteralPath $stderrPath -Tail 15) -join ' | '
        }
        throw "$Name failed with exit code $($stageProcess.ExitCode). $errorTail"
    }
    Write-RunnerLog "LOGS $Name stdout=$stdoutPath stderr=$stderrPath"
    Write-RunnerLog "COMPLETE $Name"
}

try {
    $selectionManifest = Join-Path $HeavyRoot 'selection\nested_cv_manifest.json'
    $selectionAlreadyComplete = $false
    if (Test-Path -LiteralPath $selectionManifest) {
        $existingSelection = Get-Content -LiteralPath $selectionManifest -Raw | ConvertFrom-Json
        $selectionAlreadyComplete = (
            $existingSelection.status -eq 'development_only_nested_cv_model_family_selected'
        )
    }

    if ($selectionAlreadyComplete) {
        Write-RunnerLog 'RESUME nested_cv manifest already passed'
    }
    else {
        if ($SelectionPid -gt 0) {
            Write-RunnerLog "Runner attached to existing nested-CV PID $SelectionPid"
            Write-RunnerStatus 'running' 'nested_cv' "Waiting for existing nested-CV PID $SelectionPid"
            $selectionProcess = [System.Diagnostics.Process]::GetProcessById($SelectionPid)
            $selectionProcess.WaitForExit()
        }
        else {
            Invoke-PythonStage 'nested_cv' @(
                'src\heavy_nested_cv.py',
                '--development', $Development,
                '--external', $External,
                '--external-education', $ExternalEducation,
                '--qc-output', 'outputs\workstation_light_four_level\qc',
                '--output', 'outputs\workstation_heavy_four_level_final\selection'
            )
        }
    }

    Assert-ManifestStatus `
        $selectionManifest `
        'development_only_nested_cv_model_family_selected'
    Write-RunnerLog 'GATE PASS nested_cv'

    $finalManifest = Join-Path $HeavyRoot 'final_evaluation\final_evaluation_manifest.json'
    $finalAlreadyComplete = $false
    if (Test-Path -LiteralPath $finalManifest) {
        $existingFinal = Get-Content -LiteralPath $finalManifest -Raw | ConvertFrom-Json
        $finalAlreadyComplete = (
            $existingFinal.status -eq 'manuscript_grade_locked_evaluation_completed'
        )
    }
    if ($finalAlreadyComplete) {
        Write-RunnerLog 'RESUME final_evaluation manifest already passed'
    }
    else {
        Invoke-PythonStage 'final_evaluation' @(
            'src\heavy_final_evaluation.py',
            '--development', $Development,
            '--external', $External,
            '--external-education', $ExternalEducation,
            '--qc-output', 'outputs\workstation_light_four_level\qc',
            '--selection-output', 'outputs\workstation_heavy_four_level_final\selection',
            '--output', 'outputs\workstation_heavy_four_level_final\final_evaluation'
        )
    }
    Assert-ManifestStatus `
        $finalManifest `
        'manuscript_grade_locked_evaluation_completed'
    Write-RunnerLog 'GATE PASS final_evaluation'

    $posthocManifest = Join-Path $PosthocRoot 'posthoc_manifest.json'
    $posthocAlreadyComplete = $false
    if (Test-Path -LiteralPath $posthocManifest) {
        $existingPosthoc = Get-Content -LiteralPath $posthocManifest -Raw | ConvertFrom-Json
        $posthocAlreadyComplete = (
            $existingPosthoc.status -eq 'posthoc_sensitivity_transportability_and_interpretability_completed'
        )
    }
    if ($posthocAlreadyComplete) {
        Write-RunnerLog 'RESUME posthoc_analysis manifest already passed'
    }
    else {
        Invoke-PythonStage 'posthoc_analysis' @(
            'src\heavy_posthoc_analysis.py',
            '--development', $Development,
            '--external', $External,
            '--external-education', $ExternalEducation,
            '--qc-output', 'outputs\workstation_light_four_level\qc',
            '--prior-output', 'outputs\workstation_heavy_four_level_final',
            '--output', 'outputs\workstation_posthoc_four_level_final\analysis'
        )
    }
    Assert-ManifestStatus `
        $posthocManifest `
        'posthoc_sensitivity_transportability_and_interpretability_completed'
    Write-RunnerLog 'GATE PASS posthoc_analysis'

    $manuscriptManifest = Join-Path $ManuscriptRoot 'manuscript_readiness_manifest.json'
    $manuscriptAlreadyComplete = $false
    if (Test-Path -LiteralPath $manuscriptManifest) {
        $existingManuscript = Get-Content -LiteralPath $manuscriptManifest -Raw | ConvertFrom-Json
        $manuscriptAlreadyComplete = (
            $existingManuscript.status -eq 'ready_for_manuscript_writing'
        )
    }
    if ($manuscriptAlreadyComplete) {
        Write-RunnerLog 'RESUME manuscript_artifacts manifest already passed'
    }
    else {
        Invoke-PythonStage 'manuscript_artifacts' @(
            'src\manuscript_artifacts.py',
            '--development', $Development,
            '--external', $External,
            '--external-education', $ExternalEducation,
            '--heavy-output', 'outputs\workstation_heavy_four_level_final',
            '--posthoc-output', 'outputs\workstation_posthoc_four_level_final\analysis',
            '--output', 'outputs\manuscript_four_level_final'
        )
    }
    Assert-ManifestStatus `
        $manuscriptManifest `
        'ready_for_manuscript_writing'
    Write-RunnerLog 'GATE PASS manuscript_artifacts'

    $publicationManifest = Join-Path $PublicationRoot 'publication_figure_manifest.json'
    $publicationAlreadyComplete = $false
    if (Test-Path -LiteralPath $publicationManifest) {
        $existingPublication = Get-Content -LiteralPath $publicationManifest -Raw | ConvertFrom-Json
        $publicationAlreadyComplete = (
            $existingPublication.status -eq 'publication_figure_pack_completed'
        )
    }
    if ($publicationAlreadyComplete) {
        Write-RunnerLog 'RESUME publication_figure_pack manifest already passed'
    }
    else {
        Invoke-PythonStage 'publication_figure_pack' @(
            'src\build_publication_figure_pack.py',
            '--development', $Development,
            '--external', $External,
            '--external-education', $ExternalEducation,
            '--qc-output', 'outputs\workstation_light_four_level\qc',
            '--prior-output', 'outputs\workstation_heavy_four_level_final',
            '--posthoc-output', 'outputs\workstation_posthoc_four_level_final',
            '--output', 'outputs\publication_figure_pack'
        )
    }
    Assert-ManifestStatus `
        $publicationManifest `
        'publication_figure_pack_completed'
    Write-RunnerLog 'GATE PASS publication_figure_pack'
    Write-RunnerStatus 'completed' 'all' 'All local analysis stages completed successfully.'
    Write-RunnerLog 'ALL LOCAL PIPELINE STAGES COMPLETED'
    exit 0
}
catch {
    Write-RunnerLog "FAILED $($_.Exception.Message)"
    Write-RunnerStatus 'failed' 'pipeline' $_.Exception.Message
    exit 1
}
