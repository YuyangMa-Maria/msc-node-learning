param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("vsn", "asn")]
    [string]$Node,

    [Parameter(Mandatory = $true)]
    [ValidateSet("build", "flash", "monitor")]
    [string]$Action,

    [string]$Port = "COM7"
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Join-Path $root $Node
$idf = Get-Command idf.py -ErrorAction SilentlyContinue
if ($null -eq $idf) {
    throw "idf.py is not available. Run this script from an ESP-IDF terminal."
}

Push-Location -LiteralPath $projectDir
try {
    $buildArguments = @("-B", "build")
    if ($Action -eq "build") {
        & $idf @buildArguments build
    } elseif ($Action -eq "flash") {
        & $idf @buildArguments -p $Port flash
    } else {
        & $idf @buildArguments -p $Port monitor
    }
    if ($LASTEXITCODE -ne 0) {
        throw "ESP-IDF action '$Action' failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
