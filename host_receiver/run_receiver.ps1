param(
    [string]$VsnPort = "COM7",
    [string]$AsnDebugPort = "COM8"
)

$arguments = @("$PSScriptRoot\receiver.py", "--port", $VsnPort, "--interactive")
if ($AsnDebugPort) {
    $arguments += @("--asn-debug-port", $AsnDebugPort)
}
python @arguments
