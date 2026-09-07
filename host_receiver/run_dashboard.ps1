param(
    [string]$VsnPort = "COM7",
    [string]$AsnPort = "COM8",
    [int]$HttpPort = 8765
)

python "$PSScriptRoot\demo_dashboard.py" `
    --port $VsnPort `
    --asn-debug-port $AsnPort `
    --http-port $HttpPort
