# Run in an Administrator PowerShell on L1 after confirming the cable and subnet.
# This is deliberately specific to the verified Strix built-in Ethernet adapter.
#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"
$adapter = Get-NetAdapter -Name "Ethernet"
if ($adapter.InterfaceDescription -ne "Realtek PCIe GbE Family Controller") {
    throw "Ethernet is not the verified Strix adapter. Inspect adapters before configuring."
}
$addresses = @(Get-NetIPAddress -InterfaceAlias "Ethernet" -AddressFamily IPv4)
$unexpected = @($addresses | Where-Object {
    $_.IPAddress -notlike "169.254.*" -and $_.IPAddress -ne "192.168.77.1"
})
if ($unexpected.Count -gt 0) { throw "Ethernet already has another address; inspect it first." }
if (Get-NetRoute -InterfaceAlias "Ethernet" -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue) {
    throw "Ethernet has a default route; inspect it before changing anything."
}
if (-not ($addresses | Where-Object { $_.IPAddress -eq "192.168.77.1" })) {
    # New-NetIPAddress disables DHCP automatically; intentionally omit a gateway.
    New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress "192.168.77.1" -PrefixLength 24
}
$existing = Get-NetFirewallRule -Name "Jarvis-Whisper-Direct-Link" -ErrorAction SilentlyContinue
if ($existing) {
    throw "Jarvis firewall rule already exists; verify its scope before changing it."
}
New-NetFirewallRule -Name "Jarvis-Whisper-Direct-Link" -DisplayName "Jarvis Whisper (direct link only)" `
    -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 `
    -InterfaceAlias "Ethernet" -LocalAddress "192.168.77.1" -RemoteAddress "192.168.77.2" -Profile Any
Get-NetIPConfiguration -InterfaceAlias "Ethernet"
Get-NetRoute -DestinationPrefix "0.0.0.0/0"
