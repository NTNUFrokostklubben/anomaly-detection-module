<#
.SYNOPSIS
    Fetches the SOSI driver bundle from GitHub Packages and extracts to lib/bundle.

.DESCRIPTION
    Pulls a pinned version of the SOSI driver bundle from ghcr.io using the OCI
    distribution API (curl only, no ORAS required). Verifies SHA256 and extracts.
#>

$Version = "1.0.0"
$ExpectedSHA256 = "REPLACE_WITH_ACTUAL_SHA256"

$ErrorActionPreference = "Stop"

$image = "ntnufrokostklubben/sosi-driver-bundle"
$libDir = Join-Path (Join-Path $PSScriptRoot "..") "lib"
$bundleDir = Join-Path $libDir "bundle"
$versionMarker = Join-Path $libDir ".version"

# Check if already up to date
if (Test-Path $versionMarker) {
    $installed = (Get-Content $versionMarker).Trim()
    if ($installed -eq $Version) {
        Write-Host "SOSI bundle $Version already up to date."
        exit 0
    }
}

# Remove existing and re-download
if (Test-Path $bundleDir) {
    Remove-Item $bundleDir -Recurse -Force
}
if (-not (Test-Path $libDir)) {
    New-Item -ItemType Directory -Path $libDir | Out-Null
}

# Step 1: Get anonymous pull token
Write-Host "Downloading SOSI bundle $Version..."
$tokenResponse = curl.exe -s "https://ghcr.io/token?service=ghcr.io&scope=repository:${image}:pull" | ConvertFrom-Json
$token = $tokenResponse.token

# Step 2: Fetch manifest to get the layer digest
$manifest = curl.exe -s `
    -H "Authorization: Bearer $token" `
    -H "Accept: application/vnd.oci.image.manifest.v1+json" `
    "https://ghcr.io/v2/${image}/manifests/${Version}" | ConvertFrom-Json

$layerDigest = $manifest.layers[0].digest
if (-not $layerDigest) {
    Write-Error "Failed to resolve layer digest from manifest."
    exit 1
}

# Step 3: Download the blob
$zipPath = Join-Path $env:TEMP "sosi-driver-bundle.zip"
curl.exe -L `
    -H "Authorization: Bearer $token" `
    -o $zipPath `
    "https://ghcr.io/v2/${image}/blobs/${layerDigest}"

if (-not (Test-Path $zipPath)) {
    Write-Error "Download failed."
    exit 1
}

# Verify SHA256
$actualHash = (Get-FileHash -Algorithm SHA256 $zipPath).Hash
if ($ExpectedSHA256 -ne "REPLACE_WITH_ACTUAL_SHA256" -and $actualHash -ne $ExpectedSHA256) {
    Write-Error "SHA256 mismatch!`nExpected: $ExpectedSHA256`nGot:      $actualHash"
    Remove-Item $zipPath -Force
    exit 1
}

if ($ExpectedSHA256 -eq "REPLACE_WITH_ACTUAL_SHA256") {
    Write-Warning "No expected SHA256 configured. Actual hash: $actualHash"
    Write-Warning "Update `$ExpectedSHA256 at the top of this script to enable verification."
}

# Extract into lib/ — the zip contains a bundle/ folder
Expand-Archive -Path $zipPath -DestinationPath $libDir -Force
Remove-Item $zipPath -Force

Set-Content -Path $versionMarker -Value $Version

Write-Host "Done. SOSI bundle $Version ready."