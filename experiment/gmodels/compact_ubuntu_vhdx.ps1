# Run in an ADMIN PowerShell. Shrinks Ubuntu-24.04's WSL disk image (ext4.vhdx on C:) to what it actually uses,
# after files were deleted inside WSL (a dynamic VHDX never shrinks on its own). Stops WSL first; nothing inside
# WSL is deleted. Run `fstrim -v /` inside WSL beforehand (already done 2026-09-27).
$vhdx = 'C:\Users\Asus\AppData\Local\wsl\{7c4f3436-36e8-48df-9b6f-9a778aba56fe}\ext4.vhdx'
$before = (Get-Item $vhdx).Length
wsl.exe --shutdown
Start-Sleep -Seconds 5
$dp = Join-Path $env:TEMP 'compact_wsl.txt'
@"
select vdisk file="$vhdx"
attach vdisk readonly
compact vdisk
detach vdisk
"@ | Set-Content -Encoding ascii $dp
diskpart /s $dp
$after = (Get-Item $vhdx).Length
'{0:N1} GB -> {1:N1} GB' -f ($before / 1GB), ($after / 1GB)
