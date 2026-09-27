# Disk read MB/s (all physical disks) + free RAM GB every 5 s, via CIM (typeperf needs the Performance Log Users group).
"time,read_mb_s,free_ram_gb"
while ($true) {
  $d = Get-CimInstance Win32_PerfFormattedData_PerfDisk_PhysicalDisk -Filter "Name='_Total'"
  $os = Get-CimInstance Win32_OperatingSystem
  [string]::Format([Globalization.CultureInfo]::InvariantCulture, "{0},{1:F0},{2:F1}", (Get-Date -Format HH:mm:ss), ($d.DiskReadBytesPersec / 1MB), ($os.FreePhysicalMemory / 1MB))
  Start-Sleep -Seconds 5
}
