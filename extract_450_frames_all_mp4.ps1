$totalSeconds = 0
Get-ChildItem -Filter *.mp4 | ForEach-Object {
 $file = $_.FullName
 $duration = ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file"
 $totalSeconds += [double]$duration
}

$totalFrames = 450
$fps = $totalFrames / $totalSeconds

Get-ChildItem -Filter *.mp4 | ForEach-Object {
 $baseName = $_.BaseName
 ffmpeg -i $_.FullName -vf fps=$fps "${baseName}-%03d.png"
 }