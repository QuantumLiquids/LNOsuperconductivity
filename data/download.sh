echo "Starting first rsync"
#rsync -av "Lone:/home1/10540/haoxinwang358/LNOsuperconductivity/data/*" .
echo "Finished first rsync"
echo "Starting second rsync"
rsync -av "cent:/home/gzcgu/haoxinwang/LNOsuperconductivity/data/*" .
echo "Finished second rsync"
echo "Starting third rsync"
#rsync -av "susphy:/share/home/wanghx/LNOsuperconductivity/data/*" .
echo "Finished third rsync"

rsync -av  qsc:/share/home/wanghx/LNOsuperconductivity/data/ .
rsync -av  huangwen:/work/iqse-huangw/LNOsuperconductivity/data/ .
