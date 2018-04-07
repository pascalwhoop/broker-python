#!/bin/bash
COUNTER=1
echo "Watching file tree for changed files. Running tests when it occurs"
python -m unittest
inotifywait -rm --exclude "tensorboard\/|log\/|\.idea\/|py~|__pycache__|venv\/|swp|git\/" -e close_write ./ | while read change; do
    echo "change detected"
    python -m unittest
    COUNTER=$[$COUNTER +1]
    echo "run $COUNTER"
    echo $change
done
