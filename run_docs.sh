#!/bin/bash
if pdoc --force --html --output-dir ./docs ./qose --skip-errors; then
    cd ./docs/qose
    mv * ../
    cd ..
    rm -r ./qose/
    echo "Docs generated!"
else
    echo "Docs failed!"
fi

