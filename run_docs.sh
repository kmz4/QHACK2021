#!/bin/bash
pdoc --force --html --output-dir ./docs qose
cd ./docs/qose
mv * ../
cd ..
rm -r ./qose/