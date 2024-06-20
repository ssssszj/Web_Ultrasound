#!/usr/bin/env bash

rm -rf ../../data/Bezier/*
rm -rf ../../data/Beziermask/*

echo "Drawing bezier curves..."
python3 bezier_curve.py

echo "Generating masks..."
python3 mask_bezier.py

echo "Augumenting..."
python3 augmentation.py

echo "Making additional annotations..."
python3 make_annotations.py