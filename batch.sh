#!/usr/bin/env bash

curl -X POST "http://localhost:8080/translate/batch" -H "Content-Type: application/x-yaml" --data-binary "@$1"
