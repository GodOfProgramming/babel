#!/usr/bin/env bash

curl -X POST 'http://localhost:8080/translate' -H 'Content-Type: application/json' -d "{ \"text\": \"$2\", \"n\": $1 }"
