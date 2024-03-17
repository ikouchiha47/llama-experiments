#!/bin/bash
#
# Usage ./pyenv
#       ./pyenv deactivate
#

CMDS="activate deactivate"
CMD="activate"

if [[ ! -z "$1" ]]; then
  CMD="$1"
fi


is_valid_cmd=$( \
  echo "$CMDS" \
  | awk -v pattern="$CMD" '{if ($0 ~ pattern) print NR}' \
)

if [[ $is_valid_cmd -eq 0 ]]; then
  echo "invalid cmd"
  exit 1
fi

case "$CMD" in
  "activate")
    python3 -m venv .venv
    . .venv/bin/activate
    ;;
  "deactivate")
    deactivate
    ;;
  *)
    echo "Invalid Usage"
    ;;
esac
