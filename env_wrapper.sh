#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export JAVA_HOME="$SCRIPT_DIR/work/jdk/Contents/Home"
export ELAN_HOME="$SCRIPT_DIR/work/.elan"
export PATH="$JAVA_HOME/bin:$ELAN_HOME/bin:$PATH"
export LAKE_OFFLINE=1
export ELAN_OFFLINE=1
exec "$@"
