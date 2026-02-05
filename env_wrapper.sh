#!/bin/bash
export JAVA_HOME="$(pwd)/work/jdk/Contents/Home"
export ELAN_HOME="$(pwd)/work/.elan"
export PATH="$JAVA_HOME/bin:$ELAN_HOME/bin:$PATH"
exec "$@"
