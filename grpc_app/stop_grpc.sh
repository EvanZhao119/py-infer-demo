#!/bin/bash
if [ -f "server.pid" ]; then
    kill $(cat server.pid)
    rm -f server.pid
else
    echo "No process running"
fi
