#!/bin/bash

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ -f "server.pid" ]; then
    PID=$(cat server.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "Stop PID=$PID"
        kill $PID
    fi
    rm -f server.pid
fi

echo "Start server.py..."
nohup python3 server.py > server.log 2>&1 &
echo $! > server.pid

echo "gRPC log output $(pwd)/server.log"

deactivate
