#!/bin/bash
# Connects to titanx server

ssh -L 8888:localhost:8888 -L 3306:localhost:3306 ldeng@10.55.17.240
