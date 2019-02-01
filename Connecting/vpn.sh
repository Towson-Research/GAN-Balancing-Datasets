#!/bin/bash
# Connects to towsons tu-secure

sudo /sbin/modprobe tun&&sudo openconnect $1
# sudo apt-get install openconnect
