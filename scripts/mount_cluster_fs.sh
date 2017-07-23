#!/usr/bin/env bash
umount ../biternion_net_cluster_logs
mkdir ../biternion_net_cluster_logs
sshfs  sprokudin@login.cluster.is.localnet:/home/sprokudin/biternionnet/logs ../biternion_net_cluster_logs/